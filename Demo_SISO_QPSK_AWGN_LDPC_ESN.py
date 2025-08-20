import os
import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.sparse as sp  # Added import for scipy.sparse
from pyESN import ESN
from pyldpc import make_ldpc, decode as ldpc_decode, get_message

# --------------------------
# Helpers (from MIMO code, adapted for SISO)
# --------------------------
def unit_qam_constellation(Bi):
    EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2
    PamM = EvenSquareRoot
    PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32).reshape(1, -1)
    SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
    C = SquareMatrix + 1j * (SquareMatrix.T)
    C_tmp = np.zeros(C.shape[0]*C.shape[1], dtype=np.complex128)
    for i in range(C.shape[1]):
        for j in range(C.shape[0]):
            C_tmp[i*C.shape[0] + j] = C[j][i]
    C = C_tmp
    return C / math.sqrt(np.mean(np.abs(C) ** 2))

def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]

def qam_bit_labels(M, m):
    labels = np.zeros((M, m), dtype=int)
    for idx in range(M):
        labels[idx, :] = bits_to_grayvec(idx, m)
    return labels

def ldpc_encode_bits(G, u):
    x = G.dot(u) % 2 if sp.issparse(G) else (G @ u % 2)
    x = np.asarray(x).ravel().astype(np.int8)
    return x

def hard_bits_from_syms(Xhat, Const, m, PowersOfTwo):
    N = len(Xhat)
    RxBits = np.zeros((N*m,), dtype=int)
    for ii in range(N):
        sym = Xhat[ii]
        idx = int(np.argmin(np.abs(Const - sym)))
        RxBits[m*ii:m*(ii+1)] = bits_to_grayvec(idx, m)
    return RxBits

def qam_llrs_maxlog(z_hat_1d, const, bit_labels, sigma2):
    N = z_hat_1d.shape[0]
    M = const.shape[0]
    m = bit_labels.shape[1]
    llrs = np.zeros((N, m), dtype=float)
    masks0 = [bit_labels[:, b] == 0 for b in range(m)]
    masks1 = [bit_labels[:, b] == 1 for b in range(m)]
    for i in range(N):
        d2 = np.abs(z_hat_1d[i] - const)**2
        for b in range(m):
            d0 = d2[masks0[b]]
            d1 = d2[masks1[b]]
            llrs[i, b] = (np.min(d1) - np.min(d0)) / max(sigma2, 1e-12)
    return llrs.reshape(-1)

def est_sigma2_from_decision(Xhat, const):
    idx = np.argmin(np.abs(Xhat.reshape(-1,1) - const.reshape(1,-1))**2, axis=1)
    Xhard = const[idx]
    err = Xhat - Xhard
    return float(np.mean(np.abs(err)**2) + 1e-12)

def add_cp(x, cp): return np.r_[x[-cp:], x] if cp > 0 else x
def rm_cp(x, cp): return x[cp:] if cp > 0 else x

# --------------------------
# Simulation parameters
# --------------------------
np.random.seed(42)

# Channel / noise
No = 1e-5
EbNoDB = np.arange(0, 31, 3).astype(np.int32)

# OFDM / modulation
N = 128
m = 2  # QPSK / 4-QAM
CP = 0
NumOfdmSymbols = 40

# Power normalization
Ptotal = 10**(EbNoDB/10) * No * N
Pi = Ptotal / N

# Constellation
Const = unit_qam_constellation(m)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
BIT_LABELS = qam_bit_labels(2**m, m)

# PA
p_smooth = 1
ClipLeveldB = 3

# ESN
nInputUnits = 2  # SISO: real + imag of single Rx
nOutputUnits = 2  # SISO: real + imag of single Tx
nInternalUnits = 200
inputScaler = 0.005
inputOffset = 0.0
spectralRadius = 0.9
teacherScalingBase = 0.0000005
teacherShift = np.zeros(nOutputUnits)
feedbackScaling = np.zeros(nOutputUnits)

# LDPC
USE_LDPC = True
LDPC_dv = 4
LDPC_dc = 8
LDPC_MAXITER = 100
LLR_SCALE = 1.5
LLR_CLIP = 20.0
n_code = N * m
if n_code % LDPC_dc != 0:
    raise ValueError(f"LDPC_dc={LDPC_dc} must divide n_code={n_code}.")
H, G = make_ldpc(n_code, LDPC_dv, LDPC_dc, systematic=True, sparse=True)
k_info = G.shape[1]
print(f"[LDPC] Built regular code: n={n_code}, k≈{k_info}, rate≈{k_info/n_code:.3f}")

# Decode only every k-th symbol
LDPC_DECODE_EVERY = 4

# Holders
BER_ESN = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))
BERC_ESN = np.zeros(len(EbNoDB))
BERC_MMSE = np.zeros(len(EbNoDB))

# Output directory
outdir = "./results_siso_qpsk_awgn_ldpc"
os.makedirs(outdir, exist_ok=True)

t_start_total = time.time()

# --------------------------
# Main SNR loop
# --------------------------
for jj, ebno_db in enumerate(EbNoDB):
    print(f"\n=== SNR {ebno_db} dB ===")
    t_snr = time.time()
    DEC_MAXITER = LDPC_MAXITER if ebno_db >= 6 else 2*LDPC_MAXITER

    var_x = 10**(ebno_db/10) * No * N
    A_Clip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB/20)
    inputScaling = (inputScaler/(var_x**0.5)) * np.ones(nInputUnits)
    inputShift = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    TotalErr_ESN = 0
    TotalErr_MMSE = 0
    TotalBits = 0
    TotalErrC_ESN = 0
    TotalErrC_MMSE = 0
    TotalInfoBits = 0

    esn = None

    for kk in range(1, NumOfdmSymbols+1):
        if kk % 20 == 0:
            print(f"  symbol {kk}/{NumOfdmSymbols}", flush=True)

        decode_this_symbol = USE_LDPC and ((kk % LDPC_DECODE_EVERY) == 1)

        # Pilot symbol for ESN training (every symbol for simplicity)
        if kk == 1:
            X = Const[np.random.randint(0, 2**m, size=N)]
            x_td = N * np.fft.ifft(X)
            x_cp = add_cp(x_td, CP) * math.sqrt(Pi[jj])
            x_cp_nld = x_cp / ((1 + (np.abs(x_cp)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            y_td = signal.lfilter([1.0 + 0j], [1], x_cp_nld)
            noise = math.sqrt(len(y_td)*No/2) * (np.random.randn(len(y_td)) + 1j*np.random.randn(len(y_td)))
            y_td += noise

            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                      spectral_radius=spectralRadius, sparsity=0.1,
                      input_shift=inputShift, input_scaling=inputScaling,
                      teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                      feedback_scaling=feedbackScaling)
            Ein = np.column_stack([y_td.real, y_td.imag])
            Eout = np.column_stack([x_cp.real, x_cp.imag])
            esn.fit(Ein, Eout)

        # Data symbol
        if USE_LDPC:
            u = np.random.randint(0, 2, size=(k_info,), dtype=np.int8)
            TxBits = ldpc_encode_bits(G, u)
            X = np.zeros(N, dtype=complex)
            for ii in range(N):
                idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m)])[0])
                X[ii] = Const[idx]
        else:
            TxBits = (np.random.rand(N*m) > 0.5).astype(np.int8)
            X = np.zeros(N, dtype=complex)
            for ii in range(N):
                idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m)])[0])
                X[ii] = Const[idx]

        x_td = N * np.fft.ifft(X)
        x_cp = add_cp(x_td, CP) * math.sqrt(Pi[jj])
        x_cp_nld = x_cp / ((1 + (np.abs(x_cp)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
        y_td = signal.lfilter([1.0 + 0j], [1], x_cp_nld)
        noise = math.sqrt(len(y_td)*No/2) * (np.random.randn(len(y_td)) + 1j*np.random.randn(len(y_td)))
        y_td += noise

        Y = (1/N) * np.fft.fft(rm_cp(y_td, CP))

        # ESN inference
        ESN_input = np.column_stack([y_td.real, y_td.imag])
        x_hat_tmp = esn.predict(ESN_input)
        x_hat_td = x_hat_tmp[:,0] + 1j*x_hat_tmp[:,1]
        X_hat_ESN = (1/N) * np.fft.fft(rm_cp(x_hat_td, CP)) / math.sqrt(Pi[jj])

        # MMSE (AWGN perfect)
        X_hat_MMSE = Y / math.sqrt(Pi[jj])

        # Pre-decoder BER
        RxBits_ESN = hard_bits_from_syms(X_hat_ESN, Const, m, PowersOfTwo)
        RxBits_MMSE = hard_bits_from_syms(X_hat_MMSE, Const, m, PowersOfTwo)
        TotalErr_ESN += int(np.sum(TxBits != RxBits_ESN))
        TotalErr_MMSE += int(np.sum(TxBits != RxBits_MMSE))
        TotalBits += N * m

        # LDPC decoding
        if USE_LDPC and decode_this_symbol:
            sigma2 = No  # True noise variance for AWGN
            llr_esn = qam_llrs_maxlog(X_hat_ESN, Const, BIT_LABELS, sigma2) * LLR_SCALE
            llr_mmse = qam_llrs_maxlog(X_hat_MMSE, Const, BIT_LABELS, sigma2) * LLR_SCALE
            y_obs_esn = np.clip(llr_esn, -LLR_CLIP, LLR_CLIP)
            y_obs_mmse = np.clip(llr_mmse, -LLR_CLIP, LLR_CLIP)
            d_esn = ldpc_decode(H, y_obs_esn, snr=1.0, maxiter=DEC_MAXITER)
            d_mmse = ldpc_decode(H, y_obs_mmse, snr=1.0, maxiter=DEC_MAXITER)
            u_hat_esn = get_message(G, d_esn).astype(np.int8)
            u_hat_mmse = get_message(G, d_mmse).astype(np.int8)
            TotalErrC_ESN += int(np.sum(u != u_hat_esn))
            TotalErrC_MMSE += int(np.sum(u != u_hat_mmse))
            TotalInfoBits += len(u)

    # Per-SNR aggregate
    BER_ESN[jj] = TotalErr_ESN / max(TotalBits, 1)
    BER_MMSE[jj] = TotalErr_MMSE / max(TotalBits, 1)
    if USE_LDPC and TotalInfoBits > 0:
        BERC_ESN[jj] = TotalErrC_ESN / TotalInfoBits
        BERC_MMSE[jj] = TotalErrC_MMSE / TotalInfoBits

    print(f"  -> SNR {ebno_db} dB done in {time.time()-t_snr:.1f}s")

# --------------------------
# Save & Plot
# --------------------------
results = {
    "EBN0": EbNoDB.tolist(),
    "uncoded": {"ESN": BER_ESN.tolist(), "MMSE": BER_MMSE.tolist()},
    "coded": {"ESN": BERC_ESN.tolist(), "MMSE": BERC_MMSE.tolist()}
}
with open(f"{outdir}/results_siso_qpsk_awgn.pkl", "wb") as f:
    pickle.dump(results, f)

plt.figure()
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE (pre-LDPC)')
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN (pre-LDPC)')
plt.legend(); plt.grid(True, which='both', ls=':')
plt.title('SISO QPSK AWGN | Pre-LDPC BER')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.savefig(f"{outdir}/BER_preLDPC_siso_qpsk.png", dpi=150)
plt.show()

if USE_LDPC and np.any(BERC_ESN > 0):
    plt.figure()
    plt.semilogy(EbNoDB, BERC_MMSE, 'rs-.', label='MMSE (post-LDPC)')
    plt.semilogy(EbNoDB, BERC_ESN, 'gd--', label='ESN (post-LDPC)')
    plt.grid(True, which='both', ls=':'); plt.legend()
    plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER (info bits)')
    plt.title('SISO QPSK AWGN | Post-LDPC BER')
    plt.tight_layout()
    plt.savefig(f"{outdir}/BER_postLDPC_siso_qpsk.png", dpi=150)
    plt.show()

print(f"\nTotal run time: {time.time()-t_start_total:.1f}s")
print("Saved figures to:")
print(f" - {outdir}/BER_preLDPC_siso_qpsk.png")
print(f" - {outdir}/BER_postLDPC_siso_qpsk.png")
print(f" - {outdir}/results_siso_qpsk_awgn.pkl")