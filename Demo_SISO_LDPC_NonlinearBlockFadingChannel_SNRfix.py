import numpy as np
import math
import warnings
from scipy import signal
from HelpFunc import HelpFunc
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
SISO + LDPC Nonlinear Block-Fading OFDM demo
- Leaves pyESN.py and HelpFunc.py untouched.
- Adds LDPC (pyldpc) encode/ decode around OFDM + ESN.
- One LDPC codeword per OFDM symbol: n = m * N.
Requires: pip install pyldpc
"""

# Silence pyldpc non-convergence warnings (expected at very low SNR)
warnings.filterwarnings(
    "ignore",
    message="Decoding stopped before convergence",
    module="pyldpc.decoder",
    category=UserWarning,
)

try:
    from pyldpc import make_ldpc, encode, decode, get_message
except Exception as e:
    raise ImportError("This script requires `pyldpc`. Install with `pip install pyldpc`. Original error: %s" % str(e))

# --------------------
# Physical / design
# --------------------
W = 2*1.024e6
f_D = 100
No = 1e-5
IsiDuration = 8
EbNoDB = np.arange(21, 24+1, 3).astype(np.int32)   # you can start higher (e.g., 6..24) for quicker convergence
N = 256           # subcarriers
m = 4             # 16-QAM
NumOfdmSymbols = 60

# --------------------
# LDPC: one codeword spans one OFDM symbol
# --------------------
n = m * N                    # coded length (must equal bits per OFDM symbol)
target_rate = 0.5
k_init = int(target_rate * n)

def choose_dv_dc(n):
    # Prefer ~1/2 rate; ensure dc | n for pyldpc regular code
    candidates = [(4,8), (3,6), (5,10), (2,4), (6,12)]
    for dv, dc in candidates:
        if n % dc == 0:
            return dv, dc
    for dc in range(4, 129):
        if n % dc == 0:
            dv = max(2, dc//2)
            return dv, dc
    return 4, 8

d_v, d_c = choose_dv_dc(n)
print(f"Building LDPC with n={n}, target k≈{k_init} (target rate~{k_init/n:.3f}), dv={d_v}, dc={d_c} ...")
H, G = make_ldpc(n, d_v, d_c, systematic=True, seed=42)

# pyldpc returns G with shape (n, k) and H with shape (mH, n).
n_actual, k_actual = G.shape[0], G.shape[1]
n, k = int(n_actual), int(k_actual)
print(f"LDPC built: n={n}, k={k}, rate={k/n:.3f}")

if n != m*N:
    raise ValueError(f"LDPC length n={n} must equal m*N={m*N}. Adjust N/m or LDPC such that n==m*N.")

# --------------------
# Antennas (SISO) & OFDM
# --------------------
N_t = 1
N_r = 1
p_smooth = 1
ClipLeveldB = 3
CyclicPrefixLen = IsiDuration - 1
T_OFDM_Total = (N + IsiDuration - 1)/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)    # coherence symbols

# --------------------
# Constellation & helpers
# --------------------
Const = np.array(HelpFunc.UnitQamConstellation(m)).astype(complex)  # length 2^m

def bits_to_sym_index(bits_01):
    b = np.array(bits_01, dtype=int).reshape(-1)
    idx = int(np.dot(b, 2**np.arange(len(b))))  # LSB-first index
    if idx < 0 or idx >= len(Const):
        idx = idx % len(Const)
    return idx

def gray_map_indices_to_bits(idx, m):
    bits = list(format(idx, 'b').zfill(m))
    return np.array([int(i) for i in bits])[::-1]

# Precompute bit-sets for max-log LLR
bit_sets = []
for b in range(m):
    set0, set1 = [], []
    for i, s in enumerate(Const):
        bits = gray_map_indices_to_bits(i, m)
        (set0 if bits[b]==0 else set1).append(s)
    bit_sets.append((np.array(set0), np.array(set1)))

def llr_maxlog_16qam(symbols, noise_var):
    syms = symbols.reshape(-1)
    L = np.zeros((len(syms), m))
    for b in range(m):
        S0, S1 = bit_sets[b]
        d0 = np.min(np.abs(syms[:, None] - S0[None, :])**2, axis=1)
        d1 = np.min(np.abs(syms[:, None] - S1[None, :])**2, axis=1)
        L[:, b] = (d1 - d0) / max(noise_var, 1e-12)  # positive => bit=1 more likely
    return L

def llr_to_bpsk_obs(llr):
    """Map bit-LLRs to a pseudo BPSK observation y in [-1,1] for pyldpc.decode(H, y, snr)."""
    llr = np.asarray(llr, dtype=float).reshape(-1)
    return np.tanh(llr/2.0)

def estimate_snr_db_from_syms(Xhat, Const):
    """EVM-based SNR estimate from equalized symbols."""
    Xhat = np.asarray(Xhat).reshape(-1)
    idxs = np.argmin(np.abs(Xhat[:, None] - Const[None, :])**2, axis=1)
    X_near = Const[idxs]
    err = Xhat - X_near
    sig = np.mean(np.abs(X_near)**2)
    noise = np.mean(np.abs(err)**2)
    snr_lin = sig / max(noise, 1e-12)
    return 10.0 * np.log10(max(snr_lin, 1e-12))

# --------------------
# Secondary params & ESN
# --------------------
Ptotal = 10**(EbNoDB/10)*No*N
Pi = Ptotal/N                          # per-subcarrier power
NumBitsPerSymbol = m*N

var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2
nOutputUnits = 2
nInternalUnits = 120
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

teacherShift = np.zeros(nOutputUnits)
feedbackScaling = feedbackScaler*np.ones(nOutputUnits)

Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2
DelayFlag = 0

# Channel profile
temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)

# MMSE TD regularization base
R_h = np.diag(IsiMagnitude[:IsiDuration])

# --------------------
# Accumulators
# --------------------
BER_ESN_uncoded = np.zeros(len(EbNoDB))
BER_LS_uncoded = np.zeros(len(EbNoDB))
BER_MMSE_uncoded = np.zeros(len(EbNoDB))
BER_Perfect_uncoded = np.zeros(len(EbNoDB))

BER_ESN_coded = np.zeros(len(EbNoDB))
BER_LS_coded = np.zeros(len(EbNoDB))
BER_MMSE_coded = np.zeros(len(EbNoDB))
BER_Perfect_coded = np.zeros(len(EbNoDB))

# --------------------
# Simulation
# --------------------
for jj, ebno_db in enumerate(EbNoDB):
    print(f"EbNoDB = {ebno_db}")
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    inputScaling = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    TotalNum_uncoded = 0
    TotalErr_ESN_uncoded = 0
    TotalErr_LS_uncoded = 0
    TotalErr_MMSE_uncoded = 0
    TotalErr_Perfect_uncoded = 0

    TotalNum_coded = 0
    TotalErr_ESN_coded = 0
    TotalErr_LS_coded = 0
    TotalErr_MMSE_coded = 0
    TotalErr_Perfect_coded = 0

    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), (No/Pi[jj])/(N/2)) + np.eye(IsiDuration)

    trainedEsn = None
    Delay = None
    Delay_Min = None
    Delay_Max = None
    nForgetPoints = None

    for kk in range(1, NumOfdmSymbols+1):
        if (np.remainder(kk, L) == 1):
            # New channel
            c0 = (np.random.randn(IsiDuration) + 1j*np.random.randn(IsiDuration))/np.sqrt(2)
            c0 *= np.sqrt(IsiMagnitude[:IsiDuration])

            # Pilot: use a random LDPC codeword for symmetry
            msg = (np.random.rand(k) > 0.5).astype(int)
            cw = encode(G, msg, snr=10.0)
            cw = (cw > 0).astype(int)  # ensure 0/1

            X_p = np.zeros((N,), dtype=complex)
            for ii in range(N):
                bits = cw[m*ii:m*(ii+1)]
                bits = (bits > 0).astype(int)
                idx = bits_to_sym_index(bits)
                X_p[ii] = Const[idx]

            x_temp = N * np.fft.ifft(X_p)
            x_CP_p = np.concatenate([x_temp[-CyclicPrefixLen:], x_temp]) * (Pi[jj]**0.5)

            x_CP_p_NLD = x_CP_p / ((1 + (np.abs(x_CP_p)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP_p = signal.lfilter(c0, np.array([1]), x_CP_p_NLD)
            noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
            y_CP_p = y_CP_p + noise

            Y_p = (1/N) * np.fft.fft(y_CP_p[CyclicPrefixLen:])
            Ci_true = np.fft.fft(np.concatenate([c0, np.zeros(N - len(c0))]))

            H_LS = (Y_p / X_p) / (Pi[jj]**0.5)
            c_LS = np.fft.ifft(H_LS)
            c_LS_trunc = c_LS[:IsiDuration]
            c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS_trunc)
            H_MMSE = np.fft.fft(np.concatenate([c_MMSE, np.zeros(N-IsiDuration)]))

            # Train ESN on pilot
            if DelayFlag:
                Delay_LUT = []
                for d0 in range(Min_Delay, Max_Delay+1):
                    for d1 in range(Min_Delay, Max_Delay+1):
                        Delay_LUT.append([d0, d1])
                Delay_LUT = np.array(Delay_LUT, dtype=int)
            else:
                Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay), 2), dtype=int)
                for d in range(Min_Delay, Max_Delay+1):
                    Delay_LUT[d - Min_Delay, :] = d

            Delay_Max_vec = np.amax(Delay_LUT, axis=1)
            Delay_Min_vec = np.amin(Delay_LUT, axis=1)

            NMSE_list = np.zeros(Delay_LUT.shape[0])

            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                      spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                      input_shift=inputShift, input_scaling=inputScaling,
                      teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                      feedback_scaling=feedbackScaling)

            for didx in range(Delay_LUT.shape[0]):
                d0, d1 = Delay_LUT[didx, :]
                dmax = Delay_Max_vec[didx]
                dmin = Delay_Min_vec[didx]

                ESN_input = np.zeros((N + dmax + CyclicPrefixLen, nInputUnits))
                ESN_output = np.zeros((N + dmax + CyclicPrefixLen, nOutputUnits))

                ESN_input[:, 0] = np.concatenate([y_CP_p.real, np.zeros(dmax)])
                ESN_input[:, 1] = np.concatenate([y_CP_p.imag, np.zeros(dmax)])

                ESN_output[d0:(d0+N+CyclicPrefixLen), 0] = x_CP_p.real
                ESN_output[d1:(d1+N+CyclicPrefixLen), 1] = x_CP_p.imag

                nForgetPoints_tmp = dmin + CyclicPrefixLen
                esn.fit(ESN_input, ESN_output, nForgetPoints_tmp)
                x_hat_tmp = esn.predict(ESN_input, nForgetPoints_tmp, continuation=False)

                x_hat_td = (x_hat_tmp[d0 - dmin : d0 - dmin + N + 1, 0]
                            + 1j * x_hat_tmp[d1 - dmin : d1 - dmin + N + 1, 1])

                x_ref = x_CP_p[CyclicPrefixLen:]
                NMSE = np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2
                NMSE_list[didx] = NMSE

            Delay_Idx = int(np.argmin(NMSE_list))
            Delay = Delay_LUT[Delay_Idx, :]
            Delay_Min = int(Delay_Min_vec[Delay_Idx])
            Delay_Max = int(Delay_Max_vec[Delay_Idx])

            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, nInputUnits))
            ESN_output = np.zeros((N + Delay_Max + CyclicPrefixLen, nOutputUnits))
            ESN_input[:, 0] = np.concatenate([y_CP_p.real, np.zeros(Delay_Max)])
            ESN_input[:, 1] = np.concatenate([y_CP_p.imag, np.zeros(Delay_Max)])
            ESN_output[Delay[0]:(Delay[0]+N+CyclicPrefixLen), 0] = x_CP_p.real
            ESN_output[Delay[1]:(Delay[1]+N+CyclicPrefixLen), 1] = x_CP_p.imag

            nForgetPoints = int(Delay_Min + CyclicPrefixLen)
            esn.fit(ESN_input, ESN_output, nForgetPoints)
            trainedEsn = esn

            H_true = Ci_true
            H_ls = H_LS
            H_mmse = H_MMSE

        else:
            # Data: LDPC encode, map, transmit
            msg = (np.random.rand(k) > 0.5).astype(int)
            cw = encode(G, msg, snr=10.0)
            cw = (cw > 0).astype(int)

            X = np.zeros((N,), dtype=complex)
            for ii in range(N):
                bits = cw[m*ii:m*(ii+1)]
                bits = (bits > 0).astype(int)
                idx = bits_to_sym_index(bits)
                X[ii] = Const[idx]

            x_temp = N * np.fft.ifft(X)
            x_CP = np.concatenate([x_temp[-CyclicPrefixLen:], x_temp]) * (Pi[jj]**0.5)

            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP = signal.lfilter(c0, np.array([1]), x_CP_NLD)
            noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
            y_CP = y_CP + noise

            Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:])

            # ESN detect
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, nInputUnits))
            ESN_input[:, 0] = np.concatenate([y_CP.real, np.zeros(Delay_Max)])
            ESN_input[:, 1] = np.concatenate([y_CP.imag, np.zeros(Delay_Max)])
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, nForgetPoints, continuation=False)
            x_hat_td = (x_hat_ESN_temp[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0]
                        + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1])

            X_hat_ESN = (1/N) * np.fft.fft(x_hat_td) / math.sqrt(Pi[jj])

            # Baselines
            X_hat_Perfect = (Y / H_true) / math.sqrt(Pi[jj])
            X_hat_LS = (Y / H_ls) / math.sqrt(Pi[jj])
            X_hat_MMSE = (Y / H_mmse) / math.sqrt(Pi[jj])

            # Uncoded BER checks
            def hard_bits_from_syms(X_hat):
                RxBits = np.zeros(n, dtype=int)
                for ii in range(N):
                    sym = X_hat[ii]
                    idx = int(np.argmin(np.abs(Const - sym)))
                    bits = gray_map_indices_to_bits(idx, m)
                    RxBits[m*ii:m*(ii+1)] = bits
                return RxBits

            Rx_ESN = hard_bits_from_syms(X_hat_ESN)
            Rx_LS = hard_bits_from_syms(X_hat_LS)
            Rx_MMSE = hard_bits_from_syms(X_hat_MMSE)
            Rx_Perf = hard_bits_from_syms(X_hat_Perfect)

            TotalErr_ESN_uncoded += int(np.sum(cw != Rx_ESN))
            TotalErr_LS_uncoded += int(np.sum(cw != Rx_LS))
            TotalErr_MMSE_uncoded += int(np.sum(cw != Rx_MMSE))
            TotalErr_Perfect_uncoded += int(np.sum(cw != Rx_Perf))
            TotalNum_uncoded += n

            # Soft LLRs & LDPC decode (pyldpc: decode(H, y, snr))
            noise_var = No
            def llrs_for(Xhat):
                L = llr_maxlog_16qam(Xhat, noise_var)
                return L.reshape(-1)

            llr_ESN = llrs_for(X_hat_ESN)
            llr_Perf = llrs_for(X_hat_Perfect)
            llr_LS = llrs_for(X_hat_LS)
            llr_MMSE = llrs_for(X_hat_MMSE)

            # Estimate effective SNR from equalized symbols (per stream) to help decoder
            snr_esn = estimate_snr_db_from_syms(X_hat_ESN, Const)
            snr_perf = estimate_snr_db_from_syms(X_hat_Perfect, Const)
            snr_ls   = estimate_snr_db_from_syms(X_hat_LS, Const)
            snr_mmse = estimate_snr_db_from_syms(X_hat_MMSE, Const)

            y_ESN = llr_to_bpsk_obs(llr_ESN)
            y_Perf = llr_to_bpsk_obs(llr_Perf)
            y_LS = llr_to_bpsk_obs(llr_LS)
            y_MMSE = llr_to_bpsk_obs(llr_MMSE)

            dec_ESN = decode(H, y_ESN, snr=float(snr_esn))
            dec_LS = decode(H, y_LS, snr=float(snr_ls))
            dec_MMSE = decode(H, y_MMSE, snr=float(snr_mmse))
            dec_Perf = decode(H, y_Perf, snr=float(snr_perf))

            msg_ESN = get_message(G, dec_ESN)
            msg_LS = get_message(G, dec_LS)
            msg_MMSE = get_message(G, dec_MMSE)
            msg_Perf = get_message(G, dec_Perf)

            TotalErr_ESN_coded += int(np.sum(msg_ESN != msg))
            TotalErr_LS_coded += int(np.sum(msg_LS != msg))
            TotalErr_MMSE_coded += int(np.sum(msg_MMSE != msg))
            TotalErr_Perfect_coded += int(np.sum(msg_Perf != msg))
            TotalNum_coded += k

    # BERs at this SNR
    BER_ESN_uncoded[jj] = TotalErr_ESN_uncoded / max(TotalNum_uncoded, 1)
    BER_LS_uncoded[jj] = TotalErr_LS_uncoded / max(TotalNum_uncoded, 1)
    BER_MMSE_uncoded[jj] = TotalErr_MMSE_uncoded / max(TotalNum_uncoded, 1)
    BER_Perfect_uncoded[jj] = TotalErr_Perfect_uncoded / max(TotalNum_uncoded, 1)

    BER_ESN_coded[jj] = TotalErr_ESN_coded / max(TotalNum_coded, 1)
    BER_LS_coded[jj] = TotalErr_LS_coded / max(TotalNum_coded, 1)
    BER_MMSE_coded[jj] = TotalErr_MMSE_coded / max(TotalNum_coded, 1)
    BER_Perfect_coded[jj] = TotalErr_Perfect_coded / max(TotalNum_coded, 1)

# Save & plot
out = {
    "EBN0": EbNoDB,
    "BER_uncoded": {"ESN": BER_ESN_uncoded, "LS": BER_LS_uncoded, "MMSE": BER_MMSE_uncoded, "Perfect": BER_Perfect_uncoded},
    "BER_coded": {"ESN": BER_ESN_coded, "LS": BER_LS_coded, "MMSE": BER_MMSE_coded, "Perfect": BER_Perfect_coded},
    "meta": {"n": int(n), "k": int(k), "rate": float(k/n), "N": int(N), "m": int(m), "dv": int(d_v), "dc": int(d_c)}
}
with open("./BERvsEBNo_esn_siso_ldpc.pkl", "wb") as f:
    pickle.dump(out, f)

plt.figure()
plt.semilogy(EbNoDB, BER_ESN_uncoded, label='ESN (uncoded)')
plt.semilogy(EbNoDB, BER_LS_uncoded, label='LS (uncoded)')
plt.semilogy(EbNoDB, BER_MMSE_uncoded, label='MMSE (uncoded)')
plt.semilogy(EbNoDB, BER_Perfect_uncoded, label='Perfect (uncoded)')
plt.grid(True); plt.legend(); plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER'); plt.title('SISO Uncoded BER')

plt.figure()
plt.semilogy(EbNoDB, BER_ESN_coded, label='ESN (LDPC decoded)')
plt.semilogy(EbNoDB, BER_LS_coded, label='LS (LDPC decoded)')
plt.semilogy(EbNoDB, BER_MMSE_coded, label='MMSE (LDPC decoded)')
plt.semilogy(EbNoDB, BER_Perfect_coded, label='Perfect (LDPC decoded)')
plt.grid(True); plt.legend(); plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER'); plt.title('SISO Coded BER (LDPC)')
plt.tight_layout()
plt.show()
