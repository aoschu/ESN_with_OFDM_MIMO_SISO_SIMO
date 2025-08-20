# Demo_SISO_qpsk_ldpc_esn_fixedllr.py
# SISO OFDM · QPSK (4-QAM) · AWGN (H=1) · smooth PA
# ESN time-domain recovery -> FFT -> soft symbols -> LLRs -> LDPC decode
# Baselines: MMSE (AWGN perfect) and LS (explicit)
# Notes:
#  - If LDPC decoder fails to converge, flip LLR_SIGN (see below)
#  - Uses pyESN (fit/predict) and pyldpc
# -----------------------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from scipy import signal
from pyESN import ESN
from pyldpc import make_ldpc, decode as ldpc_decode, get_message

# --------------------------
# Helpers
# --------------------------
def add_cp(x, cp): return np.r_[x[-cp:], x] if cp > 0 else x
def rm_cp(x, cp):  return x[cp:] if cp > 0 else x

def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]   # LSB-first

def qam_bit_labels(M, m):
    lab = np.zeros((M, m), dtype=int)
    for k in range(M):
        lab[k, :] = bits_to_grayvec(k, m)
    return lab

def map_bits_to_syms(bits, const, m):
    N = bits.shape[0] // m
    X = np.zeros((N,), dtype=complex)
    P2 = np.power(2, np.arange(m)).reshape(1, -1)
    for i in range(N):
        idx = int((P2 @ bits[m*i + np.arange(m), 0])[0])  # LSB-first
        X[i] = const[idx]
    return X

def hard_bits_from_syms(Xhat, const, m):
    N = len(Xhat)
    out = np.zeros((N*m, 1), dtype=int)
    for i, x in enumerate(Xhat):
        idx = int(np.argmin(np.abs(const - x)))
        out[m*i:m*(i+1), 0] = bits_to_grayvec(idx, m)
    return out

def maxlog_llr(z, const, labels, sigma2):
    """
    Max-log LLR per bit (vectorized).
    Returns shape (N, m) with convention: LLR>0 -> bit=1 more likely (by this impl).
    Implementation: LLR_b = (min_distance_to_0 - min_distance_to_1) / sigma2
    """
    N = z.shape[0]; m = labels.shape[1]
    d = np.abs(z.reshape(-1,1) - const.reshape(1,-1))**2  # (N, M)
    llr = np.zeros((N, m))
    s2 = max(float(sigma2), 1e-12)
    for b in range(m):
        d0 = np.min(d[:, labels[:, b] == 0], axis=1)
        d1 = np.min(d[:, labels[:, b] == 1], axis=1)
        llr[:, b] = (d0 - d1) / s2     # positive => bit=1 more likely (by construction)
    return llr

# --------------------------
# Simulation parameters (tweak as needed)
# --------------------------
np.random.seed(42)

# Channel / noise
No = 1e-5                      # noise spectral quantity used in your previous pipeline
EbNoDB = np.arange(0, 31, 15)   # dB grid

# OFDM / modulation
N = 128
m = 2                          # QPSK / 4-QAM
CP = 0

# Frames per SNR (reduce to speed up; increase for smoother curves)
NumFrames = 40

# Power normalization matching your previous convention
Ptotal = 10**(EbNoDB/10) * No * N
Pi = Ptotal / N

# Constellation (LSB-first indexing consistent with helpers)
Const = (np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)).astype(complex)
LABELS = qam_bit_labels(2**m, m)

# PA
p_smooth = 1
ClipLeveldB = 3

# ESN (pyESN)
nIn, nOut, nRes = 2, 2, 200   # you can tune reservoir size
inputScaler = 0.005
inputOffset = 0.0
teacherScalingBase = 5e-7
spectralRadius = 0.9

# LDPC
dv, dc = 4, 8
LLR_CLIP = 50.0
n_code = N * m
if n_code % dc != 0:
    raise ValueError(f"dc={dc} must divide n_code={n_code}.")
H, G = make_ldpc(n_code, dv, dc, systematic=True, sparse=True)
k_info = G.shape[1]
print(f"[LDPC] n={n_code}, k≈{k_info}, rate≈{k_info/n_code:.3f}")

# Debug / LLR sign convention:
# pyldpc sometimes expects positive => bit=0 or bit=1 depending on version.
# Our maxlog_llr returns positive => bit=1 more likely. If pyldpc in your env expects
# the opposite, set LLR_SIGN = -1. If decoder fails (BER ~ 0.5..1), try flipping sign.
LLR_SIGN = 1    # set -1 if needed

# Option: use true sigma2 (recommended) or decision-directed estimate
USE_TRUE_SIGMA = True

# Holders
BER_unc_ESN  = np.zeros_like(EbNoDB, dtype=float)
BER_unc_MMSE = np.zeros_like(EbNoDB, dtype=float)
BER_unc_LS   = np.zeros_like(EbNoDB, dtype=float)
BER_c_ESN    = np.zeros_like(EbNoDB, dtype=float)
BER_c_MMSE   = np.zeros_like(EbNoDB, dtype=float)
BER_c_LS     = np.zeros_like(EbNoDB, dtype=float)

# --------------------------
# Main SNR loop
# --------------------------
for jj, eb in enumerate(EbNoDB):
    print(f"\n=== Eb/N0 = {eb} dB ===")
    var_x = (10**(eb/10)) * No * N
    Aclip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB/20)

    # ESN per-SNR scaling (train matched)
    inScale  = (inputScaler/np.sqrt(var_x)) * np.ones(nIn)
    inShift  = (inputOffset/inputScaler) * np.ones(nIn)
    teachSc  = teacherScalingBase * np.ones(nOut)

    # AWGN flat
    c0 = np.array([1.0 + 0j])

    # ---------- Pilot train ESN (one pilot OFDM symbol per SNR) ----------
    Xp = Const[np.random.randint(0, 4, size=N)]
    x_td = N * np.fft.ifft(Xp)
    x_cp = add_cp(x_td, CP) * (Pi[jj]**0.5)
    x_nl = x_cp / ((1 + (np.abs(x_cp)/Aclip)**(2*p_smooth))**(1/(2*p_smooth)))
    y_td = signal.lfilter(c0, [1], x_nl)
    noise = math.sqrt(len(y_td)*No/2) * (np.random.randn(len(y_td)) + 1j*np.random.randn(len(y_td)))
    y_td = y_td + noise

    # create & train ESN (pyESN)
    esn = ESN(n_inputs=nIn, n_outputs=nOut, n_reservoir=nRes,
              spectral_radius=spectralRadius, sparsity=0.1,
              input_shift=inShift, input_scaling=inScale,
              teacher_scaling=teachSc, teacher_shift=np.zeros(nOut),
              feedback_scaling=np.zeros(nOut))

    Ein  = np.column_stack([y_td.real, y_td.imag])   # (N+CP, 2)
    Eout = np.column_stack([x_cp.real, x_cp.imag])   # (N+CP, 2)
    esn.fit(Ein, Eout)   # pyESN: simple fit

    # reset counters
    err_unc_esn = err_unc_mmse = err_unc_ls = 0
    bits_unc_total = 0
    err_c_esn = err_c_mmse = err_c_ls = 0
    info_total = 0

    # ---------- data frames ----------
    for fr in range(NumFrames):
        # LDPC encode random info bits
        u = np.random.randint(0, 2, size=(k_info,), dtype=np.int8)
        cword = np.asarray((G.dot(u) % 2)).ravel().astype(np.int8)
        bits = cword.reshape(-1, 1)            # (N*m, 1)
        X = map_bits_to_syms(bits, Const, m)  # (N,)

        # OFDM + PA + AWGN
        x_td = N * np.fft.ifft(X)
        x_cp = add_cp(x_td, CP) * (Pi[jj]**0.5)
        x_nl = x_cp / ((1 + (np.abs(x_cp)/Aclip)**(2*p_smooth))**(1/(2*p_smooth)))
        y_td = signal.lfilter(c0, [1], x_nl)
        noise = math.sqrt(len(y_td)*No/2) * (np.random.randn(len(y_td)) + 1j*np.random.randn(len(y_td)))
        y_td = y_td + noise

        # frequency-domain receive
        Y = (1/N) * np.fft.fft(rm_cp(y_td, CP))

        # ESN inference
        Xin = np.column_stack([y_td.real, y_td.imag])
        yhat = esn.predict(Xin)   # shape (len(Xin), 2)
        yhat_c = yhat[:,0] + 1j*yhat[:,1]
        xhat_td = yhat_c[-N:]
        Xhat_ESN = (1/N) * np.fft.fft(xhat_td) / math.sqrt(Pi[jj])

        # MMSE (AWGN perfect)
        Xhat_MMSE = Y / math.sqrt(Pi[jj])

        # LS (H_est = 1)
        Xhat_LS = Y / 1.0 / math.sqrt(Pi[jj])

        # ---------- Uncoded BER (hard decisions) ----------
        rx_esn  = hard_bits_from_syms(Xhat_ESN, Const, m)
        rx_mmse = hard_bits_from_syms(Xhat_MMSE, Const, m)
        rx_ls   = hard_bits_from_syms(Xhat_LS, Const, m)
        err_unc_esn  += int(np.sum(bits != rx_esn))
        err_unc_mmse += int(np.sum(bits != rx_mmse))
        err_unc_ls   += int(np.sum(bits != rx_ls))
        bits_unc_total += N * m

        # ---------- LLRs ----------
        if USE_TRUE_SIGMA:
            sigma2 = No   # using your power normalization, noise variance per subcarrier equals No
        else:
            # decision-directed fallback
            def est_sigma2_from_decision(xhat, const):
                idx = np.argmin(np.abs(xhat.reshape(-1,1) - const.reshape(1,-1))**2, axis=1)
                xhard = const[idx]
                e = xhat - xhard
                return float(np.mean(np.abs(e)**2) + 1e-12)
            sigma2 = est_sigma2_from_decision(Xhat_MMSE, Const)

        llr_esn  = maxlog_llr(Xhat_ESN,  Const, LABELS, sigma2) * LLR_SIGN
        llr_mmse = maxlog_llr(Xhat_MMSE, Const, LABELS, sigma2) * LLR_SIGN
        llr_ls   = maxlog_llr(Xhat_LS,   Const, LABELS, sigma2) * LLR_SIGN

        # clip LLRs to protect pyldpc numeric stability
        yobs_esn  = np.clip(llr_esn.reshape(-1),  -LLR_CLIP, LLR_CLIP)
        yobs_mmse = np.clip(llr_mmse.reshape(-1), -LLR_CLIP, LLR_CLIP)
        yobs_ls   = np.clip(llr_ls.reshape(-1),   -LLR_CLIP, LLR_CLIP)

        # DEBUG single-frame inspection (optional)
        # if fr == 0 and jj == 0:
        #     print("sample llr_mmse[0:10]:", yobs_mmse[:10])

        # ---------- LDPC decode (pyldpc) ----------
        d_esn  = ldpc_decode(H, yobs_esn,  snr=1.0, maxiter=100)
        d_mmse = ldpc_decode(H, yobs_mmse, snr=1.0, maxiter=100)
        d_ls   = ldpc_decode(H, yobs_ls,   snr=1.0, maxiter=100)

        uhat_esn  = get_message(G, d_esn).astype(np.int8)
        uhat_mmse = get_message(G, d_mmse).astype(np.int8)
        uhat_ls   = get_message(G, d_ls).astype(np.int8)

        err_c_esn  += int(np.sum(u != uhat_esn))
        err_c_mmse += int(np.sum(u != uhat_mmse))
        err_c_ls   += int(np.sum(u != uhat_ls))
        info_total += len(u)

    # store BERs
    BER_unc_ESN[jj]  = err_unc_esn / max(bits_unc_total, 1)
    BER_unc_MMSE[jj] = err_unc_mmse / max(bits_unc_total, 1)
    BER_unc_LS[jj]   = err_unc_ls   / max(bits_unc_total, 1)
    BER_c_ESN[jj]    = err_c_esn    / max(info_total, 1)
    BER_c_MMSE[jj]   = err_c_mmse   / max(info_total, 1)
    BER_c_LS[jj]     = err_c_ls     / max(info_total, 1)

    print(f" Uncoded: MMSE {BER_unc_MMSE[jj]:.3e}, ESN {BER_unc_ESN[jj]:.3e}")
    print(f" Coded:   MMSE {BER_c_MMSE[jj]:.3e}, ESN {BER_c_ESN[jj]:.3e}")

# --------------------------
# Save & Plot
# --------------------------
results = {
    "EBN0": EbNoDB.tolist(),
    "uncoded": {"ESN": BER_unc_ESN.tolist(), "MMSE": BER_unc_MMSE.tolist(), "LS": BER_unc_LS.tolist()},
    "coded":   {"ESN": BER_c_ESN.tolist(),   "MMSE": BER_c_MMSE.tolist(), "LS": BER_c_LS.tolist()},
}
with open("results_siso_qpsk_ldpc_esn_fixedllr.pkl", "wb") as f:
    pickle.dump(results, f)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.semilogy(EbNoDB, BER_unc_MMSE, 'rs-.', label='MMSE (uncoded)')
plt.semilogy(EbNoDB, BER_unc_LS,   'bo--', label='LS (uncoded)')
plt.semilogy(EbNoDB, BER_unc_ESN,  'g^--', label='ESN (uncoded)')
plt.grid(True, which='both', ls=':'); plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER'); plt.legend()

plt.subplot(1,2,2)
plt.semilogy(EbNoDB, BER_c_MMSE, 'r*-',  label='MMSE + LDPC')
plt.semilogy(EbNoDB, BER_c_LS,   'b*-',  label='LS + LDPC')
plt.semilogy(EbNoDB, BER_c_ESN,  'g*-',  label='ESN + LDPC')
plt.grid(True, which='both', ls=':'); plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER (info-bit)'); plt.legend()

plt.tight_layout()
plt.savefig("ber_siso_qpsk_ldpc_esn_fixedllr.png", dpi=150)
plt.show()

print("Saved results and plot.")
