# Demo_MIMO_4x8_ESN_LLRCal_LDPC_CDLB.py
# --------------------------------------------------
# 4x8 MIMO-OFDM with nonlinear PA + CDL-B (TDL equivalent) channel + block fading
# Compare MMSE vs ESN with an LLR-calibration head trained by cross-entropy.
#
# Pipeline:
#  - ESN does time-domain recovery -> FFT -> raw soft symbols
#  - MMSE equalizer gives soft symbols too (from LS->MMSE interp)
#  - For each method, compute raw max-log LLRs (per bit)
#  - Train a tiny logistic calibrator:  LLR_cal_b = -(a_b * LLR_raw_b + b_b)
#    because logistic learns log P(1)/P(0) but decoder expects log P(0)/P(1)
#  - Feed calibrated LLRs to LDPC decoder, measure post-LDPC BER
#  - Also plot pre-decoder (uncoded) BER on the same axes
#
# Saves figures:
#   results_4x8/BER_uncoded_coded_overlay_MMSE_ESN.png
#   results_4x8/LLR_calibration_params_EbNo<dB>.txt  (a,b per bit for ESN & MMSE)
#   results_4x8/results_ber_4x8_cdLB.pkl             (for notebook plotting)
#
# Requirements:
#   pip install numpy scipy matplotlib pyESN pyldpc
#   have helper_mimo_esn_generic.py available (trainMIMOESN_generic)
# --------------------------------------------------

import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import scipy.sparse as sp
from pyESN import ESN
from pyldpc import make_ldpc, decode as ldpc_decode, get_message
from helper_mimo_esn_generic import trainMIMOESN_generic

# --------------------
# Utilities
# --------------------

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

def equalize_zf(Yk, Hk, power_scale):
    HH = Hk.conj().T
    G = HH @ Hk
    G += 1e-12 * np.eye(G.shape[0], dtype=G.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def equalize_mmse(Yk, Hk, power_scale, noise_over_power):
    HH = Hk.conj().T
    G = HH @ Hk + noise_over_power * np.eye(Hk.shape[1], dtype=Hk.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def reconstruct_esn_outputs_generic(x_hat_tmp, Delay, Delay_Min, N, N_t):
    """Rebuild complex time-domain per-TX sequences from ESN outputs."""
    xs = []
    for tx in range(N_t):
        re_col = 2*tx
        im_col = 2*tx + 1
        start_re = Delay[re_col] - Delay_Min
        start_im = Delay[im_col] - Delay_Min
        re_seq = x_hat_tmp[start_re:start_re + N, re_col]
        im_seq = x_hat_tmp[start_im:start_im + N, im_col]
        xs.append(re_seq + 1j*im_seq)
    return xs

def qam_bit_labels(M, m):
    labels = np.zeros((M, m), dtype=int)
    for idx in range(M):
        labels[idx, :] = bits_to_grayvec(idx, m)
    return labels

def qam_llrs_maxlog(z_hat_1d, const, bit_labels, sigma2):
    """
    Max-log LLR for bit=0 vs bit=1:
        LLR = log P(bit=0|z) - log P(bit=1|z) ≈ (d1 - d0)/sigma2
    Positive LLR means bit 0 more likely.
    """
    N = z_hat_1d.shape[0]
    m = bit_labels.shape[1]
    llrs = np.zeros((N, m), dtype=float)
    masks0 = [bit_labels[:, b] == 0 for b in range(m)]
    masks1 = [bit_labels[:, b] == 1 for b in range(m)]
    dists = np.abs(z_hat_1d.reshape(-1,1) - const.reshape(1,-1))**2
    for b in range(m):
        d0 = np.min(dists[:, masks0[b]], axis=1)
        d1 = np.min(dists[:, masks1[b]], axis=1)
        llrs[:, b] = (d1 - d0) / max(sigma2, 1e-12)
    return llrs  # shape (N, m)

def est_sigma2_from_decision(Xhat_col, const):
    idx = np.argmin(np.abs(Xhat_col.reshape(-1,1) - const.reshape(1,-1))**2, axis=1)
    Xhard = const[idx]
    err = Xhat_col - Xhard
    return float(np.mean(np.abs(err)**2) + 1e-12)

def ldpc_encode_bits(G, u):
    x = G.dot(u) % 2 if sp.issparse(G) else (G @ u % 2)
    x = np.asarray(x).ravel().astype(np.int8)
    return x

def hard_bits_from_syms(Xhat_matrix, Const, m):
    N, N_t = Xhat_matrix.shape
    RxBits = np.zeros((N*m, N_t), dtype=int)
    for ii in range(N):
        for tx in range(N_t):
            sym = Xhat_matrix[ii, tx]
            idx = int(np.argmin(np.abs(Const - sym)))
            RxBits[m*ii:m*(ii+1), tx] = bits_to_grayvec(idx, m)
    return RxBits

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fit_logreg_1d(x, y, maxiter=400, lr=0.15, l2=1e-3):
    """Return a,b for p(y=1|x)=sigmoid(a*x+b)."""
    a, b = 1.0, 0.0
    n = len(x)
    for t in range(maxiter):
        z = a*x + b
        p = sigmoid(z)
        ga = np.dot((p - y), x)/n + l2*a
        gb = np.sum(p - y)/n
        a -= lr * ga
        b -= lr * gb
    return float(a), float(b)

# --------------------
# CDL-B (TDL-equivalent) channel helpers
# --------------------

# 3GPP TR 38.901 (Rel-16) Table 7.7.2-2: TDL-B (23 taps)
# Normalized delays (unitless) & powers [dB]
_TDLB_NORM_DELAYS = np.array([
    0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681,
    0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294,
    3.0219, 3.6187, 4.1067, 4.2790, 4.7834
], dtype=float)

_TDLB_POW_DB = np.array([
     0.0,  -2.2,  -4.0,  -3.2,  -9.8,  -1.2,  -3.4,  -5.2,  -7.6,
    -3.0,  -8.9,  -9.0,  -4.8,  -5.7,  -7.5,  -1.9,  -7.6, -12.2,
    -9.8, -11.4, -14.9,  -9.2, -11.3
], dtype=float)

def _gen_cdlb_impulse(IsiDuration, fs_Hz, DS_ns, rng):
    """
    Build a discrete-time complex impulse response of length IsiDuration
    from CDL-B (TDL) normalized delays & powers, scaled to DS_ns (RMS delay spread).
    Delays are placed to the nearest sample with linear split to next tap for fractional parts.
    Independent Rayleigh complex gains per (tap) consistent with per-path power.
    """
    pow_lin = 10.0**(_TDLB_POW_DB/10.0)
    pow_lin = pow_lin / np.sum(pow_lin)  # normalize
    delays_s = (_TDLB_NORM_DELAYS * DS_ns) * 1e-9
    delays_samp = delays_s * fs_Hz  # continuous sample delays

    h = np.zeros(IsiDuration, dtype=np.complex128)
    for p in range(len(_TDLB_NORM_DELAYS)):
        d = delays_samp[p]
        i0 = int(np.floor(d))
        frac = d - i0
        # Complex CN(0, pow_lin[p]) per path
        gp = (rng.standard_normal() + 1j*rng.standard_normal())/np.sqrt(2.0) * np.sqrt(pow_lin[p])
        if 0 <= i0 < IsiDuration:
            h[i0] += gp * (1.0 - frac)
        if 0 <= (i0+1) < IsiDuration:
            h[i0+1] += gp * frac
    # Optional overall normalization to unit power
    if np.sum(np.abs(h)**2) > 0:
        h = h / np.sqrt(np.sum(np.abs(h)**2))
    return h

def build_cdlb_mimo_taps(N_r, N_t, IsiDuration, fs_Hz, DS_ns, seed=None):
    """
    Create taps c[nr][nt] for all Rx/Tx links using CDL-B (TDL) with DS_ns.
    """
    rng = np.random.default_rng(seed)
    c = [[None for _ in range(N_t)] for __ in range(N_r)]
    for nr in range(N_r):
        for nt in range(N_t):
            # Independent link realizations (simple, no spatial corr)
            c[nr][nt] = _gen_cdlb_impulse(IsiDuration, fs_Hz, DS_ns, rng)
    return c

# --------------------
# System parameters
# --------------------
W = 2*1.024e6          # sample rate [Hz]
f_D = 100              # (unused here; Doppler=0 for block fading)
No = 1e-5
IsiDuration = 8        # channel memory (discrete taps)
EbNoDB = np.arange(0, 31, 3).astype(np.int32)  # full grid

N_t = 4
N_r = 8

N = 128                # subcarriers
m = 4                  # 16-QAM
m_pilot = 4
NumOfdmSymbols = 1000   # moderate runtime
Ptotal = 10**(EbNoDB/10)*No*N

p_smooth = 1
ClipLeveldB = 3

T_OFDM_Total = (N + IsiDuration - 1)/W
# Block-fading length L computed as before (coherence symbols)
tau_c = 0.5/ max(f_D, 1e-9)  # avoid /0
L = max(1, math.floor(tau_c/T_OFDM_Total))

Pi = Ptotal/N
NumBitsPerSymbol = m*N
Const = np.array(unit_qam_constellation(m)).astype(complex)
ConstPilot = np.array(unit_qam_constellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

# Nonlinear PA smooth clipping shape
temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/max(temp,1e-12))
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)

# ESN params
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2*N_r
nOutputUnits = 2*N_t
nInternalUnits = 300
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

teacherShift = np.zeros(nOutputUnits)
feedbackScaling = feedbackScaler*np.ones(nOutputUnits)

Min_Delay = 0
Max_Delay = int(math.ceil(IsiDuration/2) + 2)
DelayFlag = 0

TRAIN_EBNO_FIXED_DB = 12
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# LDPC params
USE_LDPC = True
LDPC_dv = 4
LDPC_dc = 8
LDPC_MAXITER = 100
LLR_CLIP = 20.0  # wider clipping is typical for LDPC

n_code = N * m
if n_code % LDPC_dc != 0:
    raise ValueError(f"LDPC_dc={LDPC_dc} must divide n_code={n_code}.")
H, G = make_ldpc(n_code, LDPC_dv, LDPC_dc, systematic=True, sparse=True)
k_info = G.shape[1]
BIT_LABELS = qam_bit_labels(2**m, m)
print(f"[LDPC] Built regular code: n={n_code}, k≈{k_info}, rate≈{k_info/n_code:.3f}")

# CDL-B delay spread selection (ns). Ensure max delay fits CP at this Fs.
# Using 300 ns (3GPP "Long") which is safely within CP for current settings.
CDLB_DS_NS = 300.0
max_norm_delay = float(np.max(_TDLB_NORM_DELAYS))
CP_seconds = CyclicPrefixLen / W
max_delay_seconds = (max_norm_delay * CDLB_DS_NS) * 1e-9
if max_delay_seconds > CP_seconds:
    print(f"[WARN] CDL-B max delay ({max_delay_seconds*1e9:.1f} ns) exceeds CP ({CP_seconds*1e9:.1f} ns). "
          f"Consider increasing CP or reducing DS. Continuing anyway.")

# Calibration/train/test split
CAL_FRAC = 0.3   # fraction of data symbols used to fit a,b

# Holders
BER_uncoded_ESN = np.zeros(len(EbNoDB))
BER_uncoded_MMSE = np.zeros(len(EbNoDB))
BER_coded_ESN   = np.zeros(len(EbNoDB))
BER_coded_MMSE  = np.zeros(len(EbNoDB))

outdir = "./results_4x8"
os.makedirs(outdir, exist_ok=True)

# Prior for MMSE on taps (diagonal covariance example)
R_h = np.diag(IsiMagnitude[:IsiDuration])

# --------------------
# Run per SNR
# --------------------
for jj, ebno_db in enumerate(EbNoDB):
    print(f"=== Eb/No {ebno_db} dB ===")
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    inputScaling_matched = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    inputScaling_trainFixed = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFixed = (inputOffset/inputScaler) * np.ones(nInputUnits)

    TotalBits_uncoded_ESN = 0
    TotalBits_uncoded_MMSE = 0
    Err_uncoded_ESN = 0
    Err_uncoded_MMSE = 0

    InfoBits_total = 0
    Err_coded_ESN = 0
    Err_coded_MMSE = 0

    esn_matched = None
    Delay_m = None; Delay_Min_m = None; Delay_Max_m = None; nForget_m = None

    xcal_esn = [[] for _ in range(m)]
    ycal_esn = [[] for _ in range(m)]
    xcal_mmse = [[] for _ in range(m)]
    ycal_mmse = [[] for _ in range(m)]

    a_esn = np.ones(m); b_esn = np.zeros(m)
    a_mmse = np.ones(m); b_mmse = np.zeros(m)

    # Seed per SNR for reproducibility of channel draw at redraws
    chan_seed = int(1234 + ebno_db)

    for kk in range(1, NumOfdmSymbols+1):
        redraw = (np.remainder(kk, L) == 1)
        if redraw:
            # ---- CDL-B TDL taps for whole 4x8 link matrix (block-fading) ----
            c = build_cdlb_mimo_taps(N_r, N_t, IsiDuration, fs_Hz=W, DS_ns=CDLB_DS_NS, seed=chan_seed+kk)
            # ---- Pilots: OFDM mapping (unchanged) ----
            TxBitsPilot = (np.random.rand(N*m_pilot, N_t) > 0.5).astype(np.int32)
            X_p = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), tx])[0])
                    X_p[ii, tx] = ConstPilot[idx]
            # Sparse LS pattern (one Tx per subcarrier round-robin)
            X_LS = np.zeros_like(X_p)
            for tx in range(N_t):
                X_LS[tx::N_t, tx] = X_p[tx::N_t, tx]

            # IFFT + CP (pilots) with power scaling
            x_temp = np.zeros((N, N_t), dtype=complex)
            x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
            x_LS_CP = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_temp[:, tx] = N * np.fft.ifft(X_p[:, tx])
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
                x_temp_ls = N * np.fft.ifft(X_LS[:, tx])
                x_LS_CP[:, tx] = np.r_[x_temp_ls[-CyclicPrefixLen:], x_temp_ls] * (Pi[jj]**0.5)
            # PA (smooth clipping)
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            x_LS_CP_NLD = x_LS_CP / ((1 + (np.abs(x_LS_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Time-domain channel application via lfilter with CDL-B TDL taps
            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            y_LS_CP = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
                    y_LS_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_LS_CP_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP[:, nr] += noise
                y_LS_CP[:, nr] += noise

            # FFT remove CP
            Y_p = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)
            Y_LS = (1/N) * np.fft.fft(y_LS_CP[CyclicPrefixLen:, :], axis=0)

            # LS -> MMSE interpolation per (rx,tx)
            H_LS = np.zeros((N, N_r, N_t), dtype=complex)
            H_MMSE = np.zeros_like(H_LS)
            MMSEScaler = (No/Pi[jj])/(N/2)
            for nr in range(N_r):
                for tx in range(N_t):
                    sc_idx = np.arange(tx, N, N_t)
                    denom = (X_LS[sc_idx, tx] * (Pi[jj]**0.5) + 1e-12)
                    Hls_sc = Y_LS[sc_idx, nr] / denom
                    tmpf = interpolate.interp1d(sc_idx, Hls_sc, kind='linear', bounds_error=False, fill_value='extrapolate')
                    Hls_full = tmpf(np.arange(N))

                    # MMSE regularization in time domain (truncate to IsiDuration)
                    c_LS = np.fft.ifft(Hls_full)
                    c_LS_trunc = c_LS[:IsiDuration]
                    # scalar-matrix multiply (correct form)
                    c_MMSE = np.linalg.solve(MMSEScaler*np.linalg.inv(R_h) + np.eye(IsiDuration), c_LS_trunc)
                    Hmmse_full = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

                    H_LS[:, nr, tx] = Hls_full
                    H_MMSE[:, nr, tx] = Hmmse_full

            # Train ESN (matched scaling)
            esn_m = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=spectralRadius, sparsity=0.1,
                        input_shift=inputShift_matched, input_scaling=inputScaling_matched,
                        teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                        feedback_scaling=feedbackScaling)
            ESN_input, ESN_output, esn_m, Delay_m, Delay_Idx_m, Delay_Min_m, Delay_Max_m, nForget_m, _ = \
                trainMIMOESN_generic(esn_m, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen,
                                     N, N_t, N_r, IsiDuration, y_CP, x_CP)
            esn_matched = esn_m

        # ----- DATA -----
        # LDPC encode per TX
        TxBits = np.zeros((N*m, N_t), dtype=np.int8)
        InfoBits = [None]*N_t
        for tx in range(N_t):
            u = np.random.randint(0, 2, size=(k_info,), dtype=np.int8)
            cword = ldpc_encode_bits(G, u)
            InfoBits[tx] = u
            TxBits[:, tx] = cword

        # Map to QAM
        X = np.zeros((N, N_t), dtype=complex)
        for ii in range(N):
            for tx in range(N_t):
                bits_idx = TxBits[m*ii + np.arange(m), tx]
                idx = int((PowersOfTwo @ bits_idx)[0])
                X[ii, tx] = Const[idx]

        # IFFT + CP + PA
        x_temp = np.zeros((N, N_t), dtype=complex)
        x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
        for tx in range(N_t):
            x_temp[:, tx] = N * np.fft.ifft(X[:, tx])
            x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
        x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

        # Pass through CDL-B (TDL) channel + AWGN
        y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
        for nr in range(N_r):
            for tx in range(N_t):
                y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
            noise = math.sqrt(len(y_CP[:, nr])*No/2) * (np.random.randn(len(y_CP[:, nr])) + 1j*np.random.randn(len(y_CP[:, nr])))
            y_CP[:, nr] += noise

        # Remove CP + FFT
        Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)

        # ESN inference
        ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
        for rx in range(N_r):
            ESN_input_m[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(Delay_Max_m)]
            ESN_input_m[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(Delay_Max_m)]
        x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
        x_time_list_m = reconstruct_esn_outputs_generic(x_hat_m_tmp, Delay_m, Delay_Min_m, N, N_t)
        X_hat_ESN = np.zeros((N, N_t), dtype=complex)
        for tx in range(N_t):
            X_hat_ESN[:, tx] = (1/N) * np.fft.fft(x_time_list_m[tx]) / math.sqrt(Pi[jj])

        # MMSE equalization
        X_hat_MMSE = np.zeros((N, N_t), dtype=complex)
        for k in range(N):
            Yk = Y[k, :].reshape(N_r, 1)
            Hk_MMSEf = H_MMSE[k, :, :]
            X_hat_MMSE[k, :] = equalize_mmse(Yk, Hk_MMSEf, math.sqrt(Pi[jj]), noise_over_power=(No/Pi[jj])).reshape(-1)

        # Uncoded BER
        RxBits_ESN   = hard_bits_from_syms(X_hat_ESN,  Const, m)
        RxBits_MMSE  = hard_bits_from_syms(X_hat_MMSE, Const, m)
        Err_uncoded_ESN  += int(np.sum(TxBits != RxBits_ESN))
        Err_uncoded_MMSE += int(np.sum(TxBits != RxBits_MMSE))
        TotalBits_uncoded_ESN  += NumBitsPerSymbol * N_t
        TotalBits_uncoded_MMSE += NumBitsPerSymbol * N_t

        # LLRs for calibration/decoding
        sigma2_esn = np.mean([est_sigma2_from_decision(X_hat_ESN[:, tx], Const) for tx in range(N_t)])
        llr_esn_all = []
        for tx in range(N_t):
            llr_esn_all.append(qam_llrs_maxlog(X_hat_ESN[:, tx], Const, BIT_LABELS, sigma2_esn))
        llr_esn_all = np.stack(llr_esn_all, axis=2)  # (N, m, N_t)

        sigma2_mmse = np.mean([est_sigma2_from_decision(X_hat_MMSE[:, tx], Const) for tx in range(N_t)])
        llr_mmse_all = []
        for tx in range(N_t):
            llr_mmse_all.append(qam_llrs_maxlog(X_hat_MMSE[:, tx], Const, BIT_LABELS, sigma2_mmse))
        llr_mmse_all = np.stack(llr_mmse_all, axis=2)

        bits_all = np.zeros((N, m, N_t), dtype=np.int8)
        for tx in range(N_t):
            for ii in range(N):
                bits_all[ii, :, tx] = TxBits[m*ii:m*(ii+1), tx]

        if kk <= int(CAL_FRAC * NumOfdmSymbols):
            # accumulate calibration data
            for b in range(m):
                xcal_esn[b].append( llr_esn_all[:, b, :].reshape(-1) )
                ycal_esn[b].append( bits_all[:, b, :].reshape(-1) )
                xcal_mmse[b].append( llr_mmse_all[:, b, :].reshape(-1) )
                ycal_mmse[b].append( bits_all[:, b, :].reshape(-1) )
        else:
            snr_for_ldpc = 1.0  # because we pass yobs = 0.5 * LLR; pyldpc builds LLR as 2*snr*yobs

            for tx in range(N_t):
                # ----- ESN calibrated -----
                llr_esn_resh = llr_esn_all[:, :, tx].copy()
                for b in range(m):
                    # Flip sign after calibration so it matches log P(0)/P(1)
                    llr_esn_resh[:, b] = np.clip(-(a_esn[b]*llr_esn_resh[:, b] + b_esn[b]),
                                                 -LLR_CLIP, LLR_CLIP)

                yobs_esn = 0.5 * llr_esn_resh.reshape(-1).astype(float)
                d_hat_esn = ldpc_decode(H, yobs_esn, snr_for_ldpc, maxiter=LDPC_MAXITER)
                u_hat_esn = get_message(G, d_hat_esn).astype(np.int8)

                # ----- MMSE calibrated -----
                llr_mmse_resh = llr_mmse_all[:, :, tx].copy()
                for b in range(m):
                    llr_mmse_resh[:, b] = np.clip(-(a_mmse[b]*llr_mmse_resh[:, b] + b_mmse[b]),
                                                  -LLR_CLIP, LLR_CLIP)

                yobs_mmse = 0.5 * llr_mmse_resh.reshape(-1).astype(float)
                d_hat_mmse = ldpc_decode(H, yobs_mmse, snr_for_ldpc, maxiter=LDPC_MAXITER)
                u_hat_mmse = get_message(G, d_hat_mmse).astype(np.int8)

                u_true = InfoBits[tx]
                Err_coded_ESN  += int(np.sum(u_true != u_hat_esn))
                Err_coded_MMSE += int(np.sum(u_true != u_hat_mmse))
                InfoBits_total += int(len(u_true))

        if kk == int(CAL_FRAC * NumOfdmSymbols):
            print("  Fitting LLR calibrators...")
            a_esn = np.ones(m); b_esn = np.zeros(m)
            a_mmse = np.ones(m); b_mmse = np.zeros(m)
            for b in range(m):
                xe = np.concatenate(xcal_esn[b]); ye = np.concatenate(ycal_esn[b]).astype(float)
                xm = np.concatenate(xcal_mmse[b]); ym = np.concatenate(ycal_mmse[b]).astype(float)
                ae, be = fit_logreg_1d(xe, ye, maxiter=400, lr=0.1, l2=1e-3)
                am, bm = fit_logreg_1d(xm, ym, maxiter=400, lr=0.1, l2=1e-3)
                a_esn[b], b_esn[b] = ae, be
                a_mmse[b], b_mmse[b] = am, bm

    if TotalBits_uncoded_ESN > 0:
        BER_uncoded_ESN[jj]  = Err_uncoded_ESN  / TotalBits_uncoded_ESN
        BER_uncoded_MMSE[jj] = Err_uncoded_MMSE / TotalBits_uncoded_MMSE
    if InfoBits_total > 0:
        BER_coded_ESN[jj]  = Err_coded_ESN  / InfoBits_total
        BER_coded_MMSE[jj] = Err_coded_MMSE / InfoBits_total

    with open(os.path.join(outdir, f"LLR_calibration_params_EbNo{ebno_db}dB.txt"), "w") as f:
        f.write("bit, a_esn, b_esn, a_mmse, b_mmse\n")
        for b in range(m):
            f.write(f"{b}, {a_esn[b]:.4f}, {b_esn[b]:.4f}, {a_mmse[b]:.4f}, {b_mmse[b]:.4f}\n")

# --------------------
# Plot: overlay uncoded & coded
# --------------------
plt.figure(figsize=(9,6))
plt.semilogy(EbNoDB, BER_uncoded_MMSE, 'rs-.', label='MMSE (uncoded)')
plt.semilogy(EbNoDB, BER_uncoded_ESN,  'g^--', label='ESN (uncoded)')
plt.semilogy(EbNoDB, BER_coded_MMSE, 'r*-',  label='MMSE + LDPC (calibrated LLRs)')
plt.semilogy(EbNoDB, BER_coded_ESN,  'g*-',  label='ESN + LDPC (calibrated LLRs)')
plt.grid(True, which='both', ls=':')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER (bit or info-bit)')
plt.title('4x8 MIMO — Uncoded vs Coded (LDPC) with LLR Calibration\nMMSE vs ESN — CDL-B (TDL) Channel + PA')
plt.legend()
plt.tight_layout()
outpng = os.path.join(outdir, "BER_uncoded_coded_overlay_MMSE_ESN.png")
plt.savefig(outpng, dpi=150)
plt.show()

# ---- Save a compact results bundle for notebooks ----
results_ber = {
    "EBN0": EbNoDB.tolist(),
    "BER_uncoded": {
        "ESN":  BER_uncoded_ESN.tolist(),
        "MMSE": BER_uncoded_MMSE.tolist(),
    },
    "BER_coded": {
        "ESN_calLLR":  BER_coded_ESN.tolist(),
        "MMSE_calLLR": BER_coded_MMSE.tolist(),
    },
    "meta": {
        "N": int(N), "N_t": int(N_t), "N_r": int(N_r),
        "IsiDuration": int(IsiDuration), "CP": int(CyclicPrefixLen),
        "NumOfdmSymbols": int(NumOfdmSymbols),
        "LDPC": {"n": int(n_code), "k": int(k_info), "dv": int(LDPC_dv), "dc": int(LDPC_dc), "maxiter": int(LDPC_MAXITER)},
        "LLR_clip": float(LLR_CLIP),
        "train_EbNo_dB": int(TRAIN_EBNO_FIXED_DB),
        "esn": {
            "n_reservoir": int(nInternalUnits),
            "spectral_radius": float(spectralRadius),
            "input_scaler": float(inputScaler),
            "teacher_scaling_base": float(teacherScalingBase),
        },
        "channel": {
            "model": "CDL-B (TDL-equivalent)",
            "DS_ns": float(CDLB_DS_NS),
            "norm_delays": _TDLB_NORM_DELAYS.tolist(),
            "pow_dB": _TDLB_POW_DB.tolist(),
            "fs_Hz": float(W),
        }
    }
}
with open(os.path.join(outdir, "results_ber_4x8_cdLB.pkl"), "wb") as f:
    pickle.dump(results_ber, f)

print("Saved:")
print(f" - {outpng}")
print(f" - {outdir}/LLR_calibration_params_EbNo*dB.txt")
print(f" - {outdir}/results_ber_4x8_cdLB.pkl")

# --- Save plot + results to a specific folder ---

import csv

# Set your target folder
OUT_DIR = "./results_4x8/CDLB_run_01"  # <--- change this path as you like
os.makedirs(OUT_DIR, exist_ok=True)

# ----- Plot and save -----
fig = plt.figure(figsize=(9,6))
plt.semilogy(EbNoDB, BER_uncoded_MMSE, 'rs-.', label='MMSE (uncoded)')
plt.semilogy(EbNoDB, BER_uncoded_ESN,  'g^--', label='ESN (uncoded)')
plt.semilogy(EbNoDB, BER_coded_MMSE,  'r*-',  label='MMSE + LDPC (calibrated LLRs)')
plt.semilogy(EbNoDB, BER_coded_ESN,   'g*-',  label='ESN + LDPC (calibrated LLRs)')
plt.grid(True, which='both', ls=':')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER (bit or info-bit)')
plt.title('4x8 MIMO — Uncoded vs Coded (LDPC) with LLR Calibration\nMMSE vs ESN — CDL-B (TDL)')
plt.legend()
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "BER_uncoded_coded_overlay_MMSE_ESN.png")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# ----- Save a compact results bundle (PKL) -----
results_ber = {
    "EBN0": EbNoDB.tolist(),
    "BER_uncoded": {
        "ESN":  BER_uncoded_ESN.tolist(),
        "MMSE": BER_uncoded_MMSE.tolist(),
    },
    "BER_coded": {
        "ESN_calLLR":  BER_coded_ESN.tolist(),
        "MMSE_calLLR": BER_coded_MMSE.tolist(),
    },
}
pkl_path = os.path.join(OUT_DIR, "results_ber.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(results_ber, f)

# ----- (Optional) Save a CSV too -----
csv_path = os.path.join(OUT_DIR, "results_ber.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["EbNo(dB)", "ESN_uncoded", "MMSE_uncoded", "ESN_coded", "MMSE_coded"])
    for i, snr in enumerate(EbNoDB):
        w.writerow([int(snr), BER_uncoded_ESN[i], BER_uncoded_MMSE[i], BER_coded_ESN[i], BER_coded_MMSE[i]])

print("Saved:")
print(f" - {png_path}")
print(f" - {pkl_path}")
print(f" - {csv_path}")
