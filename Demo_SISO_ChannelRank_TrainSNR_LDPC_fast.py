# Demo_SISO_LDPC_FastOptions_like_goodSISO.py
# --------------------------------------------------
# SISO OFDM + nonlinear PA + block fading multipath (same as your "good SISO")
# ESN training-SNR study (matched & fixed) + LS/MMSE/Perfect baselines
# Adds LDPC decoding (subsampled), LLR clipping, adaptive LDPC iterations,
# and a FAST mode knob for quick runs.
# --------------------------------------------------

import os
import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from HelpFunc import HelpFunc
from pyESN import ESN

import scipy.sparse as sp
from pyldpc import make_ldpc, decode as ldpc_decode, get_message

# --------------- FAST knob ---------------
FAST = True       # Set True for quick runs
# -----------------------------------------

# --------------------
# Physical parameters  (kept as in good SISO)
# --------------------
W = 2*1.024e6          # Available Bandwidth
f_D = 100              # Doppler Frequency
No = 1e-5              # Noise PSD
IsiDuration = 8        # Number of multipath taps
EbNoDB = np.arange(0, 31, 3).astype(np.int32) if FAST else np.arange(0, 31, 2).astype(np.int32)

# --------------------
# Antenna (SISO)
# --------------------
N_t = 1
N_r = 1

# --------------------
# Design parameters (kept as in good SISO)
# --------------------
N = 256 if FAST else 512       # Subcarriers (FAST shrinks)
m = 4                          # QAM order (16-QAM)
m_pilot = 4                    # pilot QAM order (16-QAM)
NumOfdmSymbols = 300 if FAST else 400
Ptotal = 10**(EbNoDB/10)*No*N  # total power per OFDM symbol

# --------------------
# Nonlinear PA
# --------------------
p_smooth = 1
ClipLeveldB = 3

# --------------------
# Secondary parameters
# --------------------
T_OFDM_Total = (N + IsiDuration - 1)/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)           # Coherence time (OFDM symbols)
Pi = Ptotal/N                                # equal power per subcarrier
NumBitsPerSymbol = m*N
Const = np.array(HelpFunc.UnitQamConstellation(m)).astype(complex)
ConstPilot = np.array(HelpFunc.UnitQamConstellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

# Exponential power-delay profile (normalized)
temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)

# --------------------
# ESN parameters (kept like good SISO; dense reservoir for fast convergence)
# --------------------
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2          # Re/Im of single Rx stream
nOutputUnits = 2         # Re/Im of single Tx stream
nInternalUnits = 300 if FAST else 400   # you can keep 100; use 300 in FAST if you want extra stability
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

teacherShift = np.zeros(nOutputUnits)
feedbackScaling = feedbackScaler*np.ones(nOutputUnits)

Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2
DelayFlag = 0  # 0 => same delay for Re/Im; 1 => independent (kept as in good SISO)

# Training-SNR mismatch setting (kept)
TRAIN_EBNO_FIXED_DB = 12
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# --------------------
# LDPC settings (new)
# --------------------
USE_LDPC = True
LDPC_dv = 4
LDPC_dc = 8
LDPC_MAXITER = (80 if FAST else 100)
LDPC_DECODE_EVERY = (8 if FAST else 4)  # decode every k-th data OFDM symbol
LLR_CLIP = 20.0
LLR_POLARITY = -0.5   # y_obs = LLR_POLARITY * clip(LLR); flip sign if needed once

n_code = N * m
if n_code % LDPC_dc != 0:
    raise ValueError(f"LDPC_dc={LDPC_dc} must divide n_code={n_code}.")
H, G = make_ldpc(n_code, LDPC_dv, LDPC_dc, systematic=True, sparse=True)
k_info = G.shape[1]
print(f"[LDPC] Built regular code: n={n_code}, k≈{k_info}, rate≈{k_info/n_code:.3f}")

# --------------------
# Holders
# --------------------
BER_ESN_matched = np.zeros(len(EbNoDB))
BER_ESN_trainFixed = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

BERC_ESN_matched = np.zeros(len(EbNoDB))
BERC_ESN_trainFixed = np.zeros(len(EbNoDB))
BERC_Perfect = np.zeros(len(EbNoDB))
BERC_LS = np.zeros(len(EbNoDB))
BERC_MMSE = np.zeros(len(EbNoDB))

# Channel metrics to save
Capacity_bits_per_sc = np.zeros(len(EbNoDB))   # average over subcarriers
Sval_mean = np.zeros(len(EbNoDB))              # mean |H_k|
Sval_p10 = np.zeros(len(EbNoDB))               # 10th percentile |H_k|
Sval_p90 = np.zeros(len(EbNoDB))               # 90th percentile |H_k|

# MMSE constants (TD prior)
R_h = np.diag(IsiMagnitude[:IsiDuration])

# --------------------
# Helpers
# --------------------
def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]   # LSB-first

def hard_bits_from_syms_1d(Xhat):
    RxBits = np.zeros((N*m, 1), dtype=int)
    for ii in range(N):
        sym = Xhat[ii]
        idx = int(np.argmin(np.abs(Const - sym)))
        RxBits[m*ii:m*(ii+1), 0] = bits_to_grayvec(idx, m)
    return RxBits

def qam_bit_labels(M, m):
    labels = np.zeros((M, m), dtype=int)
    for idx in range(M):
        labels[idx, :] = bits_to_grayvec(idx, m)
    return labels

BIT_LABELS = qam_bit_labels(2**m, m)

def qam_llrs_maxlog(z_hat_1d, const, bit_labels, sigma2):
    Nloc = z_hat_1d.shape[0]
    mloc = bit_labels.shape[1]
    llrs = np.zeros((Nloc, mloc), dtype=float)
    d2 = np.abs(z_hat_1d.reshape(-1,1) - const.reshape(1,-1))**2
    s2 = max(float(sigma2), 1e-12)
    for b in range(mloc):
        d0 = np.min(d2[:, bit_labels[:, b]==0], axis=1)
        d1 = np.min(d2[:, bit_labels[:, b]==1], axis=1)
        llrs[:, b] = (d1 - d0) / s2   # + => bit 0 more likely
    return llrs.reshape(-1)

def est_sigma2_from_decision(Xhat_col, const):
    idx = np.argmin(np.abs(Xhat_col.reshape(-1,1) - const.reshape(1,-1))**2, axis=1)
    Xhard = const[idx]
    err = Xhat_col - Xhard
    return float(np.mean(np.abs(err)**2) + 1e-12)

def ldpc_encode_bits(G, u):
    x = G.dot(u) % 2 if sp.issparse(G) else (G @ u % 2)
    return np.asarray(x).ravel().astype(np.int8)

# ESN training on pilot (kept logic; trimmed to exact N for robustness)
def train_esn_from_pilot(y_cp, x_cp, inputScaling_used, inputShift_used):
    # Build delay LUT (same style as good SISO)
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

    # Dense reservoir: match fast convergence behavior
    esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
              spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
              input_shift=inputShift_used, input_scaling=inputScaling_used,
              teacher_scaling=teacherScalingBase*np.ones(nOutputUnits),
              teacher_shift=teacherShift, feedback_scaling=feedbackScaling)

    for didx in range(Delay_LUT.shape[0]):
        d0, d1 = Delay_LUT[didx, :]
        dmax = Delay_Max_vec[didx]
        dmin = Delay_Min_vec[didx]

        ESN_input = np.zeros((N + dmax + CyclicPrefixLen, nInputUnits))
        ESN_output = np.zeros((N + dmax + CyclicPrefixLen, nOutputUnits))

        ESN_input[:, 0] = np.r_[y_cp.real, np.zeros(dmax)]
        ESN_input[:, 1] = np.r_[y_cp.imag, np.zeros(dmax)]
        ESN_output[d0:(d0+N+CyclicPrefixLen), 0] = x_cp.real
        ESN_output[d1:(d1+N+CyclicPrefixLen), 1] = x_cp.imag

        nForget_tmp = int(dmin + CyclicPrefixLen)
        esn.fit(ESN_input, ESN_output, nForget_tmp)
        x_hat_tmp = esn.predict(ESN_input, nForget_tmp, continuation=False)

        # Trim to exact N (fix off-by-one)
        x_hat_td = (x_hat_tmp[d0 - dmin : d0 - dmin + N, 0]
                    + 1j * x_hat_tmp[d1 - dmin : d1 - dmin + N, 1])

        x_ref = x_cp[CyclicPrefixLen:]
        NMSE_list[didx] = np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2

    didx_best = int(np.argmin(NMSE_list))
    Delay_sel = Delay_LUT[didx_best, :]
    Delay_Min_sel = int(Delay_Min_vec[didx_best])
    Delay_Max_sel = int(Delay_Max_vec[didx_best])

    # Final training with chosen delays
    ESN_input = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nInputUnits))
    ESN_output = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nOutputUnits))
    ESN_input[:, 0] = np.r_[y_cp.real, np.zeros(Delay_Max_sel)]
    ESN_input[:, 1] = np.r_[y_cp.imag, np.zeros(Delay_Max_sel)]
    ESN_output[Delay_sel[0]:(Delay_sel[0]+N+CyclicPrefixLen), 0] = x_cp.real
    ESN_output[Delay_sel[1]:(Delay_sel[1]+N+CyclicPrefixLen), 1] = x_cp.imag

    nForget_final = int(Delay_Min_sel + CyclicPrefixLen)
    esn.fit(ESN_input, ESN_output, nForget_final)

    return esn, Delay_sel, Delay_Min_sel, Delay_Max_sel, nForget_final

# --------------------
# Output paths
# --------------------
outdir = "./results_siso"
os.makedirs(outdir, exist_ok=True)
SUFFIX = "_FAST" if FAST else ""

# --------------------
# Run per SNR
# --------------------
t_start_total = time.time()

for jj, ebno_db in enumerate(EbNoDB):
    print(f"\n=== Eb/No {ebno_db} dB ===")
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    # ESN scaling per SNR (matched)
    inputScaling_matched = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched = (inputOffset/inputScaler) * np.ones(nInputUnits)

    # ESN scaling for the fixed-train ESN
    inputScaling_trainFixed = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFixed = (inputOffset/inputScaler) * np.ones(nInputUnits)

    # Reset BER counters
    TotalErr_ESN_matched = 0
    TotalErr_ESN_trainFixed = 0
    TotalErr_LS = 0
    TotalErr_MMSE = 0
    TotalErr_Perfect = 0
    TotalBits = 0

    # Coded (LDPC) counters
    TotalErrC_ESN_matched = 0
    TotalErrC_ESN_trainFixed = 0
    TotalErrC_LS = 0
    TotalErrC_MMSE = 0
    TotalErrC_Perfect = 0
    TotalInfoBits = 0

    # MMSE TD matrix
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), (No/Pi[jj])/(N/2)) + np.eye(IsiDuration)

    # placeholders carried from training to data
    esn_matched = None
    esn_trainFixed = None
    Delay_m = Delay_Min_m = Delay_Max_m = nForget_m = None
    Delay_f = Delay_Min_f = Delay_Max_f = nForget_f = None

    # channel stats accumulators
    cap_acc = []
    sabs_acc = []

    for kk in range(1, NumOfdmSymbols+1):
        # New channel realization at start of each coherence block
        if (np.remainder(kk, L) == 1):
            # Random SISO channel (time domain) and its FFT
            c0 = (np.random.randn(IsiDuration) + 1j*np.random.randn(IsiDuration))/np.sqrt(2)
            c0 = c0 * (IsiMagnitude[:IsiDuration]**0.5)
            H_true = np.fft.fft(np.r_[c0, np.zeros(N - len(c0))])

            # -------- Pilot OFDM (matched SNR) --------
            TxBitsPilot = (np.random.rand(N*m_pilot, 1) > 0.5).astype(np.int32)
            X_p = np.zeros((N,), dtype=complex)
            for ii in range(N):
                idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), 0])[0])
                X_p[ii] = ConstPilot[idx]

            # IFFT + CP and power loading with matched Pi
            x_temp = N * np.fft.ifft(X_p)
            x_CP_p = np.r_[x_temp[-CyclicPrefixLen:], x_temp] * (Pi[jj]**0.5)

            # Nonlinearity
            x_CP_p_NLD = x_CP_p / ((1 + (np.abs(x_CP_p)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + AWGN
            y_CP_p = signal.lfilter(c0, np.array([1]), x_CP_p_NLD)
            noise = math.sqrt(len(y_CP_p)*No/2) * (np.random.randn(len(y_CP_p)) + 1j*np.random.randn(len(y_CP_p)))
            y_CP_p = y_CP_p + noise

            # Freq-domain received pilots
            Y_p = (1/N) * np.fft.fft(y_CP_p[CyclicPrefixLen:])

            # LS / MMSE
            H_LS = (Y_p / (X_p + 1e-12)) / (Pi[jj]**0.5)
            c_LS = np.fft.ifft(H_LS)
            c_LS_trunc = c_LS[:IsiDuration]
            c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS_trunc)
            H_MMSE = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

            # -------- Pilot OFDM (fixed training SNR) --------
            x_CP_pf = np.r_[x_temp[-CyclicPrefixLen:], x_temp] * (Pi_train_fixed**0.5)
            A_Clip_train = np.sqrt(var_x_train_fixed) * np.float_power(10, ClipLeveldB/20)
            x_CP_pf_NLD = x_CP_pf / ((1 + (np.abs(x_CP_pf)/A_Clip_train)**(2*p_smooth))**(1/(2*p_smooth)))
            y_CP_pf = signal.lfilter(c0, np.array([1]), x_CP_pf_NLD) + noise  # reuse noise draw

            # -------- Train ESNs (matched & fixed-train) --------
            esn_matched, Delay_m, Delay_Min_m, Delay_Max_m, nForget_m = \
                train_esn_from_pilot(y_CP_p, x_CP_p, inputScaling_matched, inputShift_matched)
            esn_trainFixed, Delay_f, Delay_Min_f, Delay_Max_f, nForget_f = \
                train_esn_from_pilot(y_CP_pf, x_CP_pf, inputScaling_trainFixed, inputShift_trainFixed)

            # Channel metrics (capacity and |H_k| stats)
            snr_sc = (Pi[jj]/No) * (np.abs(H_true)**2)  # per-subcarrier post-channel SNR
            cap = np.mean(np.log2(1 + snr_sc))
            sabs = np.abs(H_true)
            cap_acc.append(cap)
            sabs_acc.append(sabs)

        else:
            # -------- Data OFDM symbol --------
            decode_this_symbol = USE_LDPC and ((kk % LDPC_DECODE_EVERY) == 1)

            if USE_LDPC:
                # LDPC-coded bits
                u = np.random.randint(0, 2, size=(k_info,), dtype=np.int8)
                cword = ldpc_encode_bits(G, u)
                TxBits = cword.reshape(-1,1)
            else:
                # Uncoded bits
                TxBits = (np.random.rand(N*m, 1) > 0.5).astype(np.int32)

            # Map bits -> QAM
            X = np.zeros((N,), dtype=complex)
            for ii in range(N):
                idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), 0])[0])
                X[ii] = Const[idx]

            # IFFT + CP + PA + channel + noise
            x_temp = N * np.fft.ifft(X)
            x_CP = np.r_[x_temp[-CyclicPrefixLen:], x_temp] * (Pi[jj]**0.5)
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP = signal.lfilter(c0, np.array([1]), x_CP_NLD)   # pass through same c0 used for this block
            noise = math.sqrt(len(y_CP)*No/2) * (np.random.randn(len(y_CP)) + 1j*np.random.randn(len(y_CP)))
            y_CP = y_CP + noise

            # Frequency-domain receive
            Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:])

            # -------- ESN Detector predictions (two ESNs) --------
            def esn_recover_to_freq(esn_obj, Delay_sel, Delay_Min_sel, Delay_Max_sel, nForget_sel):
                ESN_input = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nInputUnits))
                ESN_input[:, 0] = np.r_[y_CP.real, np.zeros(Delay_Max_sel)]
                ESN_input[:, 1] = np.r_[y_CP.imag, np.zeros(Delay_Max_sel)]
                x_hat_tmp = esn_obj.predict(ESN_input, nForget_sel, continuation=False)
                # Trim to exact N (fix off-by-one)
                x_hat_td = (x_hat_tmp[Delay_sel[0] - Delay_Min_sel : Delay_sel[0] - Delay_Min_sel + N, 0]
                            + 1j * x_hat_tmp[Delay_sel[1] - Delay_Min_sel : Delay_sel[1] - Delay_Min_sel + N, 1])
                return (1/N) * np.fft.fft(x_hat_td) / math.sqrt(Pi[jj])

            X_hat_ESN_m = esn_recover_to_freq(esn_matched,    Delay_m, Delay_Min_m, Delay_Max_m, nForget_m)
            X_hat_ESN_f = esn_recover_to_freq(esn_trainFixed, Delay_f, Delay_Min_f, Delay_Max_f, nForget_f)

            # -------- Channel-equalized baselines --------
            X_hat_Perfect = (Y / (H_true + 1e-12)) / math.sqrt(Pi[jj])
            X_hat_LS      = (Y / (H_LS   + 1e-12)) / math.sqrt(Pi[jj])
            X_hat_MMSE    = (Y / (H_MMSE + 1e-12)) / math.sqrt(Pi[jj])

            # -------- Uncoded BER (pre-decoder) --------
            RxBits_ESN_m  = hard_bits_from_syms_1d(X_hat_ESN_m)
            RxBits_ESN_f  = hard_bits_from_syms_1d(X_hat_ESN_f)
            RxBits_LS     = hard_bits_from_syms_1d(X_hat_LS)
            RxBits_MMSE   = hard_bits_from_syms_1d(X_hat_MMSE)
            RxBits_Perfect= hard_bits_from_syms_1d(X_hat_Perfect)

            TotalErr_ESN_matched    += int(np.sum(TxBits != RxBits_ESN_m))
            TotalErr_ESN_trainFixed += int(np.sum(TxBits != RxBits_ESN_f))
            TotalErr_LS             += int(np.sum(TxBits != RxBits_LS))
            TotalErr_MMSE           += int(np.sum(TxBits != RxBits_MMSE))
            TotalErr_Perfect        += int(np.sum(TxBits != RxBits_Perfect))
            TotalBits               += NumBitsPerSymbol

            # -------- Coded BER (LDPC) --------
            if USE_LDPC and decode_this_symbol:
                def decode_from_soft(Xhat):
                    sigma2_eff = est_sigma2_from_decision(Xhat, Const)
                    llr_vec = qam_llrs_maxlog(Xhat, Const, BIT_LABELS, sigma2_eff)
                    llr_vec = np.clip(llr_vec, -LLR_CLIP, LLR_CLIP)
                    # BPSK-like observation for pyldpc; negative sign matches our LLR polarity
                    y_obs = LLR_POLARITY * llr_vec
                    # a few extra iters at low SNR helps convergence
                    maxiter = LDPC_MAXITER if ebno_db >= 6 else 2*LDPC_MAXITER
                    d_hat = ldpc_decode(H, y_obs, snr=1.0, maxiter=maxiter)
                    u_hat = get_message(G, d_hat).astype(np.int8)
                    return u_hat

                u_true = get_message(G, TxBits.ravel().astype(np.int8)).astype(np.int8)  # since TxBits is codeword
                u_esn_m = decode_from_soft(X_hat_ESN_m)
                u_esn_f = decode_from_soft(X_hat_ESN_f)
                u_ls    = decode_from_soft(X_hat_LS)
                u_mmse  = decode_from_soft(X_hat_MMSE)
                u_perf  = decode_from_soft(X_hat_Perfect)

                TotalErrC_ESN_matched    += int(np.sum(u_true != u_esn_m))
                TotalErrC_ESN_trainFixed += int(np.sum(u_true != u_esn_f))
                TotalErrC_LS             += int(np.sum(u_true != u_ls))
                TotalErrC_MMSE           += int(np.sum(u_true != u_mmse))
                TotalErrC_Perfect        += int(np.sum(u_true != u_perf))
                TotalInfoBits            += int(len(u_true))

    # Store BER per SNR
    BER_ESN_matched[jj]    = TotalErr_ESN_matched / max(TotalBits, 1)
    BER_ESN_trainFixed[jj] = TotalErr_ESN_trainFixed / max(TotalBits, 1)
    BER_LS[jj]             = TotalErr_LS / max(TotalBits, 1)
    BER_MMSE[jj]           = TotalErr_MMSE / max(TotalBits, 1)
    BER_Perfect[jj]        = TotalErr_Perfect / max(TotalBits, 1)

    if USE_LDPC and TotalInfoBits > 0:
        BERC_ESN_matched[jj]    = TotalErrC_ESN_matched    / TotalInfoBits
        BERC_ESN_trainFixed[jj] = TotalErrC_ESN_trainFixed / TotalInfoBits
        BERC_LS[jj]             = TotalErrC_LS             / TotalInfoBits
        BERC_MMSE[jj]           = TotalErrC_MMSE           / TotalInfoBits
        BERC_Perfect[jj]        = TotalErrC_Perfect        / TotalInfoBits

    # Channel metrics (average over blocks observed at this SNR)
    if len(cap_acc) > 0:
        Capacity_bits_per_sc[jj] = float(np.mean(cap_acc))
        sab = np.concatenate(sabs_acc)
        Sval_mean[jj] = float(np.mean(sab))
        Sval_p10[jj]  = float(np.percentile(sab, 10))
        Sval_p90[jj]  = float(np.percentile(sab, 90))

# -------- Save results (NEW SETUP: dedicated folder) --------
import os

SUFFIX = "_FAST" if FAST else ""
base_dir = os.path.dirname(os.path.abspath(__file__))          # script folder
outdir = os.path.join(base_dir, f"results_siso_ldpc{SUFFIX}")  # dedicated results folder
os.makedirs(outdir, exist_ok=True)

results_ber = {
    "EBN0": EbNoDB,
    "BER_pre": {
        "ESN_matched": BER_ESN_matched,
        "ESN_trainFixed": BER_ESN_trainFixed,
        "LS": BER_LS,
        "MMSE": BER_MMSE,
        "Perfect": BER_Perfect
    },
    "BER_post": {
        "ESN_matched": BERC_ESN_matched,
        "ESN_trainFixed": BERC_ESN_trainFixed,
        "LS": BERC_LS,
        "MMSE": BERC_MMSE,
        "Perfect": BERC_Perfect
    },
    "meta": {
        "TRAIN_EBNO_FIXED_DB": TRAIN_EBNO_FIXED_DB,
        "N": int(N), "m": int(m), "IsiDuration": int(IsiDuration),
        "ldpc": {"n": int(n_code), "k": int(k_info), "dv": int(LDPC_dv), "dc": int(LDPC_dc),
                 "maxiter": int(LDPC_MAXITER), "decode_every": int(LDPC_DECODE_EVERY),
                 "clip": float(LLR_CLIP), "polarity": float(LLR_POLARITY)}
    }
}
with open(os.path.join(outdir, f"BER_pre_post_LDPC_SISO{SUFFIX}.pkl"), "wb") as f:
    pickle.dump(results_ber, f)

results_channel = {
    "EBN0": EbNoDB,
    "capacity_bits_per_sc": Capacity_bits_per_sc,  # averaged over subcarriers and blocks
    "absHk_stats": {
        "mean": Sval_mean,
        "p10": Sval_p10,
        "p90": Sval_p90
    },
    "notes": "SISO: rank is 1; we log |H_k| stats and capacity."
}
with open(os.path.join(outdir, f"channel_metrics_siso{SUFFIX}.pkl"), "wb") as f:
    pickle.dump(results_channel, f)

# -------- Plot: pre-LDPC --------
plt.figure()
plt.semilogy(EbNoDB, BER_Perfect, 'kx-', label='Perfect CSI (pre)')
plt.semilogy(EbNoDB, BER_MMSE,    'rs-.', label='MMSE (pre)')
plt.semilogy(EbNoDB, BER_LS,      'o-',   label='LS (pre)')
plt.semilogy(EbNoDB, BER_ESN_matched,    'gd--', label='ESN matched (pre)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:',  label=f'ESN train @{TRAIN_EBNO_FIXED_DB} dB (pre)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title(f'SISO | Pre-LDPC BER | Nonlinear PA + Block Fading{SUFFIX}')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"BER_preLDPC_SISO{SUFFIX}.png"), dpi=150)
plt.show()

# -------- Plot: post-LDPC --------
if USE_LDPC and np.any(BERC_ESN_matched > 0):
    plt.figure()
    plt.semilogy(EbNoDB, BERC_Perfect, 'kx-', label='Perfect CSI (post-LDPC)')
    plt.semilogy(EbNoDB, BERC_MMSE,    'rs-.', label='MMSE (post-LDPC)')
    plt.semilogy(EbNoDB, BERC_LS,      'o-',   label='LS (post-LDPC)')
    plt.semilogy(EbNoDB, BERC_ESN_matched,    'gd--', label='ESN matched (post-LDPC)')
    plt.semilogy(EbNoDB, BERC_ESN_trainFixed, 'b^:',  label=f'ESN train @{TRAIN_EBNO_FIXED_DB} dB (post-LDPC)')
    plt.grid(True, which='both', ls=':'); plt.legend()
    plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('BER (info bits)')
    plt.title(f'SISO | Post-LDPC BER | Nonlinear PA + Block Fading{SUFFIX}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"BER_postLDPC_SISO{SUFFIX}.png"), dpi=150)
    plt.show()

# -------- Plot: ESN pre vs post (overlay) --------
if USE_LDPC:
    plt.figure()
    plt.semilogy(EbNoDB, BER_ESN_matched,    'gd--', label='ESN matched (pre)')
    plt.semilogy(EbNoDB, BERC_ESN_matched,   'g*-',  label='ESN matched (post)')
    plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:',  label=f'ESN @{TRAIN_EBNO_FIXED_DB} (pre)')
    plt.semilogy(EbNoDB, BERC_ESN_trainFixed,'b*-',  label=f'ESN @{TRAIN_EBNO_FIXED_DB} (post)')
    plt.grid(True, which='both', ls=':'); plt.legend()
    plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
    plt.title(f'SISO | ESN: Pre vs Post LDPC{SUFFIX}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"BER_ESN_pre_vs_postLDPC_SISO{SUFFIX}.png"), dpi=150)
    plt.show()

print("\nSaved:")
print(f" - {os.path.join(outdir, f'BER_preLDPC_SISO{SUFFIX}.png')}")
print(f" - {os.path.join(outdir, f'BER_postLDPC_SISO{SUFFIX}.png')}")
print(f" - {os.path.join(outdir, f'BER_ESN_pre_vs_postLDPC_SISO{SUFFIX}.png')}")
print(f" - {os.path.join(outdir, f'BER_pre_post_LDPC_SISO{SUFFIX}.pkl')}")
print(f" - {os.path.join(outdir, f'channel_metrics_siso{SUFFIX}.pkl')}")
