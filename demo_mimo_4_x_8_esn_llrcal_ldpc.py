# Demo_MIMO_4x8_ESN_LLRCal_LDPC.py
# --------------------------------------------------
# 4x8 MIMO-OFDM with nonlinear PA + block fading
# Compare MMSE vs ESN with an LLR-calibration head trained by cross-entropy.
#
# Pipeline:
#  - ESN does time-domain recovery -> FFT -> raw soft symbols
#  - MMSE equalizer gives soft symbols too
#  - For each method, compute raw max-log LLRs (per bit)
#  - Train a tiny logistic calibrator:  LLR_cal_b = a_b * LLR_raw_b + b_b
#    by minimizing cross-entropy on a training subset of symbols
#  - Feed calibrated LLRs to LDPC decoder, measure post-LDPC BER
#  - Also plot pre-decoder (uncoded) BER on the same axes
#
# Saves figures:
#   results_4x8/BER_uncoded_coded_overlay_MMSE_ESN.png
#   results_4x8/LLR_calibration_params_EbNo<dB>.txt  (a,b per bit for ESN & MMSE)
#
# Requirements:
#   pip install pyESN pyldpc scipy matplotlib
#   have helper_mimo_esn_generic.py available
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
    xs = []
    for tx in range(N_t):
        re_col = 2*tx
        im_col = 2*tx + 1
        re_seq = x_hat_tmp[Delay[re_col]-Delay_Min: Delay[re_col]-Delay_Min + N + 1, re_col]
        im_seq = x_hat_tmp[Delay[im_col]-Delay_Min: Delay[imcol]-Delay_Min + N + 1, im_col]
        xs.append(re_seq + 1j*im_seq)
    return xs

def qam_bit_labels(M, m):
    labels = np.zeros((M, m), dtype=int)
    for idx in range(M):
        labels[idx, :] = bits_to_grayvec(idx, m)
    return labels

def qam_llrs_maxlog(z_hat_1d, const, bit_labels, sigma2):
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
# System parameters
# --------------------
W = 2*1.024e6
f_D = 100
No = 1e-5
IsiDuration = 8
EbNoDB = np.arange(0, 31, 3).astype(np.int32)  # full grid

N_t = 4
N_r = 8

N = 512
m = 4
m_pilot = 4
NumOfdmSymbols = 250        # moderate runtime
Ptotal = 10**(EbNoDB/10)*No*N

p_smooth = 1
ClipLeveldB = 3

T_OFDM_Total = (N + IsiDuration - 1)/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)
Pi = Ptotal/N
NumBitsPerSymbol = m*N
Const = np.array(unit_qam_constellation(m)).astype(complex)
ConstPilot = np.array(unit_qam_constellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)

# ESN params
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2*N_r
nOutputUnits = 2*N_t
nInternalUnits = 500
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
LLR_CLIP = 5.0

n_code = N * m
if n_code % LDPC_dc != 0:
    raise ValueError(f"LDPC_dc={LDPC_dc} must divide n_code={n_code}.")
H, G = make_ldpc(n_code, LDPC_dv, LDPC_dc, systematic=True, sparse=True)
k_info = G.shape[1]
BIT_LABELS = qam_bit_labels(2**m, m)
print(f"[LDPC] Built regular code: n={n_code}, k≈{k_info}, rate≈{k_info/n_code:.3f}")

# Calibration/train/test split
CAL_FRAC = 0.3   # fraction of data symbols used to fit a,b

# Holders
BER_uncoded_ESN = np.zeros(len(EbNoDB))
BER_uncoded_MMSE = np.zeros(len(EbNoDB))
BER_coded_ESN   = np.zeros(len(EbNoDB))
BER_coded_MMSE  = np.zeros(len(EbNoDB))

outdir = "./results_4x8"
os.makedirs(outdir, exist_ok=True)

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

    for kk in range(1, NumOfdmSymbols+1):
        redraw = (np.remainder(kk, L) == 1)
        if redraw:
            c = [[None for _ in range(N_t)] for __ in range(N_r)]
            H_true = np.zeros((N, N_r, N_t), dtype=complex)
            for nr in range(N_r):
                for nt in range(N_t):
                    c0 = (np.random.randn(IsiDuration) + 1j*np.random.randn(IsiDuration))/np.sqrt(2)
                    c0 *= np.sqrt(IsiMagnitude[:IsiDuration])
                    c[nr][nt] = c0
                    H_true[:, nr, nt] = np.fft.fft(np.r_[c0, np.zeros(N - len(c0))])

            TxBitsPilot = (np.random.rand(N*m_pilot, N_t) > 0.5).astype(np.int32)
            X_p = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), tx])[0])
                    X_p[ii, tx] = ConstPilot[idx]
            X_LS = np.zeros_like(X_p)
            for tx in range(N_t):
                X_LS[tx::N_t, tx] = X_p[tx::N_t, tx]

            x_temp = np.zeros((N, N_t), dtype=complex)
            x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
            x_LS_CP = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_temp[:, tx] = N * np.fft.ifft(X_p[:, tx])
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
                x_temp_ls = N * np.fft.ifft(X_LS[:, tx])
                x_LS_CP[:, tx] = np.r_[x_temp_ls[-CyclicPrefixLen:], x_temp_ls] * (Pi[jj]**0.5)

            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            x_LS_CP_NLD = x_LS_CP / ((1 + (np.abs(x_LS_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            y_LS_CP = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
                    y_LS_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_LS_CP_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP[:, nr] += noise
                y_LS_CP[:, nr] += noise

            Y_p = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)
            Y_LS = (1/N) * np.fft.fft(y_LS_CP[CyclicPrefixLen:, :], axis=0)

            H_LS = np.zeros_like(H_true)
            H_MMSE = np.zeros_like(H_true)
            MMSEScaler = (No/Pi[jj])/(N/2)
            for nr in range(N_r):
                for tx in range(N_t):
                    sc_idx = np.arange(tx, N, N_t)
                    denom = (X_LS[sc_idx, tx] * (Pi[jj]**0.5) + 1e-12)
                    Hls_sc = Y_LS[sc_idx, nr] / denom
                    tmpf = interpolate.interp1d(sc_idx, Hls_sc, kind='linear', bounds_error=False, fill_value='extrapolate')
                    Hls_full = tmpf(np.arange(N))

                    c_LS = np.fft.ifft(Hls_full)
                    c_LS_trunc = c_LS[:IsiDuration]
                    c_MMSE = np.linalg.solve(np.dot(np.linalg.inv(R_h), MMSEScaler) + np.eye(IsiDuration), c_LS_trunc)
                    Hmmse_full = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

                    H_LS[:, nr, tx] = Hls_full
                    H_MMSE[:, nr, tx] = Hmmse_full

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
        TxBits = np.zeros((N*m, N_t), dtype=np.int8)
        InfoBits = [None]*N_t
        for tx in range(N_t):
            u = np.random.randint(0, 2, size=(k_info,), dtype=np.int8)
            cword = ldpc_encode_bits(G, u)
            InfoBits[tx] = u
            TxBits[:, tx] = cword

        X = np.zeros((N, N_t), dtype=complex)
        for ii in range(N):
            for tx in range(N_t):
                bits_idx = TxBits[m*ii + np.arange(m), tx]
                idx = int((PowersOfTwo @ bits_idx)[0])
                X[ii, tx] = Const[idx]

        x_temp = np.zeros((N, N_t), dtype=complex)
        x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
        for tx in range(N_t):
            x_temp[:, tx] = N * np.fft.ifft(X[:, tx])
            x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
        x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

        y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
        for nr in range(N_r):
            for tx in range(N_t):
                y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
            noise = math.sqrt(len(y_CP[:, nr])*No/2) * (np.random.randn(len(y_CP[:, nr])) + 1j*np.random.randn(len(y_CP[:, nr])))
            y_CP[:, nr] += noise

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

        # MMSE
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
            for b in range(m):
                xcal_esn[b].append( llr_esn_all[:, b, :].reshape(-1) )
                ycal_esn[b].append( bits_all[:, b, :].reshape(-1) )
                xcal_mmse[b].append( llr_mmse_all[:, b, :].reshape(-1) )
                ycal_mmse[b].append( bits_all[:, b, :].reshape(-1) )
        else:
            snr_for_ldpc = 1.0
            for tx in range(N_t):
                # ESN calibrated
                llr_esn_resh = llr_esn_all[:, :, tx]
                for b in range(m):
                    llr_esn_resh[:, b] = np.clip(a_esn[b]*llr_esn_resh[:, b] + b_esn[b], -LLR_CLIP, LLR_CLIP)
                yobs_esn = (llr_esn_resh.reshape(-1))/2.0
                d_hat_esn = ldpc_decode(H, yobs_esn, snr_for_ldpc, maxiter=LDPC_MAXITER)
                u_hat_esn = get_message(G, d_hat_esn).astype(np.int8)

                # MMSE calibrated
                llr_mmse_resh = llr_mmse_all[:, :, tx]
                for b in range(m):
                    llr_mmse_resh[:, b] = np.clip(a_mmse[b]*llr_mmse_resh[:, b] + b_mmse[b], -LLR_CLIP, LLR_CLIP)
                yobs_mmse = (llr_mmse_resh.reshape(-1))/2.0
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
plt.title('4x8 MIMO — Uncoded vs Coded (LDPC) with LLR Calibration\nMMSE vs ESN')
plt.legend()
plt.tight_layout()
outpng = os.path.join(outdir, "BER_uncoded_coded_overlay_MMSE_ESN.png")
plt.savefig(outpng, dpi=150)
plt.show()

print("Saved:")
print(f" - {outpng}")
print(f" - {outdir}/LLR_calibration_params_EbNo*dB.txt")
