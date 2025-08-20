import numpy as np
import math
from scipy import signal, interpolate
from HelpFunc import HelpFunc
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
2x2 MIMO (N_t=2, N_r=2) Nonlinear Block-Fading OFDM with ESN
- Leaves pyESN.py and HelpFunc.py untouched (uses HelpFunc.trainMIMOESN).
- ESN operates on stacked TD observations from both Rx branches (Re/Im), and predicts both Tx streams (Re/Im).
- Baselines: ZF (Perfect/LS) and MMSE equalization per subcarrier.
- Adds:
  * ESN training-SNR study: 'matched' vs 'fixed @ TRAIN_EBNO_FIXED_DB'
  * Channel understanding outputs: per-subcarrier SVD (rank/condition number) and capacity.
- Fix: robust LS interpolation (no endpoint duplication, safe extrapolation/clamping).
No LDPC here.
"""

# --------------------
# Physical parameters
# --------------------
W = 2*1.024e6          # Available Bandwidth
f_D = 100              # Doppler Frequency
No = 1e-5              # Noise PSD
IsiDuration = 8        # Multipath taps
EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)

# --------------------
# Antennas
# --------------------
N_t = 2
N_r = 2

# --------------------
# Design parameters
# --------------------
N = 512                        # Subcarriers
m = 4                          # 16-QAM
m_pilot = 4                    # pilot QAM
NumOfdmSymbols = 400           # per SNR point
Ptotal = 10**(EbNoDB/10)*No*N  # total power per OFDM

# --------------------
# PA
# --------------------
p_smooth = 1
ClipLeveldB = 3

# --------------------
# Secondary params
# --------------------
T_OFDM_Total = (N + IsiDuration - 1)/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)    # coherence symbols
Pi = Ptotal/N                          # per-subcarrier power (per TX stream)
NumBitsPerSymbol = m*N
Const = np.array(HelpFunc.UnitQamConstellation(m)).astype(complex)
ConstPilot = np.array(HelpFunc.UnitQamConstellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

# Exponential channel power profile (normalized)
temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)

# --------------------
# ESN parameters
# --------------------
# transmit-variance proxy per SNR
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2*N_r      # Re/Im per Rx
nOutputUnits = 2*N_t     # Re/Im per Tx stream
nInternalUnits = 150
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

# Training-SNR mismatch setting
TRAIN_EBNO_FIXED_DB = 12  # one ESN trained once at 12 dB and reused at all test SNRs
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# --------------------
# Helpers
# --------------------
def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]

def equalize_zf(Yk, Hk, power_scale):
    # ZF: X_hat = (H^H H)^{-1} H^H Y
    HH = Hk.conj().T
    G = HH @ Hk
    # regularize tiny matrices to avoid singularities in deep fades
    G += 1e-12 * np.eye(G.shape[0], dtype=G.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def equalize_mmse(Yk, Hk, power_scale, noise_over_power):
    # MMSE: X_hat = (H^H H + (No/Pi) I)^{-1} H^H Y
    HH = Hk.conj().T
    G = HH @ Hk + noise_over_power * np.eye(Hk.shape[1], dtype=Hk.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def reconstruct_esn_outputs(x_hat_tmp, Delay, Delay_Min, N):
    # Rebuild per-TX time sequences from ESN outputs [Re0,Im0,Re1,Im1]
    x0 = (x_hat_tmp[Delay[0]-Delay_Min: Delay[0]-Delay_Min + N + 1, 0] +
          1j * x_hat_tmp[Delay[1]-Delay_Min: Delay[1]-Delay_Min + N + 1, 1])
    x1 = (x_hat_tmp[Delay[2]-Delay_Min: Delay[2]-Delay_Min + N + 1, 2] +
          1j * x_hat_tmp[Delay[3]-Delay_Min: Delay[3]-Delay_Min + N + 1, 3])
    return x0, x1

# --------------------
# Holders
# --------------------
BER_ESN_matched = np.zeros(len(EbNoDB))
BER_ESN_trainFixed = np.zeros(len(EbNoDB))
BER_PerfectZF = np.zeros(len(EbNoDB))
BER_LS_ZF = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

# channel metrics
Capacity_bits_per_sc = np.zeros(len(EbNoDB))
Frac_rank_ge2 = np.zeros(len(EbNoDB))
Cond_p50 = np.zeros(len(EbNoDB))
Cond_p90 = np.zeros(len(EbNoDB))

# MMSE constants
MMSEScaler_allSNR = (No/Pi) # vector across SNRs

# Time-domain correlation matrix
R_h = np.diag(IsiMagnitude[:IsiDuration])

for jj, ebno_db in enumerate(EbNoDB):
    print(f'EbNoDB = {ebno_db}')
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    # ESN scaling (matched)
    inputScaling_matched = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    # ESN scaling (fixed-train)
    inputScaling_trainFixed = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFixed = (inputOffset/inputScaler) * np.ones(nInputUnits)

    # Reset BER counters
    TotalErr_ESN_matched = 0
    TotalErr_ESN_trainFixed = 0
    TotalErr_LS_ZF = 0
    TotalErr_MMSE = 0
    TotalErr_PerfectZF = 0
    TotalBits = 0

    # MMSE TD matrix (per-link)
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler_allSNR[jj]/(N/2)) + np.eye(IsiDuration)

    # placeholders carried from training to data
    esn_matched = None
    esn_trainFixed = None
    Delay_m = None; Delay_Min_m = None; Delay_Max_m = None; nForget_m = None
    Delay_f = None; Delay_Min_f = None; Delay_Max_f = None; nForget_f = None

    # channel stats accumulators
    cap_acc = []
    cond_list = []
    rank_list = []

    for kk in range(1, NumOfdmSymbols+1):
        # New channel at start of coherence block
        if (np.remainder(kk, L) == 1):
            # 2x2 time-domain channels and FFTs
            c = [[None for _ in range(N_t)] for __ in range(N_r)]
            H_true = np.zeros((N, N_r, N_t), dtype=complex)
            for nr in range(N_r):
                for nt in range(N_t):
                    c0 = (np.random.randn(IsiDuration) + 1j*np.random.randn(IsiDuration))/np.sqrt(2)
                    c0 *= np.sqrt(IsiMagnitude[:IsiDuration])
                    c[nr][nt] = c0
                    H_true[:, nr, nt] = np.fft.fft(np.r_[c0, np.zeros(N - len(c0))])

            # Pilot bits & symbols (independent per TX)
            TxBitsPilot = (np.random.rand(N*m_pilot, N_t) > 0.5).astype(np.int32)
            X_p = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), tx])[0])
                    X_p[ii, tx] = ConstPilot[idx]

            # Build FDM pilots to separate TXs for LS
            X_LS = np.zeros_like(X_p)
            # TX0 pilots on even tones, TX1 on odd tones (keep their original symbols)
            X_LS[::2, 0] = X_p[::2, 0]
            X_LS[1::2, 1] = X_p[1::2, 1]

            # IFFT + CP (matched Pi) for pilots
            x_temp = np.zeros((N, N_t), dtype=complex)
            x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
            x_LS_CP = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_temp[:, tx] = N * np.fft.ifft(X_p[:, tx])
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
                x_temp_ls = N * np.fft.ifft(X_LS[:, tx])
                x_LS_CP[:, tx] = np.r_[x_temp_ls[-CyclicPrefixLen:], x_temp_ls] * (Pi[jj]**0.5)

            # Nonlinearity
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            x_LS_CP_NLD = x_LS_CP / ((1 + (np.abs(x_LS_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + noise for pilots
            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            y_LS_CP = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
                    y_LS_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_LS_CP_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP[:, nr] += noise
                y_LS_CP[:, nr] += noise

            # Frequency-domain received pilots
            Y_p = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)      # (N, Nr)
            Y_LS = (1/N) * np.fft.fft(y_LS_CP[CyclicPrefixLen:, :], axis=0)  # (N, Nr)

            # LS/MMSE per link using FDM pilots + robust interpolation
            H_LS = np.zeros_like(H_true)
            H_MMSE = np.zeros_like(H_true)

            for nr in range(N_r):
                for tx in range(N_t):
                    # pilot positions for this TX stream
                    sc_idx = np.arange(tx, N, N_t)  # tx=0 -> even, tx=1 -> odd
                    # LS on pilot subcarriers (avoid division by zero; X_LS is zero elsewhere)
                    Hls_sc = Y_LS[sc_idx, nr] / (X_LS[sc_idx, tx] * (Pi[jj]**0.5) + 1e-12)
                    # Interpolate to full N tones; allow safe extrapolation (or clamp edges)
                    tmpf = interpolate.interp1d(
                        sc_idx, Hls_sc, kind='linear',
                        bounds_error=False, fill_value='extrapolate'
                    )
                    Hls_full = tmpf(np.arange(N))

                    # TD trunc + TD-MMSE from full LS
                    c_LS = np.fft.ifft(Hls_full)
                    c_LS_trunc = c_LS[:IsiDuration]
                    c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS_trunc)
                    Hmmse_full = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

                    H_LS[:, nr, tx] = Hls_full
                    H_MMSE[:, nr, tx] = Hmmse_full

            # -------- Train ESNs (matched & fixed-train) --------
            # Matched
            esn_m = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                        input_shift=inputShift_matched, input_scaling=inputScaling_matched,
                        teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                        feedback_scaling=feedbackScaling)
            ESN_input, ESN_output, esn_m, Delay_m, Delay_Idx_m, Delay_Min_m, Delay_Max_m, nForget_m, _ = \
                HelpFunc.trainMIMOESN(esn_m, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen,
                                      N, N_t, N_r, IsiDuration, y_CP, x_CP)
            esn_matched = esn_m

            # Fixed-training-SNR pilot (reuse symbols, different power)
            x_CP_pf = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_CP_pf[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi_train_fixed**0.5)
            A_Clip_train = np.sqrt(var_x_train_fixed) * np.float_power(10, ClipLeveldB/20)
            x_CP_pf_NLD = x_CP_pf / ((1 + (np.abs(x_CP_pf)/A_Clip_train)**(2*p_smooth))**(1/(2*p_smooth)))
            y_CP_pf = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP_pf[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_pf_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP_pf[:, nr] += noise

            esn_f = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                        input_shift=inputShift_trainFixed, input_scaling=inputScaling_trainFixed,
                        teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                        feedback_scaling=feedbackScaling)
            ESN_input, ESN_output, esn_f, Delay_f, Delay_Idx_f, Delay_Min_f, Delay_Max_f, nForget_f, _ = \
                HelpFunc.trainMIMOESN(esn_f, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen,
                                      N, N_t, N_r, IsiDuration, y_CP_pf, x_CP_pf)
            esn_trainFixed = esn_f

            # -------- Channel metrics (SVD & Capacity) --------
            ranks = []
            conds = []
            gamma = (Pi[jj]/No) / N_t   # equal power per stream
            cap_k = []
            for k in range(N):
                Hk = H_true[k, :, :]
                U, S, Vh = np.linalg.svd(Hk, full_matrices=False)
                s1 = S[0]; s2 = S[1] if len(S) > 1 else 0.0
                # usable rank: s2 significant vs s1 or vs noise
                use2 = (s2**2 >= 1e-2*(s1**2)) or (s2**2 >= 10*(No/Pi[jj]))
                ranks.append(2 if use2 else 1)
                conds.append(s1/max(s2, 1e-12))
                cap_k.append(np.sum(np.log2(1 + gamma * (S**2))))
            cap_acc.append(np.mean(cap_k))
            cond_list.extend(conds)
            rank_list.extend(ranks)

        else:
            # -------- Data OFDM symbol --------
            TxBits = (np.random.rand(N*m, N_t) > 0.5).astype(np.int32)

            X = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), tx])[0])
                    X[ii, tx] = Const[idx]

            # IFFT + CP (matched Pi)
            x_temp = np.zeros((N, N_t), dtype=complex)
            x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
            for tx in range(N_t):
                x_temp[:, tx] = N * np.fft.ifft(X[:, tx])
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)

            # PA
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + noise per Rx
            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
                noise = math.sqrt(len(y_CP[:, nr])*No/2) * (np.random.randn(len(y_CP[:, nr])) + 1j*np.random.randn(len(y_CP[:, nr])))
                y_CP[:, nr] += noise

            # Freq-domain received per Rx
            Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)  # shape (N, Nr)

            # ESN detect (matched)
            ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
            # stack inputs [Re(y1), Im(y1), Re(y2), Im(y2)]
            ESN_input_m[:, 0] = np.r_[y_CP[:, 0].real, np.zeros(Delay_Max_m)]
            ESN_input_m[:, 1] = np.r_[y_CP[:, 0].imag, np.zeros(Delay_Max_m)]
            ESN_input_m[:, 2] = np.r_[y_CP[:, 1].real, np.zeros(Delay_Max_m)]
            ESN_input_m[:, 3] = np.r_[y_CP[:, 1].imag, np.zeros(Delay_Max_m)]
            x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
            x0_m, x1_m = reconstruct_esn_outputs(x_hat_m_tmp, Delay_m, Delay_Min_m, N)
            X_hat_ESN_m = np.zeros((N, N_t), dtype=complex)
            X_hat_ESN_m[:, 0] = (1/N) * np.fft.fft(x0_m) / math.sqrt(Pi[jj])
            X_hat_ESN_m[:, 1] = (1/N) * np.fft.fft(x1_m) / math.sqrt(Pi[jj])

            # ESN detect (fixed-train)
            ESN_input_f = np.zeros((N + Delay_Max_f + CyclicPrefixLen, nInputUnits))
            ESN_input_f[:, 0] = np.r_[y_CP[:, 0].real, np.zeros(Delay_Max_f)]
            ESN_input_f[:, 1] = np.r_[y_CP[:, 0].imag, np.zeros(Delay_Max_f)]
            ESN_input_f[:, 2] = np.r_[y_CP[:, 1].real, np.zeros(Delay_Max_f)]
            ESN_input_f[:, 3] = np.r_[y_CP[:, 1].imag, np.zeros(Delay_Max_f)]
            x_hat_f_tmp = esn_trainFixed.predict(ESN_input_f, nForget_f, continuation=False)
            x0_f, x1_f = reconstruct_esn_outputs(x_hat_f_tmp, Delay_f, Delay_Min_f, N)
            X_hat_ESN_f = np.zeros((N, N_t), dtype=complex)
            X_hat_ESN_f[:, 0] = (1/N) * np.fft.fft(x0_f) / math.sqrt(Pi[jj])
            X_hat_ESN_f[:, 1] = (1/N) * np.fft.fft(x1_f) / math.sqrt(Pi[jj])

            # Baselines per subcarrier
            X_hat_PerfZF = np.zeros((N, N_t), dtype=complex)
            X_hat_LS_ZF = np.zeros((N, N_t), dtype=complex)
            X_hat_MMSE = np.zeros((N, N_t), dtype=complex)

            for k in range(N):
                Yk = Y[k, :].reshape(N_r, 1)
                Hk_true = H_true[k, :, :]
                Hk_LS = H_LS[k, :, :]
                Hk_MMSE = H_MMSE[k, :, :]

                X_hat_PerfZF[k, :] = equalize_zf(Yk, Hk_true, math.sqrt(Pi[jj])).reshape(-1)
                X_hat_LS_ZF[k, :] = equalize_zf(Yk, Hk_LS, math.sqrt(Pi[jj])).reshape(-1)
                X_hat_MMSE[k, :] = equalize_mmse(Yk, Hk_MMSE, math.sqrt(Pi[jj]), noise_over_power=(No/Pi[jj])).reshape(-1)

            # Bit decisions
            def hard_bits_from_syms(Xhat_matrix):
                # Xhat_matrix: (N, N_t)
                RxBits = np.zeros((N*m, N_t), dtype=int)
                for ii in range(N):
                    for tx in range(N_t):
                        sym = Xhat_matrix[ii, tx]
                        idx = int(np.argmin(np.abs(Const - sym)))
                        RxBits[m*ii:m*(ii+1), tx] = bits_to_grayvec(idx, m)
                return RxBits

            RxBits_ESN_m = hard_bits_from_syms(X_hat_ESN_m)
            RxBits_ESN_f = hard_bits_from_syms(X_hat_ESN_f)
            RxBits_LS_ZF = hard_bits_from_syms(X_hat_LS_ZF)
            RxBits_MMSE = hard_bits_from_syms(X_hat_MMSE)
            RxBits_PerfZF = hard_bits_from_syms(X_hat_PerfZF)

            # Accumulate BER (sum over both streams)
            TotalErr_ESN_matched += int(np.sum(TxBits != RxBits_ESN_m))
            TotalErr_ESN_trainFixed += int(np.sum(TxBits != RxBits_ESN_f))
            TotalErr_LS_ZF += int(np.sum(TxBits != RxBits_LS_ZF))
            TotalErr_MMSE += int(np.sum(TxBits != RxBits_MMSE))
            TotalErr_PerfectZF += int(np.sum(TxBits != RxBits_PerfZF))
            TotalBits += NumBitsPerSymbol * N_t

    # Store BER per SNR
    BER_ESN_matched[jj] = TotalErr_ESN_matched / max(TotalBits, 1)
    BER_ESN_trainFixed[jj] = TotalErr_ESN_trainFixed / max(TotalBits, 1)
    BER_LS_ZF[jj] = TotalErr_LS_ZF / max(TotalBits, 1)
    BER_MMSE[jj] = TotalErr_MMSE / max(TotalBits, 1)
    BER_PerfectZF[jj] = TotalErr_PerfectZF / max(TotalBits, 1)

    # Channel metrics
    if len(cap_acc) > 0:
        Capacity_bits_per_sc[jj] = float(np.mean(cap_acc))
        rk = np.array(rank_list)
        Frac_rank_ge2[jj] = float(np.mean(rk >= 2))
        cond = np.array(cond_list)
        Cond_p50[jj] = float(np.percentile(cond, 50))
        Cond_p90[jj] = float(np.percentile(cond, 90))

# -------- Save results --------
results_ber = {
    "EBN0": EbNoDB,
    "BER": {
        "ESN_matched": BER_ESN_matched,
        "ESN_trainFixed": BER_ESN_trainFixed,
        "LS_ZF": BER_LS_ZF,
        "MMSE": BER_MMSE,
        "Perfect_ZF": BER_PerfectZF
    },
    "meta": {
        "TRAIN_EBNO_FIXED_DB": TRAIN_EBNO_FIXED_DB,
        "N": int(N), "m": int(m),
        "IsiDuration": int(IsiDuration), "Nr": int(N_r), "Nt": int(N_t)
    }
}
with open("./BERvsEBNo_ESN_trainSNR_study_mimo2x2.pkl", "wb") as f:
    pickle.dump(results_ber, f)

results_channel = {
    "EBN0": EbNoDB,
    "capacity_bits_per_sc": Capacity_bits_per_sc,
    "frac_rank_ge2": Frac_rank_ge2,
    "cond_number": {
        "p50": Cond_p50,
        "p90": Cond_p90
    },
    "notes": "Rank and conditioning computed from per-subcarrier SVD of true H_k."
}
with open("./channel_metrics_mimo2x2.pkl", "wb") as f:
    pickle.dump(results_channel, f)

# -------- Plots --------
plt.figure()
plt.semilogy(EbNoDB, BER_PerfectZF, 'kx-', label='Perfect ZF')
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE (H_MMSE)')
plt.semilogy(EbNoDB, BER_LS_ZF, 'o-', label='LS ZF')
plt.semilogy(EbNoDB, BER_ESN_matched, 'gd--', label='ESN (matched train SNR)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:', label=f'ESN (train @ {TRAIN_EBNO_FIXED_DB} dB)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title('2x2 MIMO | ESN training SNR study | Nonlinear PA + Block Fading')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(EbNoDB, Capacity_bits_per_sc, 'm.-', label='Avg. capacity per subcarrier')
plt.plot(EbNoDB, Frac_rank_ge2, 'c.-', label='Frac. rank ≥ 2')
plt.grid(True, ls=':'); plt.legend()
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Capacity [bits/s/Hz] / Fraction')
plt.title('2x2 MIMO: Capacity & usable rank')
plt.tight_layout()
plt.show()
