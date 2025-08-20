import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
SISO version with channel understanding + ESN training-SNR study (no LDPC).
- Saves: channel metrics (capacity, |H_k| stats) and BER for:
    * ESN trained at each test SNR (matched)
    * ESN trained at a fixed SNR across all tests (mismatched)
- Leaves pyESN.py and HelpFunc.py unchanged.
"""

# --------------------
# Physical parameters
# --------------------
W = 2*1.024e6          # Available Bandwidth
f_D = 100              # Doppler Frequency
No = 1e-5              # Noise power spectral density
IsiDuration = 8        # Number of multipath components
EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)

# --------------------
# Antenna parameters
# --------------------
N_t = 1
N_r = 1

# --------------------
# Design parameters
# --------------------
N = 512                        # Subcarriers
m = 4                          # QAM order for data (16-QAM)
m_pilot = 4                    # QAM order for pilots
NumOfdmSymbols = 400           # OFDM symbols per SNR point
Ptotal = 10**(EbNoDB/10)*No*N  # Total power per OFDM symbol

# --------------------
# Power Amplifier
# --------------------
p_smooth = 1
ClipLeveldB = 3

# --------------------
# Secondary parameters
# --------------------
T_OFDM_Total = (N + IsiDuration - 1)/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)    # Coherence time (OFDM symbols)
Pi = Ptotal/N                          # Equal power per subcarrier
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
# ESN input variance proxy (per-SNR time-domain transmit power)
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2          # Re/Im of single Rx stream
nOutputUnits = 2         # Re/Im of single Tx stream
nInternalUnits = 100
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
TRAIN_EBNO_FIXED_DB = 12  # ESN "fixed" training SNR for the mismatch study
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# --------------------
# Simulation holders
# --------------------
BER_ESN_matched = np.zeros(len(EbNoDB))
BER_ESN_trainFixed = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

# Channel metrics to save for "understanding the channel"
Capacity_bits_per_sc = np.zeros(len(EbNoDB))   # average over subcarriers
Sval_mean = np.zeros(len(EbNoDB))              # mean |H_k|
Sval_p10 = np.zeros(len(EbNoDB))               # 10th percentile |H_k|
Sval_p90 = np.zeros(len(EbNoDB))               # 90th percentile |H_k|

# MMSE constants
MMSEScaler_allSNR = (No/Pi) # vector across SNRs

# Time-domain correlation matrix (diagonal with tap powers)
R_h = np.diag(IsiMagnitude[:IsiDuration])

def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]

for jj, ebno_db in enumerate(EbNoDB):
    print(f'EbNoDB = {ebno_db}')
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    # ESN scaling per SNR (matched)
    inputScaling_matched = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

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

    # MMSE TD matrix
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler_allSNR[jj]/(N/2)) + np.eye(IsiDuration)

    # placeholders carried from training to data
    esn_matched = None
    esn_trainFixed = None
    Delay_m = None; Delay_Min_m = None; Delay_Max_m = None; nForget_m = None
    Delay_f = None; Delay_Min_f = None; Delay_Max_f = None; nForget_f = None

    # channel stats accumulators (per block; average to get capacity & |H| stats)
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
            H_LS = (Y_p / X_p) / (Pi[jj]**0.5)
            c_LS = np.fft.ifft(H_LS)
            c_LS_trunc = c_LS[:IsiDuration]
            c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS_trunc)
            H_MMSE = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

            # -------- Pilot OFDM (fixed training SNR) --------
            # Same bits and channel, just different Pi for the pilot power
            x_CP_pf = np.r_[x_temp[-CyclicPrefixLen:], x_temp] * (Pi_train_fixed**0.5)
            A_Clip_train = np.sqrt(var_x_train_fixed) * np.float_power(10, ClipLeveldB/20)
            x_CP_pf_NLD = x_CP_pf / ((1 + (np.abs(x_CP_pf)/A_Clip_train)**(2*p_smooth))**(1/(2*p_smooth)))
            y_CP_pf = signal.lfilter(c0, np.array([1]), x_CP_pf_NLD) + noise  # reuse noise draw for fairness

            # -------- Train ESNs (matched & fixed-train) --------
            def train_esn_from_pilot(y_cp, x_cp, inputScaling_used, inputShift_used):
                # Build delay LUT
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
                          input_shift=inputShift_used, input_scaling=inputScaling_used,
                          teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                          feedback_scaling=feedbackScaling)

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

                    x_hat_td = (x_hat_tmp[d0 - dmin : d0 - dmin + N + 1, 0]
                                + 1j * x_hat_tmp[d1 - dmin : d1 - dmin + N + 1, 1])

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
            TxBits = (np.random.rand(N*m, 1) > 0.5).astype(np.int32)

            X = np.zeros((N,), dtype=complex)
            for ii in range(N):
                idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), 0])[0])
                X[ii] = Const[idx]

            # IFFT + CP + power loading (matched SNR for data)
            x_temp = N * np.fft.ifft(X)
            x_CP = np.r_[x_temp[-CyclicPrefixLen:], x_temp] * (Pi[jj]**0.5)

            # PA nonlinearity
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + noise
            y_CP = signal.lfilter(c0, np.array([1]), x_CP_NLD)
            noise = math.sqrt(len(y_CP)*No/2) * (np.random.randn(len(y_CP)) + 1j*np.random.randn(len(y_CP)))
            y_CP = y_CP + noise

            # Freq-domain received
            Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:])

            # -------- ESN Detector predictions (two ESNs) --------
            # Matched-trained ESN
            ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
            ESN_input_m[:, 0] = np.r_[y_CP.real, np.zeros(Delay_Max_m)]
            ESN_input_m[:, 1] = np.r_[y_CP.imag, np.zeros(Delay_Max_m)]
            x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
            x_hat_td_m = (x_hat_m_tmp[Delay_m[0] - Delay_Min_m : Delay_m[0] - Delay_Min_m + N + 1, 0]
                          + 1j * x_hat_m_tmp[Delay_m[1] - Delay_Min_m : Delay_m[1] - Delay_Min_m + N + 1, 1])
            X_hat_ESN_m = (1/N) * np.fft.fft(x_hat_td_m) / math.sqrt(Pi[jj])

            # Fixed-trained ESN
            ESN_input_f = np.zeros((N + Delay_Max_f + CyclicPrefixLen, nInputUnits))
            ESN_input_f[:, 0] = np.r_[y_CP.real, np.zeros(Delay_Max_f)]
            ESN_input_f[:, 1] = np.r_[y_CP.imag, np.zeros(Delay_Max_f)]
            x_hat_f_tmp = esn_trainFixed.predict(ESN_input_f, nForget_f, continuation=False)
            x_hat_td_f = (x_hat_f_tmp[Delay_f[0] - Delay_Min_f : Delay_f[0] - Delay_Min_f + N + 1, 0]
                          + 1j * x_hat_f_tmp[Delay_f[1] - Delay_Min_f : Delay_f[1] - Delay_Min_f + N + 1, 1])
            X_hat_ESN_f = (1/N) * np.fft.fft(x_hat_td_f) / math.sqrt(Pi[jj])

            # -------- Channel-equalized baselines --------
            X_hat_Perfect = (Y / H_true) / math.sqrt(Pi[jj])
            X_hat_LS = (Y / H_LS) / math.sqrt(Pi[jj])
            X_hat_MMSE = (Y / H_MMSE) / math.sqrt(Pi[jj])

            # -------- Bit decisions --------
            def hard_bits_from_syms(Xhat):
                RxBits = np.zeros((N*m, 1), dtype=int)
                for ii in range(N):
                    sym = Xhat[ii]
                    idx = int(np.argmin(np.abs(Const - sym)))
                    RxBits[m*ii:m*(ii+1), 0] = bits_to_grayvec(idx, m)
                return RxBits

            RxBits_ESN_m = hard_bits_from_syms(X_hat_ESN_m)
            RxBits_ESN_f = hard_bits_from_syms(X_hat_ESN_f)
            RxBits_LS = hard_bits_from_syms(X_hat_LS)
            RxBits_MMSE = hard_bits_from_syms(X_hat_MMSE)
            RxBits_Perfect = hard_bits_from_syms(X_hat_Perfect)

            # Accumulate BER
            TotalErr_ESN_matched += int(np.sum(TxBits != RxBits_ESN_m))
            TotalErr_ESN_trainFixed += int(np.sum(TxBits != RxBits_ESN_f))
            TotalErr_LS += int(np.sum(TxBits != RxBits_LS))
            TotalErr_MMSE += int(np.sum(TxBits != RxBits_MMSE))
            TotalErr_Perfect += int(np.sum(TxBits != RxBits_Perfect))
            TotalBits += NumBitsPerSymbol

    # Store BER per SNR
    BER_ESN_matched[jj] = TotalErr_ESN_matched / max(TotalBits, 1)
    BER_ESN_trainFixed[jj] = TotalErr_ESN_trainFixed / max(TotalBits, 1)
    BER_LS[jj] = TotalErr_LS / max(TotalBits, 1)
    BER_MMSE[jj] = TotalErr_MMSE / max(TotalBits, 1)
    BER_Perfect[jj] = TotalErr_Perfect / max(TotalBits, 1)

    # Channel metrics (average over blocks observed at this SNR)
    if len(cap_acc) > 0:
        Capacity_bits_per_sc[jj] = float(np.mean(cap_acc))
        sab = np.concatenate(sabs_acc)
        Sval_mean[jj] = float(np.mean(sab))
        Sval_p10[jj] = float(np.percentile(sab, 10))
        Sval_p90[jj] = float(np.percentile(sab, 90))

# -------- Save results --------
results_ber = {
    "EBN0": EbNoDB,
    "BER": {
        "ESN_matched": BER_ESN_matched,
        "ESN_trainFixed": BER_ESN_trainFixed,
        "LS": BER_LS,
        "MMSE": BER_MMSE,
        "Perfect": BER_Perfect
    },
    "meta": {
        "TRAIN_EBNO_FIXED_DB": TRAIN_EBNO_FIXED_DB,
        "N": int(N), "m": int(m),
        "IsiDuration": int(IsiDuration)
    }
}
with open("./BERvsEBNo_ESN_trainSNR_study_siso.pkl", "wb") as f:
    pickle.dump(results_ber, f)

results_channel = {
    "EBN0": EbNoDB,
    "capacity_bits_per_sc": Capacity_bits_per_sc,  # averaged over subcarriers and blocks
    "absHk_stats": {
        "mean": Sval_mean,
        "p10": Sval_p10,
        "p90": Sval_p90
    },
    "notes": "SISO: channel rank is 1 by definition; we log |H_k| stats and capacity instead."
}
with open("./channel_metrics_siso.pkl", "wb") as f:
    pickle.dump(results_channel, f)

# -------- Plot BER (with training-SNR comparison) --------
plt.figure()
plt.semilogy(EbNoDB, BER_Perfect, 'kx-', label='Perfect CSI')
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE')
plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS')
plt.semilogy(EbNoDB, BER_ESN_matched, 'gd--', label='ESN (matched train SNR)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:', label=f'ESN (train @ {TRAIN_EBNO_FIXED_DB} dB)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title('SISO | ESN training SNR study | Nonlinear PA + Block Fading')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.show()

# -------- Plot capacity vs SNR --------
plt.figure()
plt.plot(EbNoDB, Capacity_bits_per_sc, 'm.-', label='Avg. capacity per subcarrier')
plt.grid(True, ls=':'); plt.legend()
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Capacity [bits/s/Hz per subcarrier]')
plt.title('SISO capacity (from |H_k|, same channel ensemble)')
plt.tight_layout()
plt.show()
