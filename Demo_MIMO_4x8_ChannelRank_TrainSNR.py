
# Demo_MIMO_4x8_ChannelRank_TrainSNR.py

import numpy as np
import math
from scipy import signal, interpolate
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle
# --- add once near the top of the file ---
import os

from helper_mimo_esn_generic import trainMIMOESN_generic
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
        im_seq = x_hat_tmp[Delay[im_col]-Delay_Min: Delay[im_col]-Delay_Min + N + 1, im_col]
        xs.append(re_seq + 1j*im_seq)
    return xs

# --------------------
# Physical parameters
# --------------------
W = 2*1.024e6
f_D = 100
No = 1e-5
IsiDuration = 8
EbNoDB = np.arange(0, 31, 3).astype(np.int32)

# --------------------
# Antennas (4x8)
# --------------------
N_t = 4
N_r = 8

# --------------------
# Design parameters
# --------------------
N = 512
m = 4
m_pilot = 4
NumOfdmSymbols = 400
Ptotal = 10**(EbNoDB/10)*No*N

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

# --------------------
# ESN parameters
# --------------------
var_x = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2*N_r
nOutputUnits = 2*N_t
nInternalUnits = 600
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

teacherShift = np.zeros(nOutputUnits)
feedbackScaling = feedbackScaler*np.ones(nOutputUnits)

Min_Delay = 0
Max_Delay = int(math.ceil(IsiDuration/2) + 2)
DelayFlag = 0  # shared delay, fast

TRAIN_EBNO_FIXED_DB = 12
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# --------------------
# Holders
# --------------------
BER_ESN_matched = np.zeros(len(EbNoDB))
BER_ESN_trainFixed = np.zeros(len(EbNoDB))
BER_PerfectZF = np.zeros(len(EbNoDB))
BER_LS_ZF = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

Capacity_bits_per_sc = np.zeros(len(EbNoDB))
Frac_rank_ge_full = np.zeros(len(EbNoDB))  # rank ≥ 4
Cond_p50 = np.zeros(len(EbNoDB))
Cond_p90 = np.zeros(len(EbNoDB))

MMSEScaler_allSNR = (No/Pi)

R_h = np.diag(IsiMagnitude[:IsiDuration])

for jj, ebno_db in enumerate(EbNoDB):
    print(f'EbNoDB = {ebno_db}')
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    inputScaling_matched = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    inputScaling_trainFixed = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFixed = (inputOffset/inputScaler) * np.ones(nInputUnits)

    TotalErr_ESN_matched = 0
    TotalErr_ESN_trainFixed = 0
    TotalErr_LS_ZF = 0
    TotalErr_MMSE = 0
    TotalErr_PerfectZF = 0
    TotalBits = 0

    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler_allSNR[jj]/(N/2)) + np.eye(IsiDuration)

    esn_matched = None
    esn_trainFixed = None
    Delay_m = None; Delay_Min_m = None; Delay_Max_m = None; nForget_m = None
    Delay_f = None; Delay_Min_f = None; Delay_Max_f = None; nForget_f = None

    cap_acc = []
    cond_list = []
    rank_list = []

    for kk in range(1, NumOfdmSymbols+1):
        if (np.remainder(kk, L) == 1):
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

            for nr in range(N_r):
                for tx in range(N_t):
                    sc_idx = np.arange(tx, N, N_t)
                    denom = (X_LS[sc_idx, tx] * (Pi[jj]**0.5) + 1e-12)
                    Hls_sc = Y_LS[sc_idx, nr] / denom
                    tmpf = interpolate.interp1d(
                        sc_idx, Hls_sc, kind='linear',
                        bounds_error=False, fill_value='extrapolate'
                    )
                    Hls_full = tmpf(np.arange(N))

                    c_LS = np.fft.ifft(Hls_full)
                    c_LS_trunc = c_LS[:IsiDuration]
                    c_MMSE = np.linalg.solve(np.dot(np.linalg.inv(R_h), MMSEScaler_allSNR[jj]/(N/2)) + np.eye(IsiDuration), c_LS_trunc)
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
                        spectral_radius=spectralRadius, sparsity=0.1,
                        input_shift=inputShift_trainFixed, input_scaling=inputScaling_trainFixed,
                        teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                        feedback_scaling=feedbackScaling)
            ESN_input, ESN_output, esn_f, Delay_f, Delay_Idx_f, Delay_Min_f, Delay_Max_f, nForget_f, _ = \
                trainMIMOESN_generic(esn_f, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen,
                                     N, N_t, N_r, IsiDuration, y_CP_pf, x_CP_pf)
            esn_trainFixed = esn_f

            ranks = []
            conds = []
            gamma = (Pi[jj]/No) / N_t
            cap_k = []
            for k in range(N):
                Hk = H_true[k, :, :]
                U, S, Vh = np.linalg.svd(Hk, full_matrices=False)
                s1 = S[0]; smin = S[-1] if len(S)>0 else 0.0
                thr = max(1e-2*(s1**2), 10*(No/Pi[jj]))
                ranks.append(np.sum(S**2 >= thr))
                conds.append(s1/max(smin, 1e-12))
                cap_k.append(np.sum(np.log2(1 + gamma * (S**2))))
            cap_acc.append(np.mean(cap_k))
            cond_list.extend(conds)
            rank_list.extend(ranks)

        else:
            TxBits = (np.random.rand(N*m, N_t) > 0.5).astype(np.int32)
            X = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), tx])[0])
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

            ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
            for rx in range(N_r):
                ESN_input_m[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(Delay_Max_m)]
                ESN_input_m[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(Delay_Max_m)]
            x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
            x_time_list_m = reconstruct_esn_outputs_generic(x_hat_m_tmp, Delay_m, Delay_Min_m, N, N_t)
            X_hat_ESN_m = np.zeros((N, N_t), dtype=complex)
            for tx in range(N_t):
                X_hat_ESN_m[:, tx] = (1/N) * np.fft.fft(x_time_list_m[tx]) / math.sqrt(Pi[jj])

            ESN_input_f = np.zeros((N + Delay_Max_f + CyclicPrefixLen, nInputUnits))
            for rx in range(N_r):
                ESN_input_f[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(Delay_Max_f)]
                ESN_input_f[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(Delay_Max_f)]
            x_hat_f_tmp = esn_trainFixed.predict(ESN_input_f, nForget_f, continuation=False)
            x_time_list_f = reconstruct_esn_outputs_generic(x_hat_f_tmp, Delay_f, Delay_Min_f, N, N_t)
            X_hat_ESN_f = np.zeros((N, N_t), dtype=complex)
            for tx in range(N_t):
                X_hat_ESN_f[:, tx] = (1/N) * np.fft.fft(x_time_list_f[tx]) / math.sqrt(Pi[jj])

            X_hat_PerfZF = np.zeros((N, N_t), dtype=complex)
            X_hat_LS_ZF  = np.zeros((N, N_t), dtype=complex)
            X_hat_MMSE   = np.zeros((N, N_t), dtype=complex)

            for k in range(N):
                Yk = Y[k, :].reshape(N_r, 1)
                Hk_true  = H_true[k, :, :]
                Hk_LS    = H_LS[k, :, :]
                Hk_MMSEf = H_MMSE[k, :, :]

                X_hat_PerfZF[k, :] = equalize_zf(Yk, Hk_true,  math.sqrt(Pi[jj])).reshape(-1)
                X_hat_LS_ZF[k, :]  = equalize_zf(Yk, Hk_LS,    math.sqrt(Pi[jj])).reshape(-1)
                X_hat_MMSE[k, :]   = equalize_mmse(Yk, Hk_MMSEf, math.sqrt(Pi[jj]), noise_over_power=(No/Pi[jj])).reshape(-1)

            def hard_bits_from_syms(Xhat_matrix):
                RxBits = np.zeros((N*m, N_t), dtype=int)
                for ii in range(N):
                    for tx in range(N_t):
                        sym = Xhat_matrix[ii, tx]
                        idx = int(np.argmin(np.abs(Const - sym)))
                        RxBits[m*ii:m*(ii+1), tx] = bits_to_grayvec(idx, m)
                return RxBits

            RxBits_ESN_m   = hard_bits_from_syms(X_hat_ESN_m)
            RxBits_ESN_f   = hard_bits_from_syms(X_hat_ESN_f)
            RxBits_LS_ZF   = hard_bits_from_syms(X_hat_LS_ZF)
            RxBits_MMSE    = hard_bits_from_syms(X_hat_MMSE)
            RxBits_PerfZF  = hard_bits_from_syms(X_hat_PerfZF)

            TotalErr_ESN_matched    += int(np.sum(TxBits != RxBits_ESN_m))
            TotalErr_ESN_trainFixed += int(np.sum(TxBits != RxBits_ESN_f))
            TotalErr_LS_ZF          += int(np.sum(TxBits != RxBits_LS_ZF))
            TotalErr_MMSE           += int(np.sum(TxBits != RxBits_MMSE))
            TotalErr_PerfectZF      += int(np.sum(TxBits != RxBits_PerfZF))
            TotalBits               += NumBitsPerSymbol * N_t

    BER_ESN_matched[jj]    = TotalErr_ESN_matched / max(TotalBits, 1)
    BER_ESN_trainFixed[jj] = TotalErr_ESN_trainFixed / max(TotalBits, 1)
    BER_LS_ZF[jj]          = TotalErr_LS_ZF / max(TotalBits, 1)
    BER_MMSE[jj]           = TotalErr_MMSE / max(TotalBits, 1)
    BER_PerfectZF[jj]      = TotalErr_PerfectZF / max(TotalBits, 1)

    if len(cap_acc) > 0:
        Capacity_bits_per_sc[jj] = float(np.mean(cap_acc))
        rk = np.array(rank_list)
        Frac_rank_ge_full[jj] = float(np.mean(rk >= min(N_t, N_r)))
        cond = np.array(cond_list)
        Cond_p50[jj] = float(np.percentile(cond, 50))
        Cond_p90[jj] = float(np.percentile(cond, 90))

# -------- Save channel metrics --------
results_channel = {
    "EBN0": EbNoDB.tolist(),
    "capacity_bits_per_sc": Capacity_bits_per_sc.tolist(),
    "frac_rank_ge_full": Frac_rank_ge_full.tolist(),
    "cond_number": {
        "p50": Cond_p50.tolist(),
        "p90": Cond_p90.tolist()
    },
    "notes": "Rank and conditioning computed from per-subcarrier SVD of true H_k."
}

outdir = "./results_4x8"   # choose any writable folder you like
os.makedirs(outdir, exist_ok=True)

with open(f"{outdir}/channel_metrics_mimo4x8.pkl", "wb") as f:
    pickle.dump(results_channel, f)

# -------- Plots: SAVE + SHOW --------
plt.figure()
plt.semilogy(EbNoDB, BER_PerfectZF, 'kx-', label='Perfect ZF')
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE (H_MMSE)')
plt.semilogy(EbNoDB, BER_LS_ZF, 'o-', label='LS ZF')
plt.semilogy(EbNoDB, BER_ESN_matched, 'gd--', label='ESN (matched train SNR)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:', label=f'ESN (train @ {TRAIN_EBNO_FIXED_DB} dB)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title('4x8 MIMO | ESN training SNR study | Nonlinear PA + Block Fading')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.savefig(f"{outdir}/BER_4x8.png", dpi=150)
plt.show()  # show the BER figure

plt.figure()
plt.plot(EbNoDB, Capacity_bits_per_sc, 'm.-', label='Avg. capacity per subcarrier')
plt.plot(EbNoDB, Frac_rank_ge_full, 'c.-', label='Frac. rank ≥ 4')
plt.grid(True, ls=':'); plt.legend()
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Capacity [bits/s/Hz] / Fraction')
plt.title('4x8 MIMO: Capacity & usable rank (≥4)')
plt.tight_layout()
plt.savefig(f"{outdir}/CapacityRank_4x8.png", dpi=150)
plt.show()  # show the capacity/rank figure

print(f"Saved: {outdir}/BER_4x8.png and {outdir}/CapacityRank_4x8.png")
