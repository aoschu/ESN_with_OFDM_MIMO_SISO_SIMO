
# ------------------------------------------------------------
# 4x8 MIMO-OFDM with nonlinear PA + block fading
# Channel: 3GPP TR 38.901 CDL (via Sionna), drawn once per coherence block.
# Receiver: ESN (time-domain), plus LS-ZF and MMSE baselines (frequency-domain).
# Plots pre-LDPC (uncoded) BER vs Eb/N0.
#
# Requirements:
#   pip install "tensorflow<2.16" sionna scipy matplotlib pyESN
#   (and ensure helper_mimo_esn_generic.py is in the same folder)
# ------------------------------------------------------------

import os, math, numpy as np, matplotlib.pyplot as plt
from scipy import signal, interpolate
from pyESN import ESN

# Your existing ESN helper
from helper_mimo_esn_generic import trainMIMOESN_generic

# ===== Try Sionna imports =====
try:
    import tensorflow as tf
    from sionna.channel.tr38901 import CDL, AntennaArray
    from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel
    SIONNA_OK = True
except Exception as e:
    SIONNA_OK = False
    SIONNA_ERR = e

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
        im_seq = x_hat_tmp[Delay[im_col]-Delay_Min: Delay[im_col]-Delay_Min + N + 1, im_col]
        xs.append(re_seq + 1j*im_seq)
    return xs

def hard_bits_from_syms(Xhat_matrix, Const, m):
    N, N_t = Xhat_matrix.shape
    RxBits = np.zeros((N*m, N_t), dtype=int)
    for ii in range(N):
        for tx in range(N_t):
            sym = Xhat_matrix[ii, tx]
            idx = int(np.argmin(np.abs(Const - sym)))
            RxBits[m*ii:m*(ii+1), tx] = bits_to_grayvec(idx, m)
    return RxBits

# --------------------
# Sionna CDL helper
# --------------------
def make_cdl_sampler(N_t, N_r, W, carrier_frequency=3.5e9,
                     model='B', delay_spread=300e-9, direction='uplink', speed=0.0):
    """
    Returns: (sample_cdl_taps, L_taps)
      - sample_cdl_taps(): -> list c[nr][nt] of complex taps @ rate W
      - L_taps: number of taps used (CP should be >= L_taps-1)
    Uses the same AntennaArray API shown in your example.
    """
    if not SIONNA_OK:
        raise RuntimeError(
            f"Sionna not available: {SIONNA_ERR}\n"
            "Install with: pip install 'tensorflow<2.16' sionna"
        )

    # Polarization 'dual' with 'cross' uses two ports per element,
    # so we use num_cols = N_ant/2 to reach N_ant ports overall.
    def _make_array(n_ant):
        return AntennaArray(num_rows=1,
                            num_cols=int(max(1, n_ant//2)),
                            polarization="dual",
                            polarization_type="cross",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)

    ut_array = _make_array(N_t)  # TX side (uplink UT)
    bs_array = _make_array(N_r)  # RX side (BS)

    cdl = CDL(model, delay_spread, carrier_frequency,
              ut_array, bs_array, direction, min_speed=speed)

    # Recommended truncation from Sionna
    l_min, l_max = time_lag_discrete_time_channel(W)
    L_taps = l_max - l_min + 1

    @tf.function(jit_compile=False)
    def _draw_cir():
        # One snapshot; we use time samples = 1 + L_taps -1 to be safe
        a, tau = cdl(batch_size=1, num_time_steps=1 + L_taps - 1, sampling_frequency=W)
        # Discrete-time taps
        h_time = cir_to_time_channel(W, a, tau, l_min=l_min, l_max=l_max, normalize=True)  # [B, T, Nr, Nt, L]
        return h_time

    def sample_cdl_taps():
        h = _draw_cir().numpy()        # [1, T, Nr, Nt, L]
        h = np.squeeze(h, axis=0)      # [T, Nr, Nt, L]
        # Use first time sample (block fading)
        h0 = h[0, ...]                 # [Nr, Nt, L]
        # Convert to list-of-lists to reuse your lfilter code
        c = [[None for _ in range(N_t)] for __ in range(N_r)]
        for nr in range(N_r):
            for nt in range(N_t):
                taps = h0[nr, nt, :]
                c[nr][nt] = np.asarray(taps, dtype=np.complex128)
        return c

    return sample_cdl_taps, L_taps

# --------------------
# Physical / system parameters
# --------------------
W = 2*1.024e6
No = 1e-5
EbNoDB = np.arange(0, 31, 3).astype(np.int32)

# Antennas
N_t = 4
N_r = 8

# OFDM
N = 512
m = 4
m_pilot = 4
NumOfdmSymbols = 200  # a bit smaller for runtime
Const = np.array(unit_qam_constellation(m)).astype(complex)
ConstPilot = np.array(unit_qam_constellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))

# PA
p_smooth = 1
ClipLeveldB = 3

# Build CDL sampler now to size CP properly once
get_cdl, L_taps = make_cdl_sampler(N_t=N_t, N_r=N_r, W=W,
                                   carrier_frequency=2.6e9,
                                   model='B', delay_spread=300e-9,
                                   direction='uplink', speed=10.0)

# CP must be >= channel length - 1
CyclicPrefixLen = max(6, L_taps - 1)  # fallback to >=6 as in your example grid

# Coherence (block fading) like before
T_OFDM_Total = (N + CyclicPrefixLen)/W
f_D = 100
tau_c = 0.5/f_D
L = max(1, math.floor(tau_c/T_OFDM_Total))

# ESN params (same style as your original)
var_x_all = np.float_power(10, (EbNoDB/10))*No*N
nInputUnits = 2*N_r
nOutputUnits = 2*N_t
nInternalUnits = 600
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

TRAIN_EBNO_FIXED_DB = 12
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# Holders
BER_ESN_matched    = np.zeros(len(EbNoDB))
BER_ESN_trainFixed = np.zeros(len(EbNoDB))
BER_PerfectZF      = np.zeros(len(EbNoDB))
BER_LS_ZF          = np.zeros(len(EbNoDB))
BER_MMSE           = np.zeros(len(EbNoDB))

# --------------------
# Simulation
# --------------------
for jj, ebno_db in enumerate(EbNoDB):
    print(f"EbNoDB = {ebno_db}")
    var_x = var_x_all[jj]
    Ptotal = 10**(ebno_db/10)*No*N
    Pi = Ptotal/N
    A_Clip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB/20)

    # ESN scalings for this SNR and for the fixed train SNR
    inputScaling_matched = (inputScaler/(var_x**0.5)) * np.ones(nInputUnits)
    inputShift_matched   = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling       = teacherScalingBase * np.ones(nOutputUnits)

    inputScaling_trainFixed = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFixed   = (inputOffset/inputScaler) * np.ones(nInputUnits)

    TotalErr_ESN_matched = 0
    TotalErr_ESN_trainFixed = 0
    TotalErr_LS_ZF = 0
    TotalErr_MMSE = 0
    TotalErr_PerfectZF = 0
    TotalBits = 0

    # ESN objects (recreated each SNR as in your original flow)
    esn_matched = None
    esn_trainFixed = None
    Delay_m = Delay_f = None
    Delay_Min_m = Delay_Min_f = None
    Delay_Max_m = Delay_Max_f = None
    nForget_m = nForget_f = None

    # Precompute a simple diagonal prior for MMSE time-domain tap denoising
    R_h = np.eye(L_taps, dtype=np.complex128)
    MMSEScaler_allSNR = (No/(Pi))  # scalar per SNR

    for kk in range(1, NumOfdmSymbols+1):
        if (np.remainder(kk, L) == 1):
            # --- Draw CDL channel taps (block fading) ---
            c = get_cdl()  # c[nr][nt] with length L_taps

            # True frequency response H_true for perfect-ZF reference
            H_true = np.zeros((N, N_r, N_t), dtype=complex)
            for nr in range(N_r):
                for nt in range(N_t):
                    taps = c[nr][nt]
                    h_pad = np.r_[taps, np.zeros(max(0, N - len(taps)))]
                    H_true[:, nr, nt] = np.fft.fft(h_pad)

            # --- Pilot OFDM to build LS/MMSE channel estimates ---
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
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi**0.5)
                x_temp_ls = N * np.fft.ifft(X_LS[:, tx])
                x_LS_CP[:, tx] = np.r_[x_temp_ls[-CyclicPrefixLen:], x_temp_ls] * (Pi**0.5)

            # PA nonlinearity
            x_CP_NLD    = x_CP    / ((1 + (np.abs(x_CP   )/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            x_LS_CP_NLD = x_LS_CP / ((1 + (np.abs(x_LS_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Pass pilots through CDL channel + AWGN
            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            y_LS_CP = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr]    += signal.lfilter(c[nr][tx], np.array([1.0]), x_CP_NLD[:, tx])
                    y_LS_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1.0]), x_LS_CP_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP[:, nr]    += noise
                y_LS_CP[:, nr] += noise

            Y_LS = (1/N) * np.fft.fft(y_LS_CP[CyclicPrefixLen:, :], axis=0)

            H_LS = np.zeros_like(H_true)
            H_MMSE = np.zeros_like(H_true)
            for nr in range(N_r):
                for tx in range(N_t):
                    sc_idx = np.arange(tx, N, N_t)
                    denom = (X_LS[sc_idx, tx] * (Pi**0.5) + 1e-12)
                    Hls_sc = Y_LS[sc_idx, nr] / denom
                    tmpf = interpolate.interp1d(sc_idx, Hls_sc, kind='linear', bounds_error=False, fill_value='extrapolate')
                    Hls_full = tmpf(np.arange(N))

                    # Time-domain MMSE denoising of LS taps
                    c_LS = np.fft.ifft(Hls_full)
                    c_LS_trunc = c_LS[:L_taps]
                    c_MMSE = np.linalg.solve(np.dot(np.linalg.inv(R_h), (MMSEScaler_allSNR/(N/2))) + np.eye(L_taps), c_LS_trunc)
                    Hmmse_full = np.fft.fft(np.r_[c_MMSE, np.zeros(N-L_taps)])

                    H_LS[:, nr, tx] = Hls_full
                    H_MMSE[:, nr, tx] = Hmmse_full

            # ------- Train ESN (matched SNR) on pilot -------
            esn_m = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=0.9, sparsity=0.1,
                        input_shift=(inputOffset/inputScaler) * np.ones(nInputUnits),
                        input_scaling=(inputScaler/(var_x**0.5)) * np.ones(nInputUnits),
                        teacher_scaling=teacherScalingBase * np.ones(nOutputUnits),
                        teacher_shift=np.zeros(nOutputUnits),
                        feedback_scaling=feedbackScaler*np.ones(nOutputUnits))
            ESN_input, ESN_output, esn_m, Delay_m, Delay_Idx_m, Delay_Min_m, Delay_Max_m, nForget_m, _ = \
                trainMIMOESN_generic(esn_m, 0, 0, int(math.ceil(L_taps/2)+2), CyclicPrefixLen,
                                     N, N_t, N_r, L_taps, y_CP, x_CP)
            esn_matched = esn_m

            # ------- Train ESN (fixed SNR = 12 dB) on pilot -------
            x_CP_pf = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_CP_pf[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi_train_fixed**0.5)
            A_Clip_train = np.sqrt(var_x_train_fixed) * np.float_power(10, ClipLeveldB/20)
            x_CP_pf_NLD = x_CP_pf / ((1 + (np.abs(x_CP_pf)/A_Clip_train)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP_pf = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP_pf[:, nr] += signal.lfilter(c[nr][tx], np.array([1.0]), x_CP_pf_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP_pf[:, nr] += noise

            esn_f = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=0.9, sparsity=0.1,
                        input_shift=(inputOffset/inputScaler) * np.ones(nInputUnits),
                        input_scaling=(inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits),
                        teacher_scaling=teacherScalingBase * np.ones(nOutputUnits),
                        teacher_shift=np.zeros(nOutputUnits),
                        feedback_scaling=feedbackScaler*np.ones(nOutputUnits))
            ESN_input, ESN_output, esn_f, Delay_f, Delay_Idx_f, Delay_Min_f, Delay_Max_f, nForget_f, _ = \
                trainMIMOESN_generic(esn_f, 0, 0, int(math.ceil(L_taps/2)+2), CyclicPrefixLen,
                                     N, N_t, N_r, L_taps, y_CP_pf, x_CP_pf)
            esn_trainFixed = esn_f

        else:
            # ---------- DATA OFDM ----------
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
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi**0.5)

            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Pass through CDL channel
            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1.0]), x_CP_NLD[:, tx])
                noise = math.sqrt(len(y_CP[:, nr])*No/2) * (np.random.randn(len(y_CP[:, nr])) + 1j*np.random.randn(len(y_CP[:, nr])))
                y_CP[:, nr] += noise

            # FFT to subcarrier domain
            Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)

            # --- ESN inference (matched SNR) ---
            ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
            for rx in range(N_r):
                ESN_input_m[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(Delay_Max_m)]
                ESN_input_m[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(Delay_Max_m)]
            x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
            x_time_list_m = reconstruct_esn_outputs_generic(x_hat_m_tmp, Delay_m, Delay_Min_m, N, N_t)
            X_hat_ESN_m = np.zeros((N, N_t), dtype=complex)
            for tx in range(N_t):
                X_hat_ESN_m[:, tx] = (1/N) * np.fft.fft(x_time_list_m[tx]) / math.sqrt(Pi)

            # --- ESN inference (fixed-train SNR) ---
            ESN_input_f = np.zeros((N + Delay_Max_f + CyclicPrefixLen, nInputUnits))
            for rx in range(N_r):
                ESN_input_f[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(Delay_Max_f)]
                ESN_input_f[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(Delay_Max_f)]
            x_hat_f_tmp = esn_trainFixed.predict(ESN_input_f, nForget_f, continuation=False)
            x_time_list_f = reconstruct_esn_outputs_generic(x_hat_f_tmp, Delay_f, Delay_Min_f, N, N_t)
            X_hat_ESN_f = np.zeros((N, N_t), dtype=complex)
            for tx in range(N_t):
                X_hat_ESN_f[:, tx] = (1/N) * np.fft.fft(x_time_list_f[tx]) / math.sqrt(Pi)

            # --- Linear equalizers ---
            X_hat_PerfZF = np.zeros((N, N_t), dtype=complex)
            X_hat_LS_ZF  = np.zeros((N, N_t), dtype=complex)
            X_hat_MMSE   = np.zeros((N, N_t), dtype=complex)

            for k in range(N):
                Yk = Y[k, :].reshape(N_r, 1)
                Hk_true  = H_true[k, :, :]
                Hk_LS    = H_LS[k, :, :]
                Hk_MMSEf = H_MMSE[k, :, :]

                X_hat_PerfZF[k, :] = equalize_zf(Yk, Hk_true,  math.sqrt(Pi)).reshape(-1)
                X_hat_LS_ZF[k, :]  = equalize_zf(Yk, Hk_LS,    math.sqrt(Pi)).reshape(-1)
                X_hat_MMSE[k, :]   = equalize_mmse(Yk, Hk_MMSEf, math.sqrt(Pi), noise_over_power=(No/Pi)).reshape(-1)

            # --- Hard demap + bit errors ---
            RxBits_ESN_m   = hard_bits_from_syms(X_hat_ESN_m, Const, m)
            RxBits_ESN_f   = hard_bits_from_syms(X_hat_ESN_f, Const, m)
            RxBits_LS_ZF   = hard_bits_from_syms(X_hat_LS_ZF, Const, m)
            RxBits_MMSE    = hard_bits_from_syms(X_hat_MMSE,  Const, m)
            RxBits_PerfZF  = hard_bits_from_syms(X_hat_PerfZF, Const, m)

            TotalErr_ESN_matched    += int(np.sum(TxBits != RxBits_ESN_m))
            TotalErr_ESN_trainFixed += int(np.sum(TxBits != RxBits_ESN_f))
            TotalErr_LS_ZF          += int(np.sum(TxBits != RxBits_LS_ZF))
            TotalErr_MMSE           += int(np.sum(TxBits != RxBits_MMSE))
            TotalErr_PerfectZF      += int(np.sum(TxBits != RxBits_PerfZF))
            TotalBits               += (m*N) * N_t

    # BERs per SNR
    BER_ESN_matched[jj]    = TotalErr_ESN_matched / max(TotalBits, 1)
    BER_ESN_trainFixed[jj] = TotalErr_ESN_trainFixed / max(TotalBits, 1)
    BER_LS_ZF[jj]          = TotalErr_LS_ZF / max(TotalBits, 1)
    BER_MMSE[jj]           = TotalErr_MMSE / max(TotalBits, 1)
    BER_PerfectZF[jj]      = TotalErr_PerfectZF / max(TotalBits, 1)

# --------------------
# Plot
# --------------------
outdir = "./results_4x8"
os.makedirs(outdir, exist_ok=True)

plt.figure(figsize=(9,6))
plt.semilogy(EbNoDB, BER_PerfectZF, 'kx-', label='Perfect ZF (CDL)')
plt.semilogy(EbNoDB, BER_MMSE,    'rs-.', label='MMSE (CDL, H_MMSE)')
plt.semilogy(EbNoDB, BER_LS_ZF,   'o-',   label='LS ZF (CDL)')
plt.semilogy(EbNoDB, BER_ESN_matched, 'gd--', label='ESN (matched SNR, CDL)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:', label=f'ESN (train @ {TRAIN_EBNO_FIXED_DB} dB, CDL)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title('4x8 MIMO-OFDM | CDL Channel (Sionna) | Uncoded BER')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
png = f"{outdir}/BER_4x8_CDL_ESN.png"
plt.savefig(png, dpi=150)
plt.show()

print("Saved:", png)

