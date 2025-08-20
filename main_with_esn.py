import numpy as np
import math
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.channel.tr38901 import CDL, PanelArray
from sionna.channel import cir_to_time_channel
from help_func import unit_qam_constellation, compute_channel_corr_matrix, train_mimo_esn
from esn import ESN

# Function to compute BER
def compute_ber(x_hat, x_true, constellation, m):
    """Compute BER by mapping complex symbols to bits."""
    symbols_pred = np.argmin(np.abs(x_hat[:, :, np.newaxis] - constellation), axis=2)
    symbols_true = np.argmin(np.abs(x_true[:, :, np.newaxis] - constellation), axis=2)
    bits_pred = np.zeros((x_hat.shape[0], x_hat.shape[1], m), dtype=np.int32)
    bits_true = np.zeros((x_true.shape[0], x_true.shape[1], m), dtype=np.int32)
    for i in range(m):
        bits_pred[:, :, i] = (symbols_pred >> i) & 1
        bits_true[:, :, i] = (symbols_true >> i) & 1
    return np.mean(bits_pred != bits_true)

# Sionna Channel Setup
N_r, N_t = 2, 2
carrier_frequency = 3.5e9
bs_array = PanelArray(
    num_rows_per_panel=1, num_cols_per_panel=1, polarization='dual', polarization_type='VH',
    antenna_pattern='38.901', carrier_frequency=carrier_frequency)
ut_array = PanelArray(
    num_rows_per_panel=1, num_cols_per_panel=1, polarization='dual', polarization_type='VH',
    antenna_pattern='omni', carrier_frequency=carrier_frequency)
delay_spread = 300e-9
cdl = CDL(model="C", delay_spread=delay_spread, carrier_frequency=carrier_frequency,
          ut_array=ut_array, bs_array=bs_array, direction="downlink", min_speed=0., max_speed=0., dtype=tf.complex64)

# Precompute IsiMagnitude
cluster_delays_ns = np.array([0, 76.6135, 80.9935, 85.0085, 79.424, 232.359, 235.352, 239.44, 240.316, 289.6275, 299.7745, 340.764, 448.4025, 477.5295, 792.196, 989.3325, 1554.4985, 1679.1095, 2003.923, 2046.8105, 2301.8725])
cluster_powers_db = np.array([-4.4215, -1.25, -3.4684, -5.2294, -2.5215, 0, -2.2185, -3.9794, -7.4215, -7.1215, -10.7215, -11.1215, -5.1215, -6.8215, -8.7215, -13.2215, -13.9215, -13.9215, -15.8215, -17.1215, -16.0215])
cluster_powers_lin = 10 ** (cluster_powers_db / 10)
cluster_powers_lin /= np.sum(cluster_powers_lin)
tau = cluster_delays_ns * 1e-9
W = 2 * 1.024e6
IsiDuration = 8
l_min = 0
l_max = IsiDuration - 1
IsiMagnitude = np.zeros(IsiDuration)
for k in range(IsiDuration):
    x = tau * W - k
    IsiMagnitude[k] = np.sum(cluster_powers_lin * np.sinc(x) ** 2)
IsiMagnitude /= np.sum(IsiMagnitude)
print("Computed IsiMagnitude:", IsiMagnitude)

# Physical/Design Parameters
f_D = 100
No = 0.00001
EbNoDB = np.arange(0, 30, 2).astype(np.int32)
N = 64
Subcarrier_Spacing = W / N
m = 4
m_pilot = 4
NumOfdmSymbols = 100
Ptotal = 10 ** (EbNoDB / 10) * No * N
p_smooth = 5
ClipLeveldB = 5
T_OFDM = N / W
T_OFDM_Total = (N + IsiDuration - 1) / W
T_s = 1 / W
tau_c = 0.5 / f_D
L = math.floor(tau_c / T_OFDM_Total)
Pi = Ptotal / N
NumBitsPerSymbol = m * N
Const = unit_qam_constellation(m)
ConstPilot = unit_qam_constellation(m_pilot)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1
R_h = np.diag(IsiMagnitude)

# ESN Parameters
var_x = np.power(10, (EbNoDB / 10)) * No * N
nInputUnits = N_t * 2
nOutputUnits = N_t * 2
nInternalUnits = 500
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.0000005 * np.ones(N_t * 2)
spectralRadius = 0.9
teacherShift = np.zeros(N_t * 2)
feedbackScaling = feedbackScaler * np.ones(N_t * 2)
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration / 2) + 2
DelayFlag = 0

# Initialize Arrays
ESN_train_input = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_train_teacher = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_test_input = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_test_output = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))
NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))
c = [[None] * N_t for _ in range(N_r)]
Ci = [[None] * N_t for _ in range(N_r)]
Ci_LS = [[None] * N_t for _ in range(N_r)]
Ci_MMSE = [[None] * N_t for _ in range(N_r)]
Ci_LS_Pilots = [[None] * N_t for _ in range(N_r)]
MMSEScaler = No / Pi

# Main Loop
for jj in range(len(EbNoDB)):
    print(f"\n=== Processing SNR: EbNoDB = {EbNoDB[jj]} dB ===")
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB / 20)
    inputScaling = inputScaler / (var_x[jj] ** 0.5) * np.ones(N_t * 2)
    inputShift = inputOffset / inputScaler * np.ones(N_t * 2)
    TotalBerNum_ESN = 0
    TotalBerNum_LS = 0
    TotalBerNum_MMSE = 0
    TotalBerNum_Perfect = 0
    TotalBerDen = 0
    MMSE_bold_TD = np.linalg.inv(R_h + MMSEScaler[jj] / (N * (N / 2)) * np.eye(IsiDuration))

    for kk in range(1, NumOfdmSymbols + 1):
        is_pilot = (kk % L == 1)  # Pilot every L symbols
        print(f"Processing {'pilot' if is_pilot else 'data'} symbol {kk}")

        # Channel Generation
        batch_size = 1
        num_time_samples = 1
        a, tau = cdl(batch_size, num_time_samples, W)
        a = a.numpy()
        tau = tau.numpy()
        h_time = cir_to_time_channel(W, a, tau, l_min=0, l_max=IsiDuration-1, normalize=True).numpy()
        for nnn in range(N_r):
            for mmm in range(N_t):
                c[nnn][mmm] = h_time[0, 0, nnn, 0, mmm, 0, :]
                Ci[nnn][mmm] = np.fft.fft(np.append(c[nnn][mmm], np.zeros(N - IsiDuration)))

        # Transmission
        constellation = ConstPilot if is_pilot else Const
        m_curr = m_pilot if is_pilot else m
        TxBits = (np.random.uniform(0, 1, size=(N * m_curr, N_t)) > 0.5).astype(np.int32)
        X = np.zeros((N, N_t)).astype('complex128')
        x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
        for ii in range(N):
            for iii in range(N_t):
                ThisQamIdx = np.matmul(PowersOfTwo[:m_curr], TxBits[m_curr * ii + np.arange(m_curr), iii])
                X[ii, iii] = constellation[ThisQamIdx[0]]
        for iii in range(N_t):
            x_temp = N * np.fft.ifft(X[:, iii])
            x_CP[:, iii] = np.append(x_temp[-(CyclicPrefixLen):], x_temp)
            x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

        # Nonlinear PA
        x_CP_NLD = x_CP / ((1 + (np.abs(x_CP) / A_Clip) ** (2 * p_smooth)) ** (1 / (2 * p_smooth)))

        # Channel and Noise
        y_CP_NLD = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
        for nnn in range(N_r):
            for mmm in range(N_t):
                h_cdl = h_time[0, 0, nnn, 0, mmm, 0, :]
                conv_result = np.convolve(x_CP_NLD[:, mmm], h_cdl, mode='full')
                y_CP_NLD[:, nnn] += conv_result[:N + CyclicPrefixLen]
            noise = np.sqrt(No / 2) * (np.random.normal(size=(N + CyclicPrefixLen)) + 1j * np.random.normal(size=(N + CyclicPrefixLen)))
            y_CP_NLD[:, nnn] += noise
        Y = np.zeros((N, N_r)).astype('complex128')
        for nnn in range(N_r):
            Y[:, nnn] = 1 / N * np.fft.fft(y_CP_NLD[IsiDuration - 1:, nnn])

        # Orthogonal Pilots for LS/MMSE
        if is_pilot:
            X_LS = X.copy()
            X_LS[np.arange(1, len(X_LS), 2), 0] = 0
            X_LS[np.arange(0, len(X_LS), 2), 1] = 0
            x_LS_CP = np.zeros(x_CP.shape).astype('complex128')
            for iii in range(N_t):
                x_temp = N * np.fft.ifft(X_LS[:, iii])
                x_LS_CP[:, iii] = np.append(x_temp[-(CyclicPrefixLen):], x_temp)
                x_LS_CP[:, iii] = x_LS_CP[:, iii] * (Pi[jj] ** 0.5)
            y_LS_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
            for nnn in range(N_r):
                for mmm in range(N_t):
                    h_cdl = h_time[0, 0, nnn, 0, mmm, 0, :]
                    conv_result = np.convolve(x_LS_CP[:, mmm], h_cdl, mode='full')
                    y_LS_CP[:, nnn] += conv_result[:N + CyclicPrefixLen]
                noise = np.sqrt(No / 2) * (np.random.normal(size=(N + CyclicPrefixLen)) + 1j * np.random.normal(size=(N + CyclicPrefixLen)))
                y_LS_CP[:, nnn] += noise
            Y_LS = np.zeros((N, N_r)).astype('complex128')
            for nnn in range(N_r):
                Y_LS[:, nnn] = 1 / N * np.fft.fft(y_LS_CP[IsiDuration - 1:, nnn])
            Y_LS = Y_LS / (Pi[jj] ** 0.5)

            # LS and MMSE Channel Estimation
            for nnn in range(N_r):
                for mmm in range(N_t):
                    Ci_LS_Pilots[nnn][mmm] = Y_LS[np.arange(mmm, len(Y_LS), 2), nnn] / X_LS[np.arange(mmm, len(X_LS), 2), mmm]
                    c_LS = np.fft.ifft(Ci_LS_Pilots[nnn][mmm])[:IsiDuration]
                    c_MMSE = np.matmul(R_h, np.matmul(MMSE_bold_TD, c_LS))
                    Ci_MMSE[nnn][mmm] = np.fft.fft(np.append(c_MMSE, np.zeros(N - IsiDuration)))
                    interp_x = np.append(np.arange(mmm, N, N_t), N - 1 if mmm == 0 else np.arange(mmm, N, N_t))
                    interp_y = np.append(Ci_LS_Pilots[nnn][mmm], Ci_LS_Pilots[nnn][mmm][-1] if mmm == 0 else Ci_LS_Pilots[nnn][mmm])
                    if mmm > 0:
                        interp_x = np.append(0, interp_x)
                        interp_y = np.append(Ci_LS_Pilots[nnn][mmm][0], interp_y)
                    tmpf = interpolate.interp1d(interp_x, interp_y)
                    Ci_LS[nnn][mmm] = tmpf(np.arange(N))

            # ESN Training
            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits, spectral_radius=spectralRadius,
                      sparsity=1 - min(0.2 * nInternalUnits, 1), input_shift=inputShift, input_scaling=inputScaling,
                      teacher_scaling=teacherScaling, teacher_shift=teacherShift, feedback_scaling=feedbackScaling)
            [ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Minn, Delay_Max, nForgetPoints, NMSE_ESN] = \
                train_mimo_esn(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP_NLD, x_CP)
            ESN_train_input[kk - 1][jj] = ESN_input
            ESN_train_teacher[kk - 1][jj] = ESN_output
            NMSE_ESN_Training[jj] += NMSE_ESN / NumOfdmSymbols

        # Data Symbol Detection (for non-pilot symbols)
        if not is_pilot:
            # ESN Prediction
            ESN_input = np.zeros((N + CyclicPrefixLen + Delay_Max, N_t * 2))
            ESN_input[:N + CyclicPrefixLen, 0] = y_CP_NLD[:, 0].real
            ESN_input[:N + CyclicPrefixLen, 1] = y_CP_NLD[:, 0].imag
            ESN_input[:N + CyclicPrefixLen, 2] = y_CP_NLD[:, 1].real
            ESN_input[:N + CyclicPrefixLen, 3] = y_CP_NLD[:, 1].imag
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
            x_hat_ESN_0 = x_hat_ESN_temp[Delay[0] - Delay_Minn: Delay[0] - Delay_Minn + N, 0] + 1j * x_hat_ESN_temp[Delay[1] - Delay_Minn: Delay[1] - Delay_Minn + N, 1]
            x_hat_ESN_1 = x_hat_ESN_temp[Delay[2] - Delay_Minn: Delay[2] - Delay_Minn + N, 2] + 1j * x_hat_ESN_temp[Delay[3] - Delay_Minn: Delay[3] - Delay_Minn + N, 3]
            X_hat_ESN = np.fft.fft(np.vstack((x_hat_ESN_0, x_hat_ESN_1)).T) / N

            # LS/MMSE/Perfect Detection
            X_hat_LS = np.zeros((N, N_t), dtype='complex128')
            X_hat_MMSE = np.zeros((N, N_t), dtype='complex128')
            X_hat_Perfect = np.zeros((N, N_t), dtype='complex128')
            for ii in range(N):
                H_LS = np.array([[Ci_LS[0][0][ii], Ci_LS[0][1][ii]], [Ci_LS[1][0][ii], Ci_LS[1][1][ii]]])
                H_MMSE = np.array([[Ci_MMSE[0][0][ii], Ci_MMSE[0][1][ii]], [Ci_MMSE[1][0][ii], Ci_MMSE[1][1][ii]]])
                H_Perfect = np.array([[Ci[0][0][ii], Ci[0][1][ii]], [Ci[1][0][ii], Ci[1][1][ii]]])
                y_vec = Y[ii, :]
                X_hat_LS[ii, :] = np.linalg.pinv(H_LS) @ y_vec
                X_hat_MMSE[ii, :] = np.linalg.pinv(H_MMSE) @ y_vec
                X_hat_Perfect[ii, :] = np.linalg.pinv(H_Perfect) @ y_vec

            # BER Calculation
            TotalBerNum_ESN += compute_ber(X_hat_ESN, X, Const, m) * NumBitsPerSymbol
            TotalBerNum_LS += compute_ber(X_hat_LS, X, Const, m) * NumBitsPerSymbol
            TotalBerNum_MMSE += compute_ber(X_hat_MMSE, X, Const, m) * NumBitsPerSymbol
            TotalBerNum_Perfect += compute_ber(X_hat_Perfect, X, Const, m) * NumBitsPerSymbol
            TotalBerDen += NumBitsPerSymbol

        # Data Flow Check for First Symbol
        if kk == 1:
            print("\n=== Pilot Symbol Data Flow ===")
            print(f"Channel c[0][0] shape: {c[0][0].shape}, first few taps: {c[0][0][:3]}")
            print(f"Frequency response Ci[0][0] shape: {Ci[0][0].shape}, first few values: {Ci[0][0][:3]}")
            print(f"Transmitted X shape: {X.shape}, first few symbols: {X[:3, :]}")
            print(f"Received Y shape: {Y.shape}, first few symbols: {Y[:3, :]}")
            print(f"LS Estimated Ci_LS[0][0] shape: {Ci_LS[0][0].shape}, first few values: {Ci_LS[0][0][:3]}")
            print(f"MMSE Estimated Ci_MMSE[0][0] shape: {Ci_MMSE[0][0].shape}, first few values: {Ci_MMSE[0][0][:3]}")
            print(f"ESN Input shape: {ESN_input.shape}, first few values: {ESN_input[:3, :]}")
            print(f"ESN Output shape: {ESN_output.shape}, first few values: {ESN_output[:3, :]}")
            print(f"ESN NMSE Training: {NMSE_ESN}")
            print(f"Optimal Delays: {Delay}, Delay Index: {Delay_Idx}")

    # Compute Average BER
    BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen if TotalBerDen > 0 else 0
    BER_LS[jj] = TotalBerNum_LS / TotalBerDen if TotalBerDen > 0 else 0
    BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen if TotalBerDen > 0 else 0
    BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen if TotalBerDen > 0 else 0

# Plot Results
plt.figure(figsize=(8, 6))
plt.semilogy(EbNoDB, BER_ESN, 'o-', label='ESN')
plt.semilogy(EbNoDB, BER_LS, 's-', label='LS')
plt.semilogy(EbNoDB, BER_MMSE, '^-', label='MMSE')
plt.semilogy(EbNoDB, BER_Perfect, 'd-', label='Perfect CSI')
plt.xlabel('Eb/No (dB)')
plt.ylabel('Bit Error Rate')
plt.title('BER vs Eb/No for MIMO-OFDM System')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

print("\n=== Final Results ===")
print(f"EbNoDB: {EbNoDB.tolist()}")
print(f"BER ESN: {BER_ESN.tolist()}")
print(f"BER LS: {BER_LS.tolist()}")
print(f"BER MMSE: {BER_MMSE.tolist()}")
print(f"BER Perfect: {BER_Perfect.tolist()}")
print(f"NMSE ESN Training: {NMSE_ESN_Training.tolist()}")