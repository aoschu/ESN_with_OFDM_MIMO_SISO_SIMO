import numpy as np
import math
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.channel.tr38901 import CDL, PanelArray
from sionna.channel import cir_to_time_channel
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from HelpFunc import HelpFunc
from pyESN import ESN


# Sionna Channel Setup
N_r, N_t = 2, 2
carrier_frequency = 3.5e9
bs_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization='dual',
    polarization_type='VH',
    antenna_pattern='38.901',
    carrier_frequency=carrier_frequency,
)
ut_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization='dual',
    polarization_type='VH',
    antenna_pattern='omni',
    carrier_frequency=carrier_frequency,
)
delay_spread = 300e-9
cdl = CDL(
    model="C",
    delay_spread=delay_spread,
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction="downlink",
    min_speed=0.,
    max_speed=0.,
    dtype=tf.complex64
)

# LDPC Parameters
code_rate = 1/2
n_ldpc = 256  # Matches N * m = 64 * 4
k_ldpc = int(n_ldpc * code_rate)  # 128 information bits
num_codewords = 1
if (64 * 4) != n_ldpc:
    raise ValueError("Number of bits per symbol per antenna must equal n_ldpc")
ldpc_encoder = LDPC5GEncoder(k=k_ldpc, n=n_ldpc, dtype=tf.float32)
ldpc_decoder = LDPC5GDecoder(encoder=ldpc_encoder, num_iter=20)

# Physical Parameters
W = 2 * 1.024e6
f_D = 100
No = 0.00001
IsiDuration = 8
cFlag = False
EbNoDB = np.arange(25, 31, 5).astype(np.int32)

# MIMO Parameters
N = 64
Subcarrier_Spacing = W / N
m = 4
m_pilot = 4
NumOfdmSymbols = 1000
Ptotal = 10 ** (EbNoDB / 10) * No * N * code_rate

# Secondary Parameters
T_OFDM = N / W
T_OFDM_Total = (N + IsiDuration - 1) / W
T_s = 1 / W
tau_c = 0.5 / f_D
L = math.floor(tau_c / T_OFDM_Total)
Pi = Ptotal / N
NumBitsPerSymbol = m * N
Const = HelpFunc.UnitQamConstellation(m)
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

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
R_h = np.eye(IsiDuration) * (No / 2)  # Simplified for MMSE without custom PDP

# Data Flow Check
print("=== Parameter Initialization ===")
print(f"EbNoDB: {EbNoDB}")
print(f"Ptotal shape: {Ptotal.shape}, values: {Ptotal}")
print(f"Subcarrier Spacing: {Subcarrier_Spacing:.2e} Hz")
print(f"Coherence Length (L): {L} symbols")
print(f"Data Constellation shape: {Const.shape}, first few points: {Const[:3]}")
print(f"BER Arrays shape: {BER_ESN.shape}")
print(f"Channel Matrix (R_h) shape: {R_h.shape}, diagonal: {np.diag(R_h)}")




#blcok 2

# Select first SNR
jj = 0
print(f"\n=== Processing SNR: EbNoDB = {EbNoDB[jj]} dB ===")

# SNR-specific parameters
inputScaling = inputScaler / (var_x[jj] ** 0.5) * np.ones(N_t * 2)
inputShift = inputOffset / inputScaler * np.ones(N_t * 2)
TotalBerNum_ESN = 0
TotalBerNum_LS = 0
TotalBerNum_MMSE = 0
TotalBerNum_Perfect = 0
TotalBerDen = 0
MMSE_bold_TD = np.linalg.inv(R_h + MMSEScaler[jj] / (N * (N / 2)) * np.eye(IsiDuration))

# Process one pilot symbol (kk=1)
kk = 1
print(f"Processing pilot symbol {kk}")

# Channel Generation using Sionna CDL
batch_size = 1
num_time_samples = 1
a, tau = cdl(batch_size, num_time_samples, W)
a = a.numpy()
tau = tau.numpy()

# Compute discrete-time channel impulse response
h_time = cir_to_time_channel(W, a, tau, l_min=0, l_max=IsiDuration-1, normalize=True)
h_time = h_time.numpy()

# Store channel for LS and MMSE estimation
for nnn in range(N_r):
    for mmm in range(N_t):
        c[nnn][mmm] = h_time[0, 0, nnn, 0, mmm, 0, :]
        Ci[nnn][mmm] = np.fft.fft(np.append(c[nnn][mmm], np.zeros(N - IsiDuration)))

# Pilot Transmission
TxBits = (np.random.uniform(0, 1, size=(N * m_pilot, N_t)) > 0.5).astype(np.int32)
X = np.zeros((N, N_t)).astype('complex128')
x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
for ii in range(N):
    for iii in range(N_t):
        ThisQamIdx = np.matmul(PowersOfTwo[:m_pilot], TxBits[m_pilot * ii + np.arange(m_pilot), iii])
        X[ii, iii] = ConstPilot[ThisQamIdx[0]]
for iii in range(N_t):
    x_temp = N * np.fft.ifft(X[:, iii])
    x_CP[:, iii] = np.append(x_temp[-(CyclicPrefixLen):], x_temp)
    x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

# Channel and Noise (No PA)
y_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
for nnn in range(N_r):
    for mmm in range(N_t):
        h_cdl = h_time[0, 0, nnn, 0, mmm, 0, :]
        conv_result = np.convolve(x_CP[:, mmm], h_cdl, mode='full')
        y_CP[:, nnn] += conv_result[:N + CyclicPrefixLen]
    noise = np.sqrt(No / 2) * (np.random.normal(size=(N + CyclicPrefixLen)) + 1j * np.random.normal(size=(N + CyclicPrefixLen)))
    y_CP[:, nnn] += noise
Y = np.zeros((N, N_r)).astype('complex128')
for nnn in range(N_r):
    Y[:, nnn] = 1 / N * np.fft.fft(y_CP[IsiDuration - 1:, nnn])

# Orthogonal Pilots for LS
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
[ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN] = \
    HelpFunc.trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP)
ESN_train_input[kk - 1][jj] = ESN_input
ESN_train_teacher[kk - 1][jj] = ESN_output
NMSE_ESN_Training[jj] += NMSE_ESN

# Data Flow Check
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



#blcok 3
# Process one data symbol (kk=10)
kk = 10
print(f"\n=== Processing data symbol {kk} ===")

# Data Transmission with LDPC Encoding
info_bits = (np.random.uniform(0, 1, size=(num_codewords * k_ldpc, N_t)) > 0.5).astype(np.float32)
TxBits = np.zeros((N * m, N_t), dtype=np.int32)
for iii in range(N_t):
    info_bits_tx = info_bits[:, iii].reshape(num_codewords, k_ldpc)
    coded_bits = ldpc_encoder(info_bits_tx).numpy()
    TxBits[:, iii] = coded_bits.flatten().astype(np.int32)
X = np.zeros((N, N_t)).astype('complex128')
x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
for ii in range(N):
    for iii in range(N_t):
        ThisQamIdx = np.matmul(PowersOfTwo[:m], TxBits[m * ii + np.arange(m), iii])
        X[ii, iii] = Const[ThisQamIdx[0]]
for iii in range(N_t):
    x_temp = N * np.fft.ifft(X[:, iii])
    x_CP[:, iii] = np.append(x_temp[-(CyclicPrefixLen):], x_temp)
    x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

# Reception (No PA)
y_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
for nnn in range(N_r):
    for mmm in range(N_t):
        h_cdl = h_time[0, 0, nnn, 0, mmm, 0, :]
        conv_result = np.convolve(x_CP[:, mmm], h_cdl, mode='full')
        y_CP[:, nnn] += conv_result[:N + CyclicPrefixLen]
    noise = np.sqrt(No / 2) * (np.random.normal(size=(N + CyclicPrefixLen)) + 1j * np.random.normal(size=(N + CyclicPrefixLen)))
    y_CP[:, nnn] += noise
Y = np.zeros((N, N_r)).astype('complex128')
for nnn in range(N_r):
    Y[:, nnn] = 1 / N * np.fft.fft(y_CP[IsiDuration - 1:, nnn])

# ESN Detection
ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max))
ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max))
ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max))
ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max))
nForgetPoints = Delay_Min + CyclicPrefixLen
x_hat_ESN_temp = trainedEsn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
x_hat_ESN_0 = x_hat_ESN_temp[Delay[0] - Delay_Min: Delay[0] - Delay_Min + N, 0] + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min: Delay[1] - Delay_Min + N, 1]
x_hat_ESN_1 = x_hat_ESN_temp[Delay[2] - Delay_Min: Delay[2] - Delay_Min + N, 2] + 1j * x_hat_ESN_temp[Delay[3] - Delay_Min: Delay[3] - Delay_Min + N, 3]
x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)
x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))
x = x_CP[IsiDuration - 1:, :]
NMSE_ESN_Testing[jj] = (
    np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0]) ** 2 / np.linalg.norm(x[:, 0]) ** 2 +
    np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1]) ** 2 / np.linalg.norm(x[:, 1]) ** 2
)
X_hat_ESN = np.zeros(X.shape).astype('complex128')
for ii in range(N_t):
    X_hat_ESN[:, ii] = 1 / N * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi[jj])
ESN_test_input[kk - 1][jj] = ESN_input
ESN_test_output[kk - 1][jj] = x_hat_ESN

# Classical Detection
H_temp = np.zeros((N_r, N_t)).astype('complex128')
H_temp_LS = np.zeros((N_r, N_t)).astype('complex128')
H_temp_MMSE = np.zeros((N_r, N_t)).astype('complex128')
X_hat_Perfect = np.zeros(X.shape).astype('complex128')
X_hat_LS = np.zeros(X.shape).astype('complex128')
X_hat_MMSE = np.zeros(X.shape).astype('complex128')
for ii in range(N):
    Y_temp = np.transpose(Y[ii, :])
    for nnn in range(N_r):
        for mmm in range(N_t):
            H_temp[nnn, mmm] = Ci[nnn][mmm][ii]
            H_temp_LS[nnn, mmm] = Ci_LS[nnn][mmm][ii]
            H_temp_MMSE[nnn, mmm] = Ci_MMSE[nnn][mmm][ii]
    X_hat_Perfect[ii, :] = np.linalg.solve(H_temp, Y_temp) / math.sqrt(Pi[jj])
    X_hat_LS[ii, :] = np.linalg.solve(H_temp_LS, Y_temp) / math.sqrt(Pi[jj])
    X_hat_MMSE[ii, :] = np.linalg.solve(H_temp_MMSE, Y_temp) / math.sqrt(Pi[jj])

# Bit Detection with LDPC Decoding
RxBits_ESN = np.zeros(TxBits.shape)
RxBits_LS = np.zeros(TxBits.shape)
RxBits_MMSE = np.zeros(TxBits.shape)
RxBits_Perfect = np.zeros(TxBits.shape)
for ii in range(N):
    for iii in range(N_t):
        # Perfect
        ThisQamIdx = np.argmin(np.absolute(Const - X_hat_Perfect[ii, iii]))
        ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
        ThisBits = np.array([int(i) for i in ThisBits])[::-1]
        RxBits_Perfect[m * ii: m * (ii + 1), iii] = ThisBits
        # ESN
        ThisQamIdx = np.argmin(np.absolute(Const - X_hat_ESN[ii, iii]))
        ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
        ThisBits = np.array([int(i) for i in ThisBits])[::-1]
        RxBits_ESN[m * ii: m * (ii + 1), iii] = ThisBits
        # LS
        ThisQamIdx = np.argmin(np.absolute(Const - X_hat_LS[ii, iii]))
        ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
        ThisBits = np.array([int(i) for i in ThisBits])[::-1]
        RxBits_LS[m * ii: m * (ii + 1), iii] = ThisBits
        # MMSE
        ThisQamIdx = np.argmin(np.absolute(Const - X_hat_MMSE[ii, iii]))
        ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
        ThisBits = np.array([int(i) for i in ThisBits])[::-1]
        RxBits_MMSE[m * ii: m * (ii + 1), iii] = ThisBits

# LDPC Decoding
decoded_bits_ESN = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
decoded_bits_LS = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
decoded_bits_MMSE = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
decoded_bits_Perfect = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
for iii in range(N_t):
    rx_bits_ESN = RxBits_ESN[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
    rx_bits_LS = RxBits_LS[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
    rx_bits_MMSE = RxBits_MMSE[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
    rx_bits_Perfect = RxBits_Perfect[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
    decoded_bits_ESN[:, iii] = ldpc_decoder(rx_bits_ESN).numpy().flatten().astype(np.int32)
    decoded_bits_LS[:, iii] = ldpc_decoder(rx_bits_LS).numpy().flatten().astype(np.int32)
    decoded_bits_MMSE[:, iii] = ldpc_decoder(rx_bits_MMSE).numpy().flatten().astype(np.int32)
    decoded_bits_Perfect[:, iii] = ldpc_decoder(rx_bits_Perfect).numpy().flatten().astype(np.int32)

# Accumulate BER
TotalBerNum_ESN += np.sum(decoded_bits_ESN != info_bits.astype(np.int32))
TotalBerNum_LS += np.sum(decoded_bits_LS != info_bits.astype(np.int32))
TotalBerNum_MMSE += np.sum(decoded_bits_MMSE != info_bits.astype(np.int32))
TotalBerNum_Perfect += np.sum(decoded_bits_Perfect != info_bits.astype(np.int32))
TotalBerDen += num_codewords * k_ldpc * N_t

# Data Flow Check
print("\n=== Data Symbol Data Flow ===")
print(f"Transmitted X shape: {X.shape}, first few symbols: {X[:3, :]}")
print(f"Received Y shape: {Y.shape}, first few symbols: {Y[:3, :]}")
print(f"ESN Estimated X_hat_ESN shape: {X_hat_ESN.shape}, first few symbols: {X_hat_ESN[:3, :]}")
print(f"Perfect Estimated X_hat_Perfect shape: {X_hat_Perfect.shape}, first few symbols: {X_hat_Perfect[:3, :]}")
print(f"LS Estimated X_hat_LS shape: {X_hat_LS.shape}, first few symbols: {X_hat_LS[:3, :]}")
print(f"MMSE Estimated X_hat_MMSE shape: {X_hat_MMSE.shape}, first few symbols: {X_hat_MMSE[:3, :]}")
print(f"ESN NMSE Testing: {NMSE_ESN_Testing[jj]}")
print(f"Bit Errors (this symbol) - ESN: {np.sum(decoded_bits_ESN != info_bits.astype(np.int32))}, "
      f"LS: {np.sum(decoded_bits_LS != info_bits.astype(np.int32))}, "
      f"MMSE: {np.sum(decoded_bits_MMSE != info_bits.astype(np.int32))}, "
      f"Perfect: {np.sum(decoded_bits_Perfect != info_bits.astype(np.int32))}")
print(f"Total Bits: {TotalBerDen}")

# Initialize NMSE count
NMSE_count = 1

# Loop over additional data symbols (kk=3 to 6)
for kk in range(3, 7):
    print(f"\n=== Processing data symbol {kk} ===")

    # Data Transmission with LDPC Encoding
    info_bits = (np.random.uniform(0, 1, size=(num_codewords * k_ldpc, N_t)) > 0.5).astype(np.float32)
    TxBits = np.zeros((N * m, N_t), dtype=np.int32)
    for iii in range(N_t):
        info_bits_tx = info_bits[:, iii].reshape(num_codewords, k_ldpc)
        coded_bits = ldpc_encoder(info_bits_tx).numpy()
        TxBits[:, iii] = coded_bits.flatten().astype(np.int32)
    X = np.zeros((N, N_t)).astype('complex128')
    x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
    for ii in range(N):
        for iii in range(N_t):
            ThisQamIdx = np.matmul(PowersOfTwo[:m], TxBits[m * ii + np.arange(m), iii])
            X[ii, iii] = Const[ThisQamIdx[0]]
    for iii in range(N_t):
        x_temp = N * np.fft.ifft(X[:, iii])
        x_CP[:, iii] = np.append(x_temp[-(CyclicPrefixLen):], x_temp)
        x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

    # Reception (No PA)
    y_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
    for nnn in range(N_r):
        for mmm in range(N_t):
            h_cdl = h_time[0, 0, nnn, 0, mmm, 0, :]
            conv_result = np.convolve(x_CP[:, mmm], h_cdl, mode='full')
            y_CP[:, nnn] += conv_result[:N + CyclicPrefixLen]
        noise = np.sqrt(No / 2) * (np.random.normal(size=(N + CyclicPrefixLen)) + 1j * np.random.normal(size=(N + CyclicPrefixLen)))
        y_CP[:, nnn] += noise
    Y = np.zeros((N, N_r)).astype('complex128')
    for nnn in range(N_r):
        Y[:, nnn] = 1 / N * np.fft.fft(y_CP[IsiDuration - 1:, nnn])

    # ESN Detection
    ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
    ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max))
    ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max))
    ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max))
    ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max))
    nForgetPoints = Delay_Min + CyclicPrefixLen
    x_hat_ESN_temp = trainedEsn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
    x_hat_ESN_0 = x_hat_ESN_temp[Delay[0] - Delay_Min: Delay[0] - Delay_Min + N, 0] + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min: Delay[1] - Delay_Min + N, 1]
    x_hat_ESN_1 = x_hat_ESN_temp[Delay[2] - Delay_Min: Delay[2] - Delay_Min + N, 2] + 1j * x_hat_ESN_temp[Delay[3] - Delay_Min: Delay[3] - Delay_Min + N, 3]
    x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
    x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)
    x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))
    x = x_CP[IsiDuration - 1:, :]
    this_nmse = (
        np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0]) ** 2 / np.linalg.norm(x[:, 0]) ** 2 +
        np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1]) ** 2 / np.linalg.norm(x[:, 1]) ** 2
    )
    NMSE_ESN_Testing[jj] += this_nmse
    NMSE_count += 1
    X_hat_ESN = np.zeros(X.shape).astype('complex128')
    for ii in range(N_t):
        X_hat_ESN[:, ii] = 1 / N * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi[jj])
    ESN_test_input[kk - 1][jj] = ESN_input
    ESN_test_output[kk - 1][jj] = x_hat_ESN

    # Classical Detection
    H_temp = np.zeros((N_r, N_t)).astype('complex128')
    H_temp_LS = np.zeros((N_r, N_t)).astype('complex128')
    H_temp_MMSE = np.zeros((N_r, N_t)).astype('complex128')
    X_hat_Perfect = np.zeros(X.shape).astype('complex128')
    X_hat_LS = np.zeros(X.shape).astype('complex128')
    X_hat_MMSE = np.zeros(X.shape).astype('complex128')
    for ii in range(N):
        Y_temp = np.transpose(Y[ii, :])
        for nnn in range(N_r):
            for mmm in range(N_t):
                H_temp[nnn, mmm] = Ci[nnn][mmm][ii]
                H_temp_LS[nnn, mmm] = Ci_LS[nnn][mmm][ii]
                H_temp_MMSE[nnn, mmm] = Ci_MMSE[nnn][mmm][ii]
        X_hat_Perfect[ii, :] = np.linalg.solve(H_temp, Y_temp) / math.sqrt(Pi[jj])
        X_hat_LS[ii, :] = np.linalg.solve(H_temp_LS, Y_temp) / math.sqrt(Pi[jj])
        X_hat_MMSE[ii, :] = np.linalg.solve(H_temp_MMSE, Y_temp) / math.sqrt(Pi[jj])

    # Bit Detection with LDPC Decoding
    RxBits_ESN = np.zeros(TxBits.shape)
    RxBits_LS = np.zeros(TxBits.shape)
    RxBits_MMSE = np.zeros(TxBits.shape)
    RxBits_Perfect = np.zeros(TxBits.shape)
    for ii in range(N):
        for iii in range(N_t):
            # Perfect
            ThisQamIdx = np.argmin(np.absolute(Const - X_hat_Perfect[ii, iii]))
            ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
            ThisBits = np.array([int(i) for i in ThisBits])[::-1]
            RxBits_Perfect[m * ii: m * (ii + 1), iii] = ThisBits
            # ESN
            ThisQamIdx = np.argmin(np.absolute(Const - X_hat_ESN[ii, iii]))
            ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
            ThisBits = np.array([int(i) for i in ThisBits])[::-1]
            RxBits_ESN[m * ii: m * (ii + 1), iii] = ThisBits
            # LS
            ThisQamIdx = np.argmin(np.absolute(Const - X_hat_LS[ii, iii]))
            ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
            ThisBits = np.array([int(i) for i in ThisBits])[::-1]
            RxBits_LS[m * ii: m * (ii + 1), iii] = ThisBits
            # MMSE
            ThisQamIdx = np.argmin(np.absolute(Const - X_hat_MMSE[ii, iii]))
            ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
            ThisBits = np.array([int(i) for i in ThisBits])[::-1]
            RxBits_MMSE[m * ii: m * (ii + 1), iii] = ThisBits

    # LDPC Decoding
    decoded_bits_ESN = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
    decoded_bits_LS = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
    decoded_bits_MMSE = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
    decoded_bits_Perfect = np.zeros((num_codewords * k_ldpc, N_t), dtype=np.int32)
    for iii in range(N_t):
        rx_bits_ESN = RxBits_ESN[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
        rx_bits_LS = RxBits_LS[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
        rx_bits_MMSE = RxBits_MMSE[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
        rx_bits_Perfect = RxBits_Perfect[:, iii].reshape(num_codewords, n_ldpc).astype(np.float32)
        decoded_bits_ESN[:, iii] = ldpc_decoder(rx_bits_ESN).numpy().flatten().astype(np.int32)
        decoded_bits_LS[:, iii] = ldpc_decoder(rx_bits_LS).numpy().flatten().astype(np.int32)
        decoded_bits_MMSE[:, iii] = ldpc_decoder(rx_bits_MMSE).numpy().flatten().astype(np.int32)
        decoded_bits_Perfect[:, iii] = ldpc_decoder(rx_bits_Perfect).numpy().flatten().astype(np.int32)

    # Accumulate BER
    bit_errors_ESN = np.sum(decoded_bits_ESN != info_bits.astype(np.int32))
    bit_errors_LS = np.sum(decoded_bits_LS != info_bits.astype(np.int32))
    bit_errors_MMSE = np.sum(decoded_bits_MMSE != info_bits.astype(np.int32))
    bit_errors_Perfect = np.sum(decoded_bits_Perfect != info_bits.astype(np.int32))
    TotalBerNum_ESN += bit_errors_ESN
    TotalBerNum_LS += bit_errors_LS
    TotalBerNum_MMSE += bit_errors_MMSE
    TotalBerNum_Perfect += bit_errors_Perfect
    TotalBerDen += num_codewords * k_ldpc * N_t

    # Per-symbol print
    print(f"Bit Errors (this symbol) - ESN: {bit_errors_ESN}, LS: {bit_errors_LS}, MMSE: {bit_errors_MMSE}, Perfect: {bit_errors_Perfect}")
    print(f"ESN NMSE (this symbol): {this_nmse}")

# Compute averages (1 pilot, 5 data symbols: kk=2,3,4,5,6)
num_symbols_so_far = 6
num_pilots = 1
BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen
BER_LS[jj] = TotalBerNum_LS / TotalBerDen
BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen
BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen
NMSE_ESN_Training[jj] /= num_pilots
NMSE_ESN_Testing[jj] /= NMSE_count

# Data Flow Check: Final for this SNR segment
print(f"\n=== Partial Results for SNR {EbNoDB[jj]} dB (after {num_symbols_so_far} symbols) ===")
print(f"BER_ESN: {BER_ESN[jj]:.6f}, BER_LS: {BER_LS[jj]:.6f}, BER_MMSE: {BER_MMSE[jj]:.6f}, BER_Perfect: {BER_Perfect[jj]:.6f}")
print(f"NMSE_ESN_Training: {NMSE_ESN_Training[jj]:.6f}")
print(f"NMSE_ESN_Testing: {NMSE_ESN_Testing[jj]:.6f}")
print(f"Total Bits Processed: {TotalBerDen}")

# Optional Plot
plt.figure(figsize=(10, 4))
plt.plot(x[:, 0].real, label='True Tx Signal (Real, Tx0)')
plt.plot(x_hat_ESN[:, 0].real, label='ESN Estimated (Real, Tx0)')
plt.legend()
plt.title('ESN vs True Time-Domain Signal (Last Data Symbol)')
plt.xlabel('Sample Index')
plt.ylabel('Real Part')
plt.grid(True)
plt.show()



# ESN Training with Diagnostics
print("\n=== ESN Training Diagnostics ===")
esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits, spectral_radius=spectralRadius,
          sparsity=1 - min(0.2 * nInternalUnits, 1), input_shift=inputShift, input_scaling=inputScaling,
          teacher_scaling=teacherScaling, teacher_shift=teacherShift, feedback_scaling=feedbackScaling)

# Prepare ESN input/output
ESN_input = np.zeros((N + Max_Delay + CyclicPrefixLen, N_t * 2))
ESN_output = np.zeros((N + Max_Delay + CyclicPrefixLen, N_t * 2))
ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Max_Delay))
ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Max_Delay))
ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Max_Delay))
ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Max_Delay))
ESN_output[:N + CyclicPrefixLen, 0] = x_CP[:, 0].real
ESN_output[:N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
ESN_output[:N + CyclicPrefixLen, 2] = x_CP[:, 1].real
ESN_output[:N + CyclicPrefixLen, 3] = x_CP[:, 1].imag

# Debug: Check input/output norms
print(f"ESN Input Norm: {np.linalg.norm(ESN_input)}")
print(f"ESN Output Norm: {np.linalg.norm(ESN_output)}")
print(f"Input Scaling: {inputScaling}, Teacher Scaling: {teacherScaling}")

# Train ESN without delay optimization for simplicity
nForgetPoints = CyclicPrefixLen
esn.fit(ESN_input, ESN_output, nForgetPoints)

# Predict and compute NMSE manually
x_hat_ESN_temp = esn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
x_hat_ESN_0 = x_hat_ESN_temp[:N, 0] + 1j * x_hat_ESN_temp[:N, 1]
x_hat_ESN_1 = x_hat_ESN_temp[:N, 2] + 1j * x_hat_ESN_temp[:N, 3]
x_hat_ESN = np.hstack((x_hat_ESN_0.reshape(-1, 1), x_hat_ESN_1.reshape(-1, 1)))
x = x_CP[IsiDuration - 1:, :]
nmse_manual = (
    np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0]) ** 2 / np.linalg.norm(x[:, 0]) ** 2 +
    np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1]) ** 2 / np.linalg.norm(x[:, 1]) ** 2
)

# Debug: Check prediction norm
print(f"ESN Prediction Norm: {np.linalg.norm(x_hat_ESN)}")
print(f"True Signal Norm: {np.linalg.norm(x)}")
print(f"Manual NMSE: {nmse_manual}")

# Store results
ESN_train_input[kk - 1][jj] = ESN_input
ESN_train_teacher[kk - 1][jj] = ESN_output
NMSE_ESN_Training[jj] += nmse_manual
trainedEsn = esn
Delay = [0, 0, 0, 0]  # Placeholder, no delay optimization
Delay_Idx = 0
Delay_Min = 0
Delay_Max = Max_Delay