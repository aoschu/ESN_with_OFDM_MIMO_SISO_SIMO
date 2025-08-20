import mitsuba as mi
mi.set_variant("scalar_rgb")


from HelpFunc import HelpFunc
from pyESN import ESN
import numpy as np
import math
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
import csv
import tensorflow as tf
from sionna.channel.tr38901 import CDL, PanelArray
from sionna.channel import cir_to_time_channel
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

class HelpFunc:
    @staticmethod
    def UnitQamConstellation(Bi):
        EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2
        PamM = EvenSquareRoot
        PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32)
        PamConstellation = np.reshape(PamConstellation, (1, -1))
        SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
        C = SquareMatrix + 1j * (SquareMatrix.T)
        C_tmp = np.zeros(C.shape[0] * C.shape[1]).astype('complex128')
        for i in range(C.shape[1]):
            for j in range(C.shape[0]):
                C_tmp[i * C.shape[0] + j] = C[j][i]
        C = C_tmp
        return C / math.sqrt(np.mean(abs(C) ** 2))

def trainSISOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, IsiDuration, y_CP, x_CP):
    if DelayFlag:
        Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay) ** 2, 2)).astype('int32')
        count = 0
        temp = np.zeros(Delay_LUT.shape[0])
        for ii in range(Min_Delay, Max_Delay + 1):
            for jj in range(Min_Delay, Max_Delay + 1):
                Delay_LUT[count, :] = np.array([ii, jj])
                if abs(ii - jj) > 2:
                    temp[count] = 1
                count += 1
        Delay_LUT = np.delete(Delay_LUT, np.where(temp > 0)[0], axis=0)
    else:
        Delay_LUT = np.zeros((Max_Delay - Min_Delay + 1, 2)).astype('int32')
        for jjjj in range(Min_Delay, Max_Delay + 1):
            Delay_LUT[jjjj - Min_Delay, :] = jjjj * np.ones(2)

    Delay_Max = np.amax(Delay_LUT, axis=1)
    Delay_Min = np.amin(Delay_LUT, axis=1)
    NMSE_ESN_Training = np.zeros(Delay_LUT.shape[0])
    for jjj in range(Delay_LUT.shape[0]):
        Curr_Delay = Delay_LUT[jjj, :]
        ESN_input = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, 2))
        ESN_output = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, 2))
        ESN_input[:, 0] = np.append(y_CP.real, np.zeros(Delay_Max[jjj]))
        ESN_input[:, 1] = np.append(y_CP.imag, np.zeros(Delay_Max[jjj]))
        ESN_output[Curr_Delay[0]: Curr_Delay[0] + N + CyclicPrefixLen, 0] = x_CP.real
        ESN_output[Curr_Delay[1]: Curr_Delay[1] + N + CyclicPrefixLen, 1] = x_CP.imag
        nForgetPoints = Delay_Min[jjj] + CyclicPrefixLen
        esn.fit(ESN_input, ESN_output, nForgetPoints)
        x_hat_ESN_temp = esn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
        x_hat_ESN = x_hat_ESN_temp[Curr_Delay[0] - Delay_Min[jjj]: Curr_Delay[0] - Delay_Min[jjj] + N, 0] + 1j * x_hat_ESN_temp[Curr_Delay[1] - Delay_Min[jjj]: Curr_Delay[1] - Delay_Min[jjj] + N, 1]
        x = x_CP[IsiDuration - 1:]
        NMSE_ESN_Training[jjj] = np.linalg.norm(x_hat_ESN - x) ** 2 / np.linalg.norm(x) ** 2
    Delay_Idx = np.argmin(NMSE_ESN_Training)
    NMSE_ESN = np.min(NMSE_ESN_Training)
    Delay = Delay_LUT[Delay_Idx, :]
    ESN_input = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, 2))
    ESN_output = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, 2))
    ESN_input[:, 0] = np.append(y_CP.real, np.zeros(Delay_Max[Delay_Idx]))
    ESN_input[:, 1] = np.append(y_CP.imag, np.zeros(Delay_Max[Delay_Idx]))
    ESN_output[Delay[0]: Delay[0] + N + CyclicPrefixLen, 0] = x_CP.real
    ESN_output[Delay[1]: Delay[1] + N + CyclicPrefixLen, 1] = x_CP.imag
    nForgetPoints = Delay_Min[Delay_Idx] + CyclicPrefixLen
    esn.fit(ESN_input, ESN_output, nForgetPoints)
    Delay_Minn = Delay_Min[Delay_Idx]
    Delay_Maxx = Delay_Max[Delay_Idx]
    return [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Minn, Delay_Maxx, nForgetPoints, NMSE_ESN]

# Sionna Setup for SISO
carrier_frequency = 3.5e9
delay_spread = 300e-9
N_r, N_t = 1, 1
bs_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V', antenna_pattern='38.901', carrier_frequency=carrier_frequency)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V', antenna_pattern='omni', carrier_frequency=carrier_frequency)
cdl = CDL(model="C", delay_spread=delay_spread, carrier_frequency=carrier_frequency, ut_array=ut_array, bs_array=bs_array, direction="downlink", min_speed=0., max_speed=0., dtype=tf.complex64)

# LDPC Setup
code_rate = 1/2
n_ldpc = 64 * 4  # N * m
k_ldpc = int(n_ldpc * code_rate)
num_codewords = 1
ldpc_encoder = LDPC5GEncoder(k=k_ldpc, n=n_ldpc, dtype=tf.float32)
ldpc_decoder = LDPC5GDecoder(encoder=ldpc_encoder, num_iter=20)

# Physical parameters
W = 2*1.024e6
f_D = 100
No = 0.00001
IsiDuration = 8
EbNoDB = np.arange(25, 31, 5).astype(np.int32)  # Full range for complete graph

# Parameters
N = 64
Subcarrier_Spacing = W / N
m = 4
m_pilot = 4
NumOfdmSymbols = 500  # Balanced for time and statistics
Ptotal = 10**(EbNoDB/10)*No*N * code_rate

# Secondary Parameters
T_OFDM = N/W
T_OFDM_Total = (N + IsiDuration -1)/W
T_s = 1/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)
Pi = Ptotal/N
NumBitsPerSymbol = m*N
Const = HelpFunc.UnitQamConstellation(m)
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

# ESN Parameters
var_x = np.power(10, (EbNoDB/10))*No*N * code_rate
nInputUnits = 2
nOutputUnits = 2
nInternalUnits = 64
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.0000005*np.ones(2)
spectralRadius = 0.9
teacherShift = np.zeros(2)
feedbackScaling = feedbackScaler*np.ones(2)
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2
DelayFlag = 0

# Initializations
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))
NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))
c = np.zeros(IsiDuration).astype('complex128')
Ci = np.zeros(N).astype('complex128')
Ci_LS = np.zeros(N).astype('complex128')
Ci_MMSE = np.zeros(N).astype('complex128')
MMSEScaler = (No/Pi)
R_h = np.eye(IsiDuration) / IsiDuration  # Uniform PDP for simplicity

# Lists for constellation (collect from last data symbol per SNR)
const_tx = []
const_esn = []
const_mmse = []

# Simulation
for jj in range(len(EbNoDB)):
    print(f'EbNoDB = {EbNoDB[jj]}')
    inputScaling = inputScaler/(var_x[jj]**0.5)*np.ones(2)
    inputShift = inputOffset/inputScaler*np.ones(2)
    TotalBerNum_ESN = 0
    TotalBerNum_LS = 0
    TotalBerNum_MMSE = 0
    TotalBerNum_Perfect = 0
    TotalBerDen = 0
    NMSE_count = 0
    MMSE_bold_TD = np.linalg.inv(R_h + MMSEScaler[jj]/(N) * np.eye(IsiDuration))  # Fixed formula
    trainedEsn = None
    Delay = None
    Delay_Min = None
    Delay_Max = None
    const_tx_j = None
    const_esn_j = None
    const_mmse_j = None

    for kk in range(1, NumOfdmSymbols+1):
        # Generate channel using Sionna
        batch_size = 1
        num_time_samples = 1
        a, tau = cdl(batch_size, num_time_samples, W)
        a = a.numpy()
        tau = tau.numpy()
        h_time = cir_to_time_channel(W, a, tau, l_min=0, l_max=IsiDuration-1, normalize=True)
        h_time = h_time.numpy()
        c = h_time[0, 0, 0, 0, 0, 0, :]
        Ci = np.fft.fft(np.append(c, np.zeros(N - IsiDuration)))

        if (kk % L == 1):
            # Pilot symbol
            TxBits = (np.random.uniform(0, 1, size=(N*m_pilot,)) > 0.5).astype(np.int32)
            X = np.zeros(N).astype('complex128')
            x_CP = np.zeros(N+CyclicPrefixLen).astype('complex128')
            for ii in range(N):
                ThisQamIdx = np.matmul(PowersOfTwo[:m_pilot], TxBits[m_pilot * ii + np.arange(m_pilot)])
                X[ii] = ConstPilot[ThisQamIdx[0]]
            x_temp = N * np.fft.ifft(X)
            x_CP = np.append(x_temp[-CyclicPrefixLen:], x_temp)
            x_CP = x_CP * (Pi[jj]**0.5)

            y_CP = np.zeros(N + CyclicPrefixLen).astype('complex128')
            conv_result = np.convolve(x_CP, c, mode='full')
            y_CP = conv_result[:N + CyclicPrefixLen]
            noise = np.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j * np.random.randn(N + CyclicPrefixLen))
            y_CP += noise

            # LS and MMSE estimation
            Y = 1 / N * np.fft.fft(y_CP[IsiDuration-1:])
            Ci_LS = Y / X / (Pi[jj]**0.5)
            c_LS = np.fft.ifft(Ci_LS)[:IsiDuration]
            c_MMSE = np.matmul(MMSE_bold_TD, c_LS)
            Ci_MMSE = np.fft.fft(np.append(c_MMSE, np.zeros(N - IsiDuration)))

            # ESN Training
            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits, spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1), input_shift=inputShift, input_scaling=inputScaling, teacher_scaling=teacherScaling, teacher_shift=teacherShift, feedback_scaling=feedbackScaling)
            [ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN] = trainSISOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, IsiDuration, y_CP, x_CP)
            NMSE_ESN_Training[jj] += NMSE_ESN

        else:
            # Data symbol with LDPC
            info_bits = (np.random.rand(num_codewords * k_ldpc) > 0.5).astype(np.float32)
            TxBits = np.zeros(N*m, dtype=np.int32)
            info_bits_tx = info_bits.reshape(num_codewords, k_ldpc)
            coded_bits = ldpc_encoder(info_bits_tx).numpy()
            TxBits = coded_bits.flatten().astype(np.int32)
            X = np.zeros(N).astype('complex128')
            x_CP = np.zeros(N + CyclicPrefixLen).astype('complex128')
            for ii in range(N):
                ThisQamIdx = np.matmul(PowersOfTwo[:m], TxBits[m * ii + np.arange(m)])
                X[ii] = Const[ThisQamIdx[0]]
            x_temp = N * np.fft.ifft(X)
            x_CP = np.append(x_temp[-CyclicPrefixLen:], x_temp)
            x_CP = x_CP * (Pi[jj]**0.5)

            y_CP = np.zeros(N + CyclicPrefixLen).astype('complex128')
            conv_result = np.convolve(x_CP, c, mode='full')
            y_CP = conv_result[:N + CyclicPrefixLen]
            noise = np.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j * np.random.randn(N + CyclicPrefixLen))
            y_CP += noise
            Y = 1 / N * np.fft.fft(y_CP[IsiDuration-1:])

            # ESN Detection
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, 2))
            ESN_input[:, 0] = np.append(y_CP.real, np.zeros(Delay_Max))
            ESN_input[:, 1] = np.append(y_CP.imag, np.zeros(Delay_Max))
            nForgetPoints = Delay_Min + CyclicPrefixLen
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
            x_hat_ESN = x_hat_ESN_temp[Delay[0] - Delay_Min: Delay[0] - Delay_Min + N, 0] + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min: Delay[1] - Delay_Min + N, 1]
            x = x_CP[IsiDuration - 1:]
            this_nmse = np.linalg.norm(x_hat_ESN - x) ** 2 / np.linalg.norm(x) ** 2
            NMSE_ESN_Testing[jj] += this_nmse
            NMSE_count += 1
            X_hat_ESN = 1 / N * np.fft.fft(x_hat_ESN) / math.sqrt(Pi[jj])

            # Classical Detection
            X_hat_Perfect = Y / Ci / math.sqrt(Pi[jj])
            X_hat_LS = Y / Ci_LS / math.sqrt(Pi[jj])
            X_hat_MMSE = Y / Ci_MMSE / math.sqrt(Pi[jj])

            # Collect for constellation (last symbol)
            if kk == NumOfdmSymbols:
                const_tx_j = X.copy()
                const_esn_j = X_hat_ESN.copy()
                const_mmse_j = X_hat_MMSE.copy()

            # LLR Computation for LDPC (fixed to use min for max-log approx)
            sigma2 = No / 2
            def compute_llrs(x_hat, const, sigma2):
                llrs = np.zeros(N * m)
                const_power = np.mean(np.abs(const) ** 2)
                for ii in range(N):
                    symbol = x_hat[ii] / np.sqrt(const_power)
                    llr_bits = np.zeros(m)
                    for b in range(m):
                        idx_0 = np.bitwise_and(np.arange(len(const)), 1 << b) == 0
                        idx_1 = np.bitwise_and(np.arange(len(const)), 1 << b) != 0
                        dist_0 = np.min(np.abs(symbol - const[idx_0] / np.sqrt(const_power)) ** 2)
                        dist_1 = np.min(np.abs(symbol - const[idx_1] / np.sqrt(const_power)) ** 2)
                        llr_bits[b] = (dist_1 - dist_0) / (2 * sigma2)  # Positive for bit=1 likely
                    llrs[m * ii:m * (ii + 1)] = llr_bits
                return llrs

            llrs_ESN = compute_llrs(X_hat_ESN, Const, sigma2)
            llrs_LS = compute_llrs(X_hat_LS, Const, sigma2)
            llrs_MMSE = compute_llrs(X_hat_MMSE, Const, sigma2)
            llrs_Perfect = compute_llrs(X_hat_Perfect, Const, sigma2)

            decoded_bits_ESN = np.zeros(num_codewords * k_ldpc, dtype=np.int32)
            decoded_bits_LS = np.zeros(num_codewords * k_ldpc, dtype=np.int32)
            decoded_bits_MMSE = np.zeros(num_codewords * k_ldpc, dtype=np.int32)
            decoded_bits_Perfect = np.zeros(num_codewords * k_ldpc, dtype=np.int32)
            llrs_ESN_tx = llrs_ESN.reshape(num_codewords, n_ldpc).astype(np.float32)
            llrs_LS_tx = llrs_LS.reshape(num_codewords, n_ldpc).astype(np.float32)
            llrs_MMSE_tx = llrs_MMSE.reshape(num_codewords, n_ldpc).astype(np.float32)
            llrs_Perfect_tx = llrs_Perfect.reshape(num_codewords, n_ldpc).astype(np.float32)
            decoded_bits_ESN = ldpc_decoder(llrs_ESN_tx).numpy().flatten().astype(np.int32)
            decoded_bits_LS = ldpc_decoder(llrs_LS_tx).numpy().flatten().astype(np.int32)
            decoded_bits_MMSE = ldpc_decoder(llrs_MMSE_tx).numpy().flatten().astype(np.int32)
            decoded_bits_Perfect = ldpc_decoder(llrs_Perfect_tx).numpy().flatten().astype(np.int32)

            bit_errors_ESN = np.sum(decoded_bits_ESN != info_bits.astype(np.int32))
            bit_errors_LS = np.sum(decoded_bits_LS != info_bits.astype(np.int32))
            bit_errors_MMSE = np.sum(decoded_bits_MMSE != info_bits.astype(np.int32))
            bit_errors_Perfect = np.sum(decoded_bits_Perfect != info_bits.astype(np.int32))
            TotalBerNum_ESN += bit_errors_ESN
            TotalBerNum_LS += bit_errors_LS
            TotalBerNum_MMSE += bit_errors_MMSE
            TotalBerNum_Perfect += bit_errors_Perfect
            TotalBerDen += num_codewords * k_ldpc

    BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen
    BER_LS[jj] = TotalBerNum_LS / TotalBerDen
    BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen
    BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen
    NMSE_ESN_Testing[jj] /= NMSE_count if NMSE_count > 0 else 1
    NMSE_ESN_Training[jj] /= (NumOfdmSymbols / L)

    # Save constellation data for this SNR
    if const_tx_j is not None:
        const_tx.append(const_tx_j)
        const_esn.append(const_esn_j)
        const_mmse.append(const_mmse_j)

# Save and Plot BER
BERvsEBNo = {"EBN0": EbNoDB, "BER_ESN": BER_ESN, "BER_LS": BER_LS, "BER_MMSE": BER_MMSE, "BER_Perfect": BER_Perfect}
with open('./BERvsEBNo_esn_siso.pkl', 'wb') as f:
    pickle.dump(BERvsEBNo, f)

with open('ber_results_siso.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["EbNoDB (dB)", "BER_ESN", "BER_LS", "BER_MMSE", "BER_Perfect"])
    for i in range(len(EbNoDB)):
        writer.writerow([EbNoDB[i], BER_ESN[i], BER_LS[i], BER_MMSE[i], BER_Perfect[i]])

plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS', linewidth=1.5)
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE', linewidth=1.5)
plt.semilogy(EbNoDB, BER_Perfect, 'k-', label='Perfect', linewidth=1.5)
plt.legend()
plt.grid(True)
plt.title('BER vs SNR - ESN vs Baselines (Coded, SISO)')
plt.xlabel('Eb/N0 [dB]')
plt.ylabel('Bit Error Rate (BER)')
plt.tight_layout()
plt.savefig("ber_vs_snr_esn_siso.png", dpi=300)
plt.show()

# Constellation Plot (for last SNR as example; loop over const_* for all)
if const_tx:
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(const_tx[-1]), np.imag(const_tx[-1]), c='green', label='Transmitted', alpha=0.5)
    plt.scatter(np.real(const_esn[-1]), np.imag(const_esn[-1]), c='blue', label='ESN Equalized', alpha=0.5)
    plt.scatter(np.real(const_mmse[-1]), np.imag(const_mmse[-1]), c='red', label='MMSE Equalized', alpha=0.5)
    plt.scatter(np.real(Const), np.imag(Const), c='black', marker='x', label='Ideal 16-QAM')
    plt.grid(True)
    plt.axhline(0, color='gray', lw=1)
    plt.axvline(0, color='gray', lw=1)
    plt.title('Constellation Diagram at {} dB (ESN vs MMSE)'.format(EbNoDB[-1]))
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("constellation_esn_siso.png", dpi=300)
    plt.show()