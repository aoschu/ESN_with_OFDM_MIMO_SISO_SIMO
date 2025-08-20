import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from scipy import interpolate
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
SISO version of Demo_MIMO_NonlinearBlockFadingChannel.py
- Keeps pyESN and HelpFunc unchanged.
- Only adapts the demo to SISO (N_t=1, N_r=1).
"""

# --------------------
# Physical parameters
# --------------------
W = 2*1.024e6          # Available Bandwidth
f_D = 100              # Doppler Frequency
No = 0.00001           # Noise power spectral density
IsiDuration = 8        # Number of multipath components
cFlag = False          # Fixed CIR flag (not used)
EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)

# --------------------
# Antenna parameters
# --------------------
N_t = 1
N_r = 1

# --------------------
# Design parameters
# --------------------
N = 512                       # Subcarriers
Subcarrier_Spacing = W/N
m = 4                         # QAM order for data (16-QAM)
m_pilot = 4                   # QAM order for pilots
NumOfdmSymbols = 400         # OFDM symbols per SNR point
Ptotal = 10**(EbNoDB/10)*No*N # Total power per OFDM symbol

# --------------------
# Power Amplifier
# --------------------
p_smooth = 1
ClipLeveldB = 3

# --------------------
# Secondary parameters
# --------------------
T_OFDM = N/W
T_OFDM_Total = (N + IsiDuration - 1)/W
T_s = 1/W
tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)    # Coherence time (OFDM symbols)
Pi = Ptotal/N                          # Equal power per subcarrier
NumBitsPerSymbol = m*N
Const = HelpFunc.UnitQamConstellation(m)
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)
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
# Time-domain channel input variance (per SNR)
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

# --------------------
# Simulation holders
# --------------------
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))

# Channel responses (for completeness; each is now scalar-length arrays)
c = [None]             # time-domain CIR c[rx*tx] => just [0]
Ci = [None]            # freq-domain true
Ci_LS  = [None]
Ci_MMSE  = [None]

# MMSE constants
MMSEScaler_allSNR = (No/Pi) # vector across SNRs

# Time-domain correlation matrix (diagonal with tap powers)
R_h = np.zeros((IsiDuration, IsiDuration))
for ii in range(IsiDuration):
    R_h[ii, ii] = IsiMagnitude[ii]

for jj in range(len(EbNoDB)):
    print('EbNoDB = %d' % EbNoDB[jj])
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    # ESN scaling per SNR
    inputScaling = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling = teacherScalingBase * np.ones(nOutputUnits)

    # Reset BER counters
    TotalBerNum_ESN = 0
    TotalBerNum_LS = 0
    TotalBerNum_MMSE = 0
    TotalBerNum_Perfect = 0
    TotalBerDen = 0

    NMSE_count = 0
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler_allSNR[jj]/(N/2)) + np.eye(IsiDuration)

    # placeholders carried from training to data
    trainedEsn = None
    Delay = None
    Delay_Min = None
    Delay_Max = None
    nForgetPoints = None

    for kk in range(1, NumOfdmSymbols+1):
        # New channel realization at start of each coherence block
        if (np.remainder(kk, L) == 1):
            # Random SISO channel (time domain) and its FFT
            c0 = np.random.normal(size=IsiDuration)/(2**0.5) + 1j*np.random.normal(size=IsiDuration)/(2**0.5)
            c0 = c0 * (IsiMagnitude**0.5)
            c[0] = c0
            Ci[0] = np.fft.fft(np.append(c0, np.zeros(N - len(c0))))

            # -------- Pilot OFDM symbol (full grid pilots) --------
            TxBitsPilot = (np.random.uniform(0, 1, size=(N*m_pilot, 1)) > 0.5).astype(np.int32)
            X_p = np.zeros((N, 1), dtype='complex128')
            x_CP_p = np.zeros((N + CyclicPrefixLen, 1), dtype='complex128')
            for ii in range(N):
                idx = np.matmul(PowersOfTwoPilot, TxBitsPilot[m_pilot*ii + np.arange(m_pilot), 0])
                X_p[ii, 0] = ConstPilot[idx[0]]

            # IFFT + CP and power loading
            x_temp = N * np.fft.ifft(X_p[:, 0])
            x_CP_p[:, 0] = np.append(x_temp[(-1 - CyclicPrefixLen + 1):], x_temp)
            x_CP_p[:, 0] = x_CP_p[:, 0] * (Pi[jj]**0.5)

            # Pass PA nonlinearity
            x_CP_p_NLD = x_CP_p / ((1 + (np.abs(x_CP_p)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + noise (time domain)
            y_CP_p = np.zeros((N + CyclicPrefixLen, 1), dtype='complex128')
            y_CP_p[:, 0] = signal.lfilter(c[0], np.array([1]), x_CP_p_NLD[:, 0])
            noise = math.sqrt(y_CP_p.shape[0]*No/2) * np.matmul(
                np.random.normal(size=(y_CP_p.shape[0], 2)),
                np.array([[1], [1j]])
            ).reshape(-1)
            y_CP_p[:, 0] = y_CP_p[:, 0] + noise

            # Frequency-domain received pilots
            Y_p = (1/N) * np.fft.fft(y_CP_p[IsiDuration-1:, 0]).reshape(-1, 1)

            # LS channel estimate on full grid
            H_LS = (Y_p[:, 0] / X_p[:, 0]) / (Pi[jj]**0.5)
            # MMSE (via TD truncation and TD-MMSE matrix)
            c_LS = np.fft.ifft(H_LS)
            c_LS = np.delete(c_LS, np.arange(IsiDuration, len(c_LS)))
            c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS)
            Ci_MMSE[0] = np.fft.fft(np.append(c_MMSE, np.zeros(N-IsiDuration)))
            Ci_LS[0] = H_LS

            # -------- Train ESN on time domain --------
            # Use the pilot time-domain snapshot for training (teacher: pre-PA transmit CP signal)
            # Build a simple delay LUT for 2 outputs (Re, Im)
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

            # ESN instance per coherence block / SNR
            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                      spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                      input_shift=inputShift, input_scaling=inputScaling,
                      teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                      feedback_scaling=feedbackScaling)

            for didx in range(Delay_LUT.shape[0]):
                d0, d1 = Delay_LUT[didx, :]
                dmax = Delay_Max_vec[didx]
                dmin = Delay_Min_vec[didx]

                ESN_input = np.zeros((N + dmax + CyclicPrefixLen, nInputUnits))
                ESN_output = np.zeros((N + dmax + CyclicPrefixLen, nOutputUnits))

                ESN_input[:, 0] = np.append(y_CP_p[:, 0].real, np.zeros(dmax))
                ESN_input[:, 1] = np.append(y_CP_p[:, 0].imag, np.zeros(dmax))

                ESN_output[d0:(d0+N+CyclicPrefixLen), 0] = x_CP_p[:, 0].real
                ESN_output[d1:(d1+N+CyclicPrefixLen), 1] = x_CP_p[:, 0].imag

                nForgetPoints_tmp = dmin + CyclicPrefixLen
                esn.fit(ESN_input, ESN_output, nForgetPoints_tmp)
                # Predict on the same input for NMSE computation
                x_hat_tmp = esn.predict(ESN_input, nForgetPoints_tmp, continuation=False)

                # stitch complex prediction and compute NMSE on useful N samples (post-CP)
                x_hat_td = (x_hat_tmp[d0 - dmin : d0 - dmin + N + 1, 0]
                            + 1j * x_hat_tmp[d1 - dmin : d1 - dmin + N + 1, 1])

                # reference TD signal after CP removal
                x_ref = x_CP_p[IsiDuration - 1:, 0]
                NMSE = np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2
                NMSE_list[didx] = NMSE

            Delay_Idx = np.argmin(NMSE_list)
            Delay = Delay_LUT[Delay_Idx, :]
            Delay_Min = Delay_Min_vec[Delay_Idx]
            Delay_Max = Delay_Max_vec[Delay_Idx]
            NMSE_ESN = NMSE_list[Delay_Idx]
            NMSE_ESN_Training[jj] += NMSE_ESN

            # Train final ESN with chosen delays
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, nInputUnits))
            ESN_output = np.zeros((N + Delay_Max + CyclicPrefixLen, nOutputUnits))
            ESN_input[:, 0] = np.append(y_CP_p[:, 0].real, np.zeros(Delay_Max))
            ESN_input[:, 1] = np.append(y_CP_p[:, 0].imag, np.zeros(Delay_Max))
            ESN_output[Delay[0]:(Delay[0]+N+CyclicPrefixLen), 0] = x_CP_p[:, 0].real
            ESN_output[Delay[1]:(Delay[1]+N+CyclicPrefixLen), 1] = x_CP_p[:, 0].imag

            nForgetPoints = Delay_Min + CyclicPrefixLen
            esn.fit(ESN_input, ESN_output, nForgetPoints)
            trainedEsn = esn

        else:
            # -------- Data OFDM symbol --------
            TxBits = (np.random.uniform(0, 1, size=(N*m, 1)) > 0.5).astype(np.int32)

            X = np.zeros((N, 1), dtype='complex128')
            x_CP = np.zeros((N + CyclicPrefixLen, 1), dtype='complex128')

            # QAM mapping
            for ii in range(N):
                idx = np.matmul(PowersOfTwo, TxBits[m*ii + np.arange(m), 0])
                X[ii, 0] = Const[idx[0]]

            # IFFT + CP + power loading
            x_temp = N * np.fft.ifft(X[:, 0])
            x_CP[:, 0] = np.append(x_temp[(-1 - CyclicPrefixLen + 1):], x_temp)
            x_CP[:, 0] = x_CP[:, 0] * (Pi[jj]**0.5)

            # PA nonlinearity
            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            # Channel + noise
            y_CP_NLD = np.zeros((N + CyclicPrefixLen, 1), dtype='complex128')
            y_CP_NLD[:, 0] = signal.lfilter(c[0], np.array([1]), x_CP_NLD[:, 0])
            noise = math.sqrt(y_CP_NLD.shape[0]*No/2) * np.matmul(
                np.random.normal(size=(y_CP_NLD.shape[0], 2)),
                np.array([[1], [1j]])
            ).reshape(-1)
            y_CP_NLD[:, 0] = y_CP_NLD[:, 0] + noise

            # Freq-domain received
            Y_NLD = (1/N) * np.fft.fft(y_CP_NLD[IsiDuration-1:, 0]).reshape(-1, 1)

            # -------- ESN Detector (prediction on time domain) --------
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, nInputUnits))
            ESN_input[:, 0] = np.append(y_CP_NLD[:, 0].real, np.zeros(Delay_Max))
            ESN_input[:, 1] = np.append(y_CP_NLD[:, 0].imag, np.zeros(Delay_Max))
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, nForgetPoints, continuation=False)

            x_hat_td = (x_hat_ESN_temp[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0]
                        + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1])

            # NMSE tracking on data
            x_ref = x_CP[IsiDuration - 1:, 0]
            NMSE_ESN_Testing[jj] += np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2
            NMSE_count += 1

            # FFT to symbol domain
            X_hat_ESN = ((1/N) * np.fft.fft(x_hat_td) / math.sqrt(Pi[jj])).reshape(-1, 1)

            # -------- Channel-equalized baselines --------
            X_hat_Perfect = (Y_NLD[:, 0] / Ci[0]) / math.sqrt(Pi[jj])
            X_hat_LS = (Y_NLD[:, 0] / Ci_LS[0]) / math.sqrt(Pi[jj])
            X_hat_MMSE = (Y_NLD[:, 0] / Ci_MMSE[0]) / math.sqrt(Pi[jj])

            # -------- Bit decisions --------
            RxBits_ESN = np.zeros(TxBits.shape, dtype=int)
            RxBits_LS = np.zeros(TxBits.shape, dtype=int)
            RxBits_MMSE = np.zeros(TxBits.shape, dtype=int)
            RxBits_Perfect = np.zeros(TxBits.shape, dtype=int)

            for ii in range(N):
                # Perfect
                sym = X_hat_Perfect[ii]
                idx = np.argmin(np.abs(Const - sym))
                bits = list(format(idx, 'b').zfill(m))
                bits = np.array([int(i) for i in bits])[::-1]
                RxBits_Perfect[m*ii:m*(ii+1), 0] = bits

                # ESN
                sym = X_hat_ESN[ii, 0]
                idx = np.argmin(np.abs(Const - sym))
                bits = list(format(idx, 'b').zfill(m))
                bits = np.array([int(i) for i in bits])[::-1]
                RxBits_ESN[m*ii:m*(ii+1), 0] = bits

                # LS
                sym = X_hat_LS[ii]
                idx = np.argmin(np.abs(Const - sym))
                bits = list(format(idx, 'b').zfill(m))
                bits = np.array([int(i) for i in bits])[::-1]
                RxBits_LS[m*ii:m*(ii+1), 0] = bits

                # MMSE
                sym = X_hat_MMSE[ii]
                idx = np.argmin(np.abs(Const - sym))
                bits = list(format(idx, 'b').zfill(m))
                bits = np.array([int(i) for i in bits])[::-1]
                RxBits_MMSE[m*ii:m*(ii+1), 0] = bits

            # Accumulate BER
            TotalBerNum_ESN += np.sum(TxBits != RxBits_ESN)
            TotalBerNum_LS += np.sum(TxBits != RxBits_LS)
            TotalBerNum_MMSE += np.sum(TxBits != RxBits_MMSE)
            TotalBerNum_Perfect += np.sum(TxBits != RxBits_Perfect)
            TotalBerDen += NumBitsPerSymbol

    # Store BER per SNR
    BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen
    BER_LS[jj] = TotalBerNum_LS / TotalBerDen
    BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen
    BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen

# Average NMSE
NMSE_ESN_Testing = NMSE_ESN_Testing / NMSE_count
NMSE_ESN_Training = NMSE_ESN_Training / (NumOfdmSymbols - NMSE_count)

# -------- Save and plot --------
BERvsEBNo = {"EBN0": EbNoDB, "BER": BER_ESN}
with open("./BERvsEBNo_esn_siso.pkl", "wb") as f:
    pickle.dump(BERvsEBNo, f)

plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS', linewidth=1.5)
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE', linewidth=1.5)
plt.semilogy(EbNoDB, BER_Perfect, 'kx-', label='Perfect CSI', linewidth=1.5)
plt.legend()
plt.grid(True)
plt.title('SISO | 100 ESN neurons | Nonlinear PA + Block Fading')
plt.xlabel('E_b/N_0 [dB]')
plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.show()
