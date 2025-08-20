import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from scipy import interpolate
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
SIMO (1 x N_r) version of Demo_MIMO_NonlinearBlockFadingChannel.py
- Leaves pyESN.py and HelpFunc.py untouched.
- Adapts the demo to SIMO (N_t=1, N_r>=1).
- ESN works on stacked time-domain Rx branches [Re/Im per antenna].
- Baselines use MRC combining with Perfect/LS/MMSE CSI.
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
N_r = 2   # <-- change this to any SIMO size you want (e.g., 2, 4, 8)

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
nInputUnits = 2*N_r      # Re/Im for each Rx
nOutputUnits = 2         # Re/Im of single Tx stream
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

# --------------------
# Simulation holders
# --------------------
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))

NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))

# Channel responses per Rx
c = [None for _ in range(N_r)]          # time domain taps per Rx
Ci = [None for _ in range(N_r)]         # freq domain per Rx
Ci_LS  = [None for _ in range(N_r)]
Ci_MMSE  = [None for _ in range(N_r)]

# MMSE constants
MMSEScaler_allSNR = (No/Pi) # vector across SNRs

# Time-domain correlation matrix (diagonal with tap powers)
R_h = np.zeros((IsiDuration, IsiDuration))
for ii in range(IsiDuration):
    R_h[ii, ii] = IsiMagnitude[ii]

def mrc_equalize(Y_list, H_list, power_scale):
    """
    Maximal-Ratio Combining equalizer in frequency domain.
    Y_list: list of length N_r, each is shape (N,)
    H_list: list of length N_r, each is shape (N,)
    power_scale: scalar (sqrt(Pi[j]))
    returns: X_hat shape (N,)
    """
    num = np.zeros_like(Y_list[0], dtype=complex)
    den = np.zeros_like(Y_list[0], dtype=float)
    for rr in range(len(Y_list)):
        num += np.conj(H_list[rr]) * Y_list[rr]
        den += np.abs(H_list[rr])**2
    den = np.maximum(den, 1e-12)
    return (num/den) / power_scale

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
            # Random SIMO channel (time domain) and its FFT per Rx
            for rr in range(N_r):
                c0 = np.random.normal(size=IsiDuration)/(2**0.5) + 1j*np.random.normal(size=IsiDuration)/(2**0.5)
                c0 = c0 * (IsiMagnitude**0.5)
                c[rr] = c0
                Ci[rr] = np.fft.fft(np.append(c0, np.zeros(N - len(c0))))

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

            # Channel + noise (time domain) per Rx
            y_CP_p_list = []
            for rr in range(N_r):
                y_cp = signal.lfilter(c[rr], np.array([1]), x_CP_p_NLD[:, 0])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * np.matmul(
                    np.random.normal(size=((N + CyclicPrefixLen), 2)),
                    np.array([[1], [1j]])
                ).reshape(-1)
                y_cp = y_cp + noise
                y_CP_p_list.append(y_cp.reshape(-1, 1))

            # Frequency-domain received pilots per Rx
            Y_p_list = [(1/N) * np.fft.fft(y_CP_p_list[rr][IsiDuration-1:, 0]) for rr in range(N_r)]

            # LS/MMSE channel estimate on full grid per Rx
            for rr in range(N_r):
                H_LS = (Y_p_list[rr] / X_p[:, 0]) / (Pi[jj]**0.5)
                c_LS = np.fft.ifft(H_LS)
                c_LS = np.delete(c_LS, np.arange(IsiDuration, len(c_LS)))
                c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS)
                Ci_MMSE[rr] = np.fft.fft(np.append(c_MMSE, np.zeros(N-IsiDuration)))
                Ci_LS[rr] = H_LS

            # -------- Train ESN on time domain --------
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

                # stack all Rx branches: [Re(y1), Im(y1), Re(y2), Im(y2), ...]
                for rr in range(N_r):
                    ESN_input[:, 2*rr + 0] = np.append(y_CP_p_list[rr][:, 0].real, np.zeros(dmax))
                    ESN_input[:, 2*rr + 1] = np.append(y_CP_p_list[rr][:, 0].imag, np.zeros(dmax))

                ESN_output[d0:(d0+N+CyclicPrefixLen), 0] = x_CP_p[:, 0].real
                ESN_output[d1:(d1+N+CyclicPrefixLen), 1] = x_CP_p[:, 0].imag

                nForgetPoints_tmp = dmin + CyclicPrefixLen
                esn.fit(ESN_input, ESN_output, nForgetPoints_tmp)

                x_hat_tmp = esn.predict(ESN_input, nForgetPoints_tmp, continuation=False)
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

            for rr in range(N_r):
                ESN_input[:, 2*rr + 0] = np.append(y_CP_p_list[rr][:, 0].real, np.zeros(Delay_Max))
                ESN_input[:, 2*rr + 1] = np.append(y_CP_p_list[rr][:, 0].imag, np.zeros(Delay_Max))

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

            # Channel + noise per Rx
            y_CP_NLD_list = []
            for rr in range(N_r):
                y_cp = signal.lfilter(c[rr], np.array([1]), x_CP_NLD[:, 0])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * np.matmul(
                    np.random.normal(size=((N + CyclicPrefixLen), 2)),
                    np.array([[1], [1j]])
                ).reshape(-1)
                y_cp = y_cp + noise
                y_CP_NLD_list.append(y_cp.reshape(-1, 1))

            # Freq-domain received per Rx
            Y_list = [(1/N) * np.fft.fft(y_CP_NLD_list[rr][IsiDuration-1:, 0]) for rr in range(N_r)]

            # -------- ESN Detector (prediction on time domain) --------
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, nInputUnits))
            for rr in range(N_r):
                ESN_input[:, 2*rr + 0] = np.append(y_CP_NLD_list[rr][:, 0].real, np.zeros(Delay_Max))
                ESN_input[:, 2*rr + 1] = np.append(y_CP_NLD_list[rr][:, 0].imag, np.zeros(Delay_Max))

            x_hat_ESN_temp = trainedEsn.predict(ESN_input, nForgetPoints, continuation=False)

            x_hat_td = (x_hat_ESN_temp[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0]
                        + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1])

            # NMSE tracking on data
            x_ref = x_CP[IsiDuration - 1:, 0]
            NMSE_ESN_Testing[jj] += np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2
            NMSE_count += 1

            # FFT to symbol domain for ESN
            X_hat_ESN = ((1/N) * np.fft.fft(x_hat_td) / math.sqrt(Pi[jj])).reshape(-1, 1)

            # -------- Channel-equalized baselines with MRC --------
            X_hat_Perfect = mrc_equalize([Y_list[rr] for rr in range(N_r)],
                                         [Ci[rr] for rr in range(N_r)],
                                         math.sqrt(Pi[jj]))

            X_hat_LS = mrc_equalize([Y_list[rr] for rr in range(N_r)],
                                    [Ci_LS[rr] for rr in range(N_r)],
                                    math.sqrt(Pi[jj]))

            X_hat_MMSE = mrc_equalize([Y_list[rr] for rr in range(N_r)],
                                      [Ci_MMSE[rr] for rr in range(N_r)],
                                      math.sqrt(Pi[jj]))

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
NMSE_ESN_Testing = NMSE_ESN_Testing / max(NMSE_count, 1)
NMSE_ESN_Training = NMSE_ESN_Training / max((NumOfdmSymbols - NMSE_count), 1)

# -------- Save and plot --------
BERvsEBNo = {"EBN0": EbNoDB, "BER": {"ESN": BER_ESN, "LS": BER_LS, "MMSE": BER_MMSE, "Perfect": BER_Perfect}}
with open("./BERvsEBNo_esn_simo.pkl", "wb") as f:
    pickle.dump(BERvsEBNo, f)

plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS', linewidth=1.5)
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE', linewidth=1.5)
plt.semilogy(EbNoDB, BER_Perfect, 'kx-', label='Perfect CSI', linewidth=1.5)
plt.legend()
plt.grid(True)
plt.title(f'SIMO (1x{N_r}) | {nInternalUnits} ESN neurons | Nonlinear PA + Block Fading')
plt.xlabel('E_b/N_0 [dB]')
plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.show()
