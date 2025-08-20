import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from scipy import interpolate
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# New model definitions
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 4, kernel_size=5, padding=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # batch, time, features -> batch, channels, time
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # back to batch, time, channels
        return x

class RNNModel(nn.Module):
    def __init__(self, hidden_size=100):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(4, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=4):
        super(FNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ELM:
    def __init__(self, input_size, hidden_size=100, output_size=4):
        self.hidden_size = hidden_size
        self.W_in = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.b = np.random.uniform(-1, 1, hidden_size)
        self.W_out = None

    def _activate(self, x):
        return np.tanh(x)

    def fit(self, inputs, targets):
        hidden = self._activate(np.dot(inputs, self.W_in.T) + self.b)
        extended = np.hstack((hidden, np.ones((inputs.shape[0], 1))))  # add bias
        self.W_out = np.dot(np.linalg.pinv(extended), targets).T

    def predict(self, inputs):
        hidden = self._activate(np.dot(inputs, self.W_in.T) + self.b)
        extended = np.hstack((hidden, np.ones((inputs.shape[0], 1))))
        return np.dot(extended, self.W_out.T)

# trainMIMOModel function
def trainMIMOModel(model_type, y_CP, x_CP, N, N_t, CyclicPrefixLen, IsiDuration):
    fixed_delay = 3
    Delay = [fixed_delay] * 4
    Delay_Min = fixed_delay
    Delay_Max = fixed_delay

    ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
    ESN_output = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))

    ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max))
    ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max))
    ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max))
    ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max))

    ESN_output[Delay[0]: Delay[0] + N + CyclicPrefixLen, 0] = x_CP[:, 0].real
    ESN_output[Delay[1]: Delay[1] + N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
    ESN_output[Delay[2]: Delay[2] + N + CyclicPrefixLen, 2] = x_CP[:, 1].real
    ESN_output[Delay[3]: Delay[3] + N + CyclicPrefixLen, 3] = x_CP[:, 1].imag

    nForgetPoints = Delay_Min + CyclicPrefixLen

    if model_type in ['CNN', 'RNN']:
        input_tensor = torch.from_numpy(ESN_input).float().unsqueeze(0)  # batch 1, time, features
        target_tensor = torch.from_numpy(ESN_output).float().unsqueeze(0)

        if model_type == 'CNN':
            model = CNNModel()
        elif model_type == 'RNN':
            model = RNNModel()

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(50):  
            optimizer.zero_grad()
            out = model(input_tensor)
            loss = loss_fn(out, target_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x_hat_temp = model(input_tensor).squeeze(0).numpy()

    elif model_type in ['FNN', 'ELM']:
        window = 8  
        num_samples = ESN_input.shape[0] - window + 1
        inputs_window = np.zeros((num_samples, 4 * window))
        for i in range(window - 1, ESN_input.shape[0]):
            inputs_window[i - window + 1] = ESN_input[i - window + 1:i + 1].flatten()

        targets_window = ESN_output[window - 1:, :]

        if model_type == 'ELM':
            model = ELM(4 * window)
            model.fit(inputs_window, targets_window)
            x_hat_temp_window = model.predict(inputs_window)

        elif model_type == 'FNN':
            input_tensor = torch.from_numpy(inputs_window).float()
            target_tensor = torch.from_numpy(targets_window).float()

            model = FNNModel(4 * window)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            for epoch in range(50):
                optimizer.zero_grad()
                out = model(input_tensor)
                loss = loss_fn(out, target_tensor)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                x_hat_temp_window = model(input_tensor).numpy()

        x_hat_temp = np.zeros(ESN_output.shape)
        x_hat_temp[window - 1:, :] = x_hat_temp_window
        nForgetPoints += window - 1  

    x_hat_0 = x_hat_temp[0: N, 0] + 1j * x_hat_temp[0: N, 1]
    x_hat_1 = x_hat_temp[0: N, 2] + 1j * x_hat_temp[0: N, 3]
    x_hat = np.column_stack((x_hat_0, x_hat_1))

    x = x_CP[IsiDuration - 1:, :]
    NMSE = (np.linalg.norm(x_hat[:, 0] - x[:, 0])**2 / np.linalg.norm(x[:, 0])**2 +
            np.linalg.norm(x_hat[:, 1] - x[:, 1])**2 / np.linalg.norm(x[:, 1])**2)

    return [ESN_input, ESN_output, model, Delay, 3, Delay_Min, Delay_Max, nForgetPoints, NMSE]

"""
Physical parameters
"""
# Available Bandwidth
W = 2*1.024e6
# Doppler Frequency
f_D = 100
# Noise power spectral density
No = 0.00001
# Number of multipath components
IsiDuration = 8
# This flag is used to set the CIR c to a fixed value.
cFlag = False
# Signal-to-noise ratio
#EbNoDB = np.arange(0, 30+1, 5).astype(np.int32)

EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)
'''
MIMO Parameters
'''
N_t = 2
N_r = 2

'''
Design Parameters
'''
# Number of Subcarriers
N = 512
# Subcarrier spacing of OFDM signals
Subcarrier_Spacing = W/N
# Data symbols QAM Modulation Order
m = 4
# Pilot Symbols Modulation Order
m_pilot = 4
# Number of OFDM symbols to simulate for the BER curve
NumOfdmSymbols = 400
# Total power available for allocation to the subcarriers
Ptotal = 10**(EbNoDB/10)*No*N

'''
Power Amplifier
'''
p_smooth = 1
ClipLeveldB = 3

'''
Secondary Parameters
'''
# OFDM Symbol Duration
T_OFDM = N/W
# OFDM symbol duration including the cyclic prefix.
T_OFDM_Total = (N + IsiDuration -1)/W
# Sampling Period
T_s = 1/W
# Channel Coherence Time
tau_c = 0.5/f_D
# Coherence time in terms of OFDM symbols
L = math.floor(tau_c/T_OFDM_Total)
# Equal power distribution over all subcarriers
Pi = Ptotal/N
# Number of bits for OFDM symbol
NumBitsPerSymbol = m*N
# The normalized signal constellation for data symbols
Const = HelpFunc.UnitQamConstellation(m)
# The normalized signal constellation for pilot symbols
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)

# This variable used for bit-symbol mapping
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
# Number of cyclic prefix samples
CyclicPrefixLen = IsiDuration - 1

'''
Initializations
'''
# Generate a one-sided exponential channel power profile, and normalize total power to 1
temp = CyclicPrefixLen/9 # This line guarantees that the last CIR tap has less power than 0.01 of the first path.
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp)
IsiMagnitude = IsiMagnitude/sum(IsiMagnitude)


'''
ESN Parameters
'''
# This is the variance of the time-domain
var_x = np.float_power(10, (EbNoDB/10))*No*N
# channel input sequence
nInputUnits = N_t*2
nOutputUnits = N_t*2
# This is the number of neurons in the reservoir. We set this value as a
# function of the time-domain channel input length.
nInternalUnits = 100
inputScaler = 0.005

inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.0000005*np.ones(N_t*2)
spectralRadius = 0.9

# during training.
# Secondary parameters
teacherShift = np.zeros(N_t*2) # No need to introduce a teacher shift
feedbackScaling = feedbackScaler*np.ones(N_t*2)

# Min_Delay and Max_Delay ar the min and max output delays considered in
# training the esn. When the DelayFlag is set more delay quadruplets are
# considered for training, which slows down the script.
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2;
DelayFlag = 0
ESN_train_input = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_train_teacher = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_test_input = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]
ESN_test_output = [ [None] * len(EbNoDB) for i in range(NumOfdmSymbols) ]


'''
Simulation
'''
# The BER and the NMSE matrices to store the simulation results.
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))
BER_CNN = np.zeros(len(EbNoDB))
BER_RNN = np.zeros(len(EbNoDB))
BER_FNN = np.zeros(len(EbNoDB))
BER_ELM = np.zeros(len(EbNoDB))

NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))
c =  [[None] * N_t for i in range(N_r)] # This cell array will store all channel impulse
# responses (CIRs) from the transmit antennas to the receive antennas.
Ci = [[None] * N_t for i in range(N_r)] # This cell array will store the channel frequency
# responses from the transmit antennas to the receive antennas.
Ci_LS  = [[None] * N_t for i in range(N_r)]
Ci_MMSE  = [[None] * N_t for i in range(N_r)]
Ci_LS_Pilots  = [[None] * N_t for i in range(N_r)]
# This is the 1/SNR constant that scales the identity matrix in MMSE channel estimation.
MMSEScaler = (No/Pi)
# Construct the time-domain channel correlation matrix
R_h = np.zeros((IsiDuration, IsiDuration))
for ii in range(IsiDuration):
    R_h[ii, ii] = IsiMagnitude[ii]


for jj in range(len(EbNoDB)):
    print('EbNoDB = %d' % EbNoDB[jj])
    A_Clip = np.sqrt(var_x[jj])* np.float_power(10, ClipLeveldB/20)

    # The ESN parameters that depend on the current SNR
    inputScaling = inputScaler/(var_x[jj]**0.5)*np.ones(N_t*2)
    inputShift = inputOffset/inputScaler*np.ones(N_t * 2)
    # Reset the accumulated number of bit errors for each new SNR value.
    TotalBerNum_ESN = 0
    TotalBerNum_LS = 0
    TotalBerNum_MMSE = 0
    TotalBerNum_Perfect = 0
    TotalBerNum_CNN = 0
    TotalBerNum_RNN = 0
    TotalBerNum_FNN = 0
    TotalBerNum_ELM = 0
    TotalBerDen = 0
    # This is just some random last C
    x_ISI = np.zeros(CyclicPrefixLen).astype('complex128')
    NMSE_count = 0
    MMSE_bold_TD = np.dot(np.linalg.inv(R_h), MMSEScaler[jj]/(N/2)) + np.eye(IsiDuration)

    for kk in range(1, NumOfdmSymbols+1):
        # Check if the current OFDM symbol is for training, or data transmission.
        if (np.remainder(kk, L) == 1):
            # Randomly generate a channel impulse response.
            for nnn in range(N_r):
                for mmm in range(N_t):
                    # Randomly generate a channel impulse response.
                    c[nnn][mmm] = np.random.normal(size=IsiDuration)/(2**0.5) + 1j * np.random.normal(size=IsiDuration)/(2 ** 0.5)
                    c[nnn][mmm] = c[nnn][mmm]*(IsiMagnitude**0.5)
                    Ci[nnn][mmm] = np.fft.fft( np.append(c[nnn][mmm], np.zeros(N - len(c[nnn][mmm]))) )


            '''
            ISI channel + AWGN
            '''
            # The data symbols are M-ary where M = 2^m. We have N_t independent
            # data streams, both of which use the same modulation scheme.
            TxBits = (np.random.uniform(0, 1, size=(N*m_pilot,N_t)) > 0.5).astype(np.int32)

            X = np.zeros((N, N_t)).astype('complex128')
            x_CP = np.zeros((N+CyclicPrefixLen, N_t)).astype('complex128')
            # Modulate the bits with M - ary QAM.
            for ii in range(N):
                for iii in range(N_t):
                    ThisQamIdx = np.matmul(PowersOfTwo[:m_pilot], TxBits[m_pilot * ii + np.arange(m_pilot), iii])
                    X[ii, iii] = ConstPilot[ThisQamIdx[0]]

            #　Create the time - domain signal for each transmit antennas and prepend the cyclic　prefix.
            for iii in range(N_t):
                x_temp = N * np.fft.ifft(X[:, iii])
                x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_CP[:, iii] = x_CP[:, iii] * (Pi[jj]**0.5)
            Y = np.zeros((N, N_r)).astype('complex128')
            y_CP = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')
            noise = np.zeros(y_CP.shape).astype('complex128')

            for nnn in range(N_r):
                    # Superimpose all transmitted and filtered streams to form
                    # the received time domain signal in each receive antenna.
                    for mmm in range(N_t):
                        y_CP[:,nnn] = y_CP[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP[:,mmm])
                    # Add noise to the signal at each receive antenna.
                    noise[:,nnn] = math.sqrt(y_CP.shape[0]*No/2) * np.matmul(np.random.normal(size=(y_CP.shape[0], 2)),
                                                              np.array([[1], [1j]])).reshape(-1)
                    y_CP[:,nnn] = y_CP[:,nnn] + noise[:,nnn]
                    # Get the frequency domain samples at each receive antenna.
                    Y[:, nnn] = 1 / N * np.fft.fft(y_CP[IsiDuration-1:len(y_CP),nnn])


            # Insert zeros at the subcarriers that are pilot locations for the other antenna.
            X_LS = X
            X_LS[np.arange(1, len(X_LS), 2), 0] = 0
            X_LS[np.arange(0, len(X_LS), 2), 1] = 0
            x_LS_CP = np.zeros(x_CP.shape).astype('complex128')
            for iii in range(N_t):
                x_temp = N*np.fft.ifft(X_LS[:,iii])
                x_LS_CP[:,iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_LS_CP[:,iii] = x_LS_CP[:,iii] * (Pi[jj] ** 0.5)
            Y_LS = np.zeros((N,N_r)).astype('complex128')
            y_LS_CP = np.zeros((N+CyclicPrefixLen, N_r)).astype('complex128')
            for nnn in range(N_r):
                for mmm in range(N_t):
                    y_LS_CP[:,nnn] = y_LS_CP[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_LS_CP[:, mmm])

                y_LS_CP[:, nnn] = y_LS_CP[:,nnn] + noise[:,nnn]
                # Get the frequency domain samples at each receive antenna.
                Y_LS[:,nnn] = 1/N*np.fft.fft(y_LS_CP[IsiDuration-1:,nnn])
            Y_LS = Y_LS/(Pi[jj]**0.5)
            '''
            LS Channel Estimation
            '''
            for nnn in range(N_r):
                for mmm in range(N_t):
                    Ci_LS_Pilots[nnn][mmm] = Y_LS[np.arange(mmm, len(Y_LS), 2), nnn]/ X_LS[np.arange(mmm, len(X_LS), 2), mmm]
                    # MMSE Channel Estimation
                    #  H_MMSE = MMSE_bold_FD * H_LS
                    c_LS = np.fft.ifft(Ci_LS_Pilots[nnn][mmm])
                    # Get time - domain LS estimates at the known CIR locations.
                    c_LS = np.delete(c_LS, np.arange(IsiDuration, len(c_LS)))
                    # Get the MMSE estimate from the LS estimate
                    c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS)
                    Ci_MMSE[nnn][mmm] = np.fft.fft(np.append(c_MMSE, np.zeros(N-IsiDuration)))
                    if (mmm == 0):
                        tmpf = interpolate.interp1d(np.append(np.arange(mmm, N-1, N_t), N-1),
                                 np.append(Ci_LS_Pilots[nnn][mmm], Ci_LS_Pilots[nnn][mmm][-1]))
                        Ci_LS[nnn][mmm] = tmpf(np.arange(N))
                    else:
                        tmpf = interpolate.interp1d(np.append(0, np.arange(mmm, N, N_t)),
                                 np.append(Ci_LS_Pilots[nnn][mmm][0], Ci_LS_Pilots[nnn][mmm]))
                        Ci_LS[nnn][mmm] = tmpf(np.arange(N))
            '''
            ESN Receiver
            '''
            # Pass the time-domain samples through the nonlinear PA
            x_CP_NLD = x_CP/( ( 1 + (np.absolute(x_CP)/A_Clip)**(2*p_smooth) )**(1/(2*p_smooth)) )

            y_CP_NLD = np.zeros((N+CyclicPrefixLen,N_r)).astype('complex128')
            for nnn in range(N_r):
                    # Superimpose all transmitted and filtered streams to form
                    # the received time domain signal in each receive antenna.
                    for mmm in range(N_t):
                        y_CP_NLD[:, nnn] = y_CP_NLD[:,nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
                    y_CP_NLD[:, nnn] = y_CP_NLD[:,nnn] + noise[:,nnn]
            # Set-up the ESN
            esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                                      spectral_radius=spectralRadius, sparsity= 1 - min(0.2*nInternalUnits, 1),
                                      input_shift=inputShift, input_scaling=inputScaling,
                                      teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                                      feedback_scaling=feedbackScaling)
            # Train the ESN
            [ESN_input, ESN_output, trainedEsn, Delay, Delay_Idx, Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN] = \
            HelpFunc.trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP_NLD, x_CP)

            # Data For AFRL
            #ESN_train_input[kk, jj] = ESN_input
            #ESN_train_teacher[kk, jj] = ESN_output

            NMSE_ESN_Training[jj] = NMSE_ESN + NMSE_ESN_Training[jj]

            # Train new models
            [_, _, trainedCNN, Delay_CNN, _, Delay_Min_CNN, Delay_Max_CNN, nForgetPoints_CNN, NMSE_CNN] = trainMIMOModel('CNN', y_CP_NLD, x_CP, N, N_t, CyclicPrefixLen, IsiDuration)
            [_, _, trainedRNN, Delay_RNN, _, Delay_Min_RNN, Delay_Max_RNN, nForgetPoints_RNN, NMSE_RNN] = trainMIMOModel('RNN', y_CP_NLD, x_CP, N, N_t, CyclicPrefixLen, IsiDuration)
            [_, _, trainedFNN, Delay_FNN, _, Delay_Min_FNN, Delay_Max_FNN, nForgetPoints_FNN, NMSE_FNN] = trainMIMOModel('FNN', y_CP_NLD, x_CP, N, N_t, CyclicPrefixLen, IsiDuration)
            [_, _, trainedELM, Delay_ELM, _, Delay_Min_ELM, Delay_Max_ELM, nForgetPoints_ELM, NMSE_ELM] = trainMIMOModel('ELM', y_CP_NLD, x_CP, N, N_t, CyclicPrefixLen, IsiDuration)

        else:
            TxBits = (np.random.uniform(0, 1, size=(N*m, N_t)) > 0.5).astype(np.int32)

            X = np.zeros((N, N_t)).astype('complex128')
            x_CP = np.zeros((N + CyclicPrefixLen, N_t)).astype('complex128')
            # Modulate the bits with M - ary QAM.
            for ii in range(N):
                for iii in range(N_t):
                    ThisQamIdx = np.matmul(PowersOfTwo[:m], TxBits[m * ii + np.arange(m), iii])
                    X[ii, iii] = Const[ThisQamIdx[0]]

            for iii in range(N_t):
                x_temp = N * np.fft.ifft(X[:, iii])
                x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                x_CP[:, iii] = x_CP[:, iii] * (Pi[jj] ** 0.5)

            # nonlinear Amplifier

            x_CP_NLD = x_CP / ((1 + (np.absolute(x_CP)/A_Clip) ** (2*p_smooth)) ** (1/(2*p_smooth)))


            Y_NLD = np.zeros((N, N_r)).astype('complex128')
            y_CP_NLD = np.zeros((N + CyclicPrefixLen, N_r)).astype('complex128')

            for nnn in range(N_r):
                # Superimpose all transmitted and filtered streams to form
                # the received time domain signal in each receive antenna.
                for mmm in range(N_t):
                    y_CP_NLD[:, nnn] = y_CP_NLD[:, nnn] + signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
# add noise
                y_CP_NLD[:, nnn] = y_CP_NLD[:, nnn] + math.sqrt(y_CP.shape[0] * No / 2) \
                     * np.matmul(np.random.normal(size=(y_CP.shape[0], 2)), np.array([[1], [1j]])).reshape(-1)
                Y_NLD[:, nnn] = 1 / N * np.fft.fft(y_CP_NLD[IsiDuration-1:, nnn])



            # ESN Detector
            ESN_input = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
            ESN_output = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))

            ESN_input[:, 0] = np.append(y_CP_NLD[:, 0].real, np.zeros(Delay_Max))
            ESN_input[:, 1] = np.append(y_CP_NLD[:, 0].imag, np.zeros(Delay_Max))
            ESN_input[:, 2] = np.append(y_CP_NLD[:, 1].real, np.zeros(Delay_Max))
            ESN_input[:, 3] = np.append(y_CP_NLD[:, 1].imag, np.zeros(Delay_Max))

            # Get the ESN output corresponding to the training input
            # Train the ESN
            nForgetPoints = Delay_Min + CyclicPrefixLen
            x_hat_ESN_temp = trainedEsn.predict(ESN_input, nForgetPoints, continuation=False)

            x_hat_ESN_0 = x_hat_ESN_temp[Delay[0] - Delay_Min: Delay[0] - Delay_Min + N, 0] \
                            + 1j * x_hat_ESN_temp[Delay[1] - Delay_Min: Delay[1] - Delay_Min + N, 1]
            x_hat_ESN_1 = x_hat_ESN_temp[Delay[2] - Delay_Min: Delay[2] - Delay_Min + N, 2] \
                            + 1j * x_hat_ESN_temp[Delay[3] - Delay_Min: Delay[3] - Delay_Min + N, 3]



            x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
            x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)
            x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))

            x = x_CP[IsiDuration - 1:, :]

            NMSE_ESN_Testing[jj] = NMSE_ESN_Testing[jj] \
                + np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0], axis=0) ** 2 / np.linalg.norm(x[:, 0], axis=0) ** 2 \
                + np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1], axis=0) ** 2 / np.linalg.norm(x[:, 1], axis=0) ** 2

            NMSE_count = NMSE_count + 1
            X_hat_ESN = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_ESN[:, ii] = 1 / N * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi[jj])
            # Data For AFRL
            #ESN_test_input[kk, jj] = ESN_input
            #ESN_test_output[kk, jj] = x_hat_ESN

            # Predictions for new models
            # CNN
            input_tensor = torch.from_numpy(ESN_input).float().unsqueeze(0)
            with torch.no_grad():
                x_hat_temp_CNN = trainedCNN(input_tensor).squeeze(0).numpy()
            x_hat_CNN_0 = x_hat_temp_CNN[Delay_CNN[0] - Delay_Min_CNN: Delay_CNN[0] - Delay_Min_CNN + N, 0] + 1j * x_hat_temp_CNN[Delay_CNN[1] - Delay_Min_CNN: Delay_CNN[1] - Delay_Min_CNN + N, 1]
            x_hat_CNN_1 = x_hat_temp_CNN[Delay_CNN[2] - Delay_Min_CNN: Delay_CNN[2] - Delay_Min_CNN + N, 2] + 1j * x_hat_temp_CNN[Delay_CNN[3] - Delay_Min_CNN: Delay_CNN[3] - Delay_Min_CNN + N, 3]
            x_hat_CNN = np.hstack((x_hat_CNN_0.reshape(-1, 1), x_hat_CNN_1.reshape(-1, 1)))
            X_hat_CNN = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_CNN[:, ii] = 1 / N * np.fft.fft(x_hat_CNN[:, ii]) / math.sqrt(Pi[jj])
            # RNN
            with torch.no_grad():
                x_hat_temp_RNN = trainedRNN(input_tensor).squeeze(0).numpy()
            x_hat_RNN_0 = x_hat_temp_RNN[Delay_RNN[0] - Delay_Min_RNN: Delay_RNN[0] - Delay_Min_RNN + N, 0] + 1j * x_hat_temp_RNN[Delay_RNN[1] - Delay_Min_RNN: Delay_RNN[1] - Delay_Min_RNN + N, 1]
            x_hat_RNN_1 = x_hat_temp_RNN[Delay_RNN[2] - Delay_Min_RNN: Delay_RNN[2] - Delay_Min_RNN + N, 2] + 1j * x_hat_temp_RNN[Delay_RNN[3] - Delay_Min_RNN: Delay_RNN[3] - Delay_Min_RNN + N, 3]
            x_hat_RNN = np.hstack((x_hat_RNN_0.reshape(-1, 1), x_hat_RNN_1.reshape(-1, 1)))
            X_hat_RNN = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_RNN[:, ii] = 1 / N * np.fft.fft(x_hat_RNN[:, ii]) / math.sqrt(Pi[jj])
            # FNN
            window = 8
            inputs_window = np.zeros((ESN_input.shape[0] - window + 1, 4 * window))
            for i in range(window - 1, ESN_input.shape[0]):
                inputs_window[i - window + 1] = ESN_input[i - window + 1:i + 1].flatten()
            input_tensor_fnn = torch.from_numpy(inputs_window).float()
            with torch.no_grad():
                x_hat_temp_window_FNN = trainedFNN(input_tensor_fnn).numpy()
            x_hat_temp_FNN = np.zeros((ESN_input.shape[0], 4))
            x_hat_temp_FNN[window - 1:, :] = x_hat_temp_window_FNN
            x_hat_FNN_0 = x_hat_temp_FNN[Delay_FNN[0] - Delay_Min_FNN: Delay_FNN[0] - Delay_Min_FNN + N, 0] + 1j * x_hat_temp_FNN[Delay_FNN[1] - Delay_Min_FNN: Delay_FNN[1] - Delay_Min_FNN + N, 1]
            x_hat_FNN_1 = x_hat_temp_FNN[Delay_FNN[2] - Delay_Min_FNN: Delay_FNN[2] - Delay_Min_FNN + N, 2] + 1j * x_hat_temp_FNN[Delay_FNN[3] - Delay_Min_FNN: Delay_FNN[3] - Delay_Min_FNN + N, 3]
            x_hat_FNN = np.hstack((x_hat_FNN_0.reshape(-1, 1), x_hat_FNN_1.reshape(-1, 1)))
            X_hat_FNN = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_FNN[:, ii] = 1 / N * np.fft.fft(x_hat_FNN[:, ii]) / math.sqrt(Pi[jj])
            # ELM
            x_hat_temp_window_ELM = trainedELM.predict(inputs_window)
            x_hat_temp_ELM = np.zeros((ESN_input.shape[0], 4))
            x_hat_temp_ELM[window - 1:, :] = x_hat_temp_window_ELM
            x_hat_ELM_0 = x_hat_temp_ELM[Delay_ELM[0] - Delay_Min_ELM: Delay_ELM[0] - Delay_Min_ELM + N, 0] + 1j * x_hat_temp_ELM[Delay_ELM[1] - Delay_Min_ELM: Delay_ELM[1] - Delay_Min_ELM + N, 1]
            x_hat_ELM_1 = x_hat_temp_ELM[Delay_ELM[2] - Delay_Min_ELM: Delay_ELM[2] - Delay_Min_ELM + N, 2] + 1j * x_hat_temp_ELM[Delay_ELM[3] - Delay_Min_ELM: Delay_ELM[3] - Delay_Min_ELM + N, 3]
            x_hat_ELM = np.hstack((x_hat_ELM_0.reshape(-1, 1), x_hat_ELM_1.reshape(-1, 1)))
            X_hat_ELM = np.zeros(X.shape).astype('complex128')
            for ii in range(N_t):
                X_hat_ELM[:, ii] = 1 / N * np.fft.fft(x_hat_ELM[:, ii]) / math.sqrt(Pi[jj])

            '''
            Channel Estimation
            '''
            H_temp = np.zeros((N_r, N_t)).astype('complex128')
            H_temp_LS = np.zeros((N_r, N_t)).astype('complex128')
            H_temp_MMSE = np.zeros((N_r, N_t)).astype('complex128')
            X_hat_Perfect = np.zeros(X.shape).astype('complex128')
            X_hat_LS = np.zeros(X.shape).astype('complex128')
            X_hat_MMSE = np.zeros(X.shape).astype('complex128')

            for ii in range(N):
                Y_temp = np.transpose(Y_NLD[ii,:])
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        H_temp[nnn, mmm] = Ci[nnn][mmm][ii]
                        H_temp_LS[nnn, mmm] = Ci_LS[nnn][mmm][ii]
                        H_temp_MMSE[nnn, mmm] = Ci_MMSE[nnn][mmm][ii]

                X_hat_Perfect[ii,:] = np.linalg.solve(H_temp, Y_temp) / math.sqrt(Pi[jj])
                X_hat_LS[ii,:] = np.linalg.solve(H_temp_LS, Y_temp) / math.sqrt(Pi[jj])
                X_hat_MMSE[ii,:] = np.linalg.solve(H_temp_MMSE, Y_temp) / math.sqrt(Pi[jj])

            RxBits_ESN = np.zeros(TxBits.shape)
            RxBits_LS = np.zeros(TxBits.shape)
            RxBits_MMSE = np.zeros(TxBits.shape)
            RxBits_Perfect = np.zeros(TxBits.shape)
            RxBits_CNN = np.zeros(TxBits.shape)
            RxBits_RNN = np.zeros(TxBits.shape)
            RxBits_FNN = np.zeros(TxBits.shape)
            RxBits_ELM = np.zeros(TxBits.shape)
            # Loop through the subcarriers and detect the QAM symbols and bits.
            for ii in range(N):
                for iii in range(N_t):
                    # Bit and symbol detection with the "exact" equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_Perfect[ii,iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_Perfect[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with ESN Receiver
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_ESN[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_ESN[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with LS equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_LS[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_LS[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with MMSE equalizer
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_MMSE[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_MMSE[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with CNN
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_CNN[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_CNN[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with RNN
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_RNN[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_RNN[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with FNN
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_FNN[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_FNN[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

                    # Bit and symbol detection with ELM
                    ThisQamIdx = np.argmin(np.absolute(Const - X_hat_ELM[ii, iii]))
                    ThisBits = list(format(ThisQamIdx, 'b').zfill(m))
                    ThisBits = np.array([int(i) for i in ThisBits])
                    RxBits_ELM[m * ii: m * (ii + 1), iii] = ThisBits[::-1]

            # Accumulate the bit errors for all three receivers
            TotalBerNum_ESN = TotalBerNum_ESN + np.sum(TxBits != RxBits_ESN)
            TotalBerNum_LS = TotalBerNum_LS + np.sum(TxBits != RxBits_LS)
            TotalBerNum_MMSE = TotalBerNum_MMSE + np.sum(TxBits != RxBits_MMSE)
            TotalBerNum_Perfect = TotalBerNum_Perfect + np.sum(TxBits != RxBits_Perfect)
            TotalBerNum_CNN = TotalBerNum_CNN + np.sum(TxBits != RxBits_CNN)
            TotalBerNum_RNN = TotalBerNum_RNN + np.sum(TxBits != RxBits_RNN)
            TotalBerNum_FNN = TotalBerNum_FNN + np.sum(TxBits != RxBits_FNN)
            TotalBerNum_ELM = TotalBerNum_ELM + np.sum(TxBits != RxBits_ELM)
            TotalBerDen = TotalBerDen + NumBitsPerSymbol * N_t

    # Compute and store the bit error rate(BER) values for the current signal to noise ratio.
    BER_ESN[jj] = TotalBerNum_ESN / TotalBerDen
    BER_LS[jj] = TotalBerNum_LS / TotalBerDen
    BER_MMSE[jj] = TotalBerNum_MMSE / TotalBerDen
    BER_Perfect[jj] = TotalBerNum_Perfect / TotalBerDen
    BER_CNN[jj] = TotalBerNum_CNN / TotalBerDen
    BER_RNN[jj] = TotalBerNum_RNN / TotalBerDen
    BER_FNN[jj] = TotalBerNum_FNN / TotalBerDen
    BER_ELM[jj] = TotalBerNum_ELM / TotalBerDen

NMSE_ESN_Testing = NMSE_ESN_Testing / NMSE_count
NMSE_ESN_Training = NMSE_ESN_Training / (NumOfdmSymbols - NMSE_count)

# Plot the BER of all three approaches.


BERvsEBNo = {
"EBN0":EbNoDB,
"BER_ESN": BER_ESN,
"BER_LS": BER_LS,
"BER_MMSE": BER_MMSE,
"BER_Perfect": BER_Perfect,
"BER_CNN": BER_CNN,
"BER_RNN": BER_RNN,
"BER_FNN": BER_FNN,
"BER_ELM": BER_ELM}

f = open('./BERvsEBNo_esn_all_ml_model.pkl','wb')

pickle.dump(BERvsEBNo,f)

f.close()


plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS', linewidth=1.5)
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE', linewidth=1.5)
plt.semilogy(EbNoDB, BER_CNN, 'b-', label='CNN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_RNN, 'c-', label='RNN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_FNN, 'm-', label='FNN', linewidth=1.5)
plt.semilogy(EbNoDB, BER_ELM, 'y-', label='ELM', linewidth=1.5)
plt.semilogy(EbNoDB, BER_Perfect, 'k-', label='Perfect', linewidth=1.5)
plt.legend()
plt.grid(True)
plt.title('100 Neurons in the Reservoir')
plt.xlabel('Signal-to-Noise Ratio[dB]')
plt.ylabel('Bit Error Rate')
plt.savefig('ber_comparison_system_2.png')
plt.show()