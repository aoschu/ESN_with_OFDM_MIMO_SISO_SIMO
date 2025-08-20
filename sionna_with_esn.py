import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.mapping import Mapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber
import math
import matplotlib.pyplot as plt
import pickle

# Copy of pyESN.py content as class
class ESN():
    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=np.identity, inverse_out_activation=np.identity,
                 random_state=None, silent=True):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)
        self.feedback_scaling = correct_dimensions(feedback_scaling, n_outputs)
        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            self.random_state_ = np.random.RandomState(random_state)
        else:
            self.random_state_ = np.random.mtrand._rand
        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / radius)
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_feedb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        if self.feedback_scaling is not None:
            self.W_feedb *= self.feedback_scaling.reshape(1, -1)  # Added to match script intent

    def _update(self, state, input_pattern, output_pattern):
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern) + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state) + np.dot(self.W_in, input_pattern))
        return np.tanh(preactivation) + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5)

    def _scale_inputs(self, inputs):
        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, transient=0, inspect=False):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))
        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :], teachers_scaled[n - 1, :])
        extended_states = np.hstack((states, inputs_scaled))
        self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]), self.inverse_out_activation(teachers_scaled[transient:, :])).T
        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]
        pred_train = self._unscale_teacher(self.out_activation(np.dot(extended_states, self.W_out.T)))
        return pred_train

    def predict(self, inputs, transient=0, continuation=True):
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]
        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)
        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])
        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out, np.concatenate([states[n + 1, :], inputs[n + 1, :]])))
        return self._unscale_teacher(self.out_activation(outputs[1 + transient:]))

# Copy of HelpFunc.py content
class HelpFunc:
    @staticmethod
    def UnitQamConstellation(Bi):
        EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2
        PamM = EvenSquareRoot
        PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32)
        PamConstellation = np.reshape(PamConstellation, (1, -1))
        SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
        C = SquareMatrix + 1j * (SquareMatrix.T)
        C_tmp = np.zeros(C.shape[0]*C.shape[1], dtype=complex)
        for i in range(C.shape[1]):
            for j in range(C.shape[0]):
                C_tmp[i*C.shape[0] + j] = C[j][i]
        C = C_tmp
        return C / math.sqrt(np.mean(np.abs(C) ** 2))

    @staticmethod
    def trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP):
        if (DelayFlag):
            Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay) ** 4, 4)).astype(np.int32)
            count = 0
            temp = np.zeros((Delay_LUT.shape[0], 1))
            for ii in range(Min_Delay, Max_Delay + 1):
                for jj in range(Min_Delay, Max_Delay + 1):
                    for kk in range(Min_Delay, Max_Delay + 1):
                        for ll in range(Min_Delay, Max_Delay + 1):
                            Delay_LUT[count, :] = np.array([ii, jj, kk, ll])
                            if (abs(ii - jj) > 2 or abs(kk - ll) > 2 or abs(ii - kk) > 2 or abs(ii - ll) > 2 or abs(jj - kk) > 2 or abs(jj - ll) > 2):
                                temp[count] = 1
                            count += 1
            Delay_LUT = Delay_LUT[temp[:,0] == 0]
        else:
            Delay_LUT = np.zeros((Max_Delay - Min_Delay + 1, 4)).astype(np.int32)
            for jjjj in range(Min_Delay, Max_Delay + 1):
                Delay_LUT[jjjj - Min_Delay, :] = jjjj * np.ones(4, dtype=np.int32)
        Delay_Max = np.max(Delay_LUT, axis=1)
        Delay_Min = np.min(Delay_LUT, axis=1)
        NMSE_ESN_Training = np.zeros(Delay_LUT.shape[0])
        for jjj in range(Delay_LUT.shape[0]):
            Curr_Delay = Delay_LUT[jjj, :]
            ESN_input = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))
            ESN_output = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))
            ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[jjj]))
            ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[jjj]))
            ESN_output[Curr_Delay[0]: Curr_Delay[0] + N + CyclicPrefixLen, 0] = x_CP[:, 0].real
            ESN_output[Curr_Delay[1]: Curr_Delay[1] + N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
            ESN_output[Curr_Delay[2]: Curr_Delay[2] + N + CyclicPrefixLen, 2] = x_CP[:, 1].real
            ESN_output[Curr_Delay[3]: Curr_Delay[3] + N + CyclicPrefixLen, 3] = x_CP[:, 1].imag
            nForgetPoints = Delay_Min[jjj] + CyclicPrefixLen
            esn.fit(ESN_input, ESN_output, nForgetPoints)
            x_hat_ESN_temp = esn.predict(ESN_input, nForgetPoints, continuation=False)
            x_hat_ESN_0 = x_hat_ESN_temp[Curr_Delay[0] - Delay_Min[jjj]: Curr_Delay[0] - Delay_Min[jjj] + N, 0] + 1j * x_hat_ESN_temp[Curr_Delay[1] - Delay_Min[jjj]: Curr_Delay[1] - Delay_Min[jjj] + N, 1]
            x_hat_ESN_1 = x_hat_ESN_temp[Curr_Delay[2] - Delay_Min[jjj]: Curr_Delay[2] - Delay_Min[jjj] + N, 2] + 1j * x_hat_ESN_temp[Curr_Delay[3] - Delay_Min[jjj]: Curr_Delay[3] - Delay_Min[jjj] + N, 3]
            x_hat_ESN = np.column_stack((x_hat_ESN_0, x_hat_ESN_1))
            x = x_CP[IsiDuration - 1:, :]
            NMSE_ESN_Training[jjj] = np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0]) ** 2 / np.linalg.norm(x[:, 0]) ** 2 + np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1]) ** 2 / np.linalg.norm(x[:, 1]) ** 2
        Delay_Idx = np.argmin(NMSE_ESN_Training)
        NMSE_ESN = np.min(NMSE_ESN_Training)
        Delay = Delay_LUT[Delay_Idx, :]
        ESN_input = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))
        ESN_output = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))
        ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[Delay_Idx]))
        ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[Delay_Idx]))
        ESN_output[Delay[0]: Delay[0] + N + CyclicPrefixLen, 0] = x_CP[:, 0].real
        ESN_output[Delay[1]: Delay[1] + N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
        ESN_output[Delay[2]: Delay[2] + N + CyclicPrefixLen, 2] = x_CP[:, 1].real
        ESN_output[Delay[3]: Delay[3] + N + CyclicPrefixLen, 3] = x_CP[:, 1].imag
        nForgetPoints = Delay_Min[Delay_Idx] + CyclicPrefixLen
        esn.fit(ESN_input, ESN_output, nForgetPoints)
        return [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Min[Delay_Idx], Delay_Max[Delay_Idx], nForgetPoints, NMSE_ESN]

def correct_dimensions(s, targetlength):
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s

# Physical parameters from original
W = 2 * 1.024e6
f_D = 100
No = 0.00001
IsiDuration = 8
N = 512
Subcarrier_Spacing = W / N
m = 4
m_pilot = 4
NumOfdmSymbols = 400
EbNoDB = np.arange(0, 31, 3)
Ptotal = 10**(EbNoDB / 10) * No * N
p_smooth = 1
ClipLeveldB = 3
N_t = 2
N_r = 2
CyclicPrefixLen = IsiDuration - 1
temp = CyclicPrefixLen / 9
IsiMagnitude = np.exp(-np.arange(CyclicPrefixLen + 1) / temp)
IsiMagnitude = IsiMagnitude / np.sum(IsiMagnitude)
var_x = np.power(10, EbNoDB / 10) * No * N
nInputUnits = N_r * 2
nOutputUnits = N_t * 2
nInternalUnits = 100
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.0000005 * np.ones(N_t * 2)
spectralRadius = 0.9
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration / 2) + 2
DelayFlag = 0

# Sionna setup
binary_source = BinarySource()
constellation = sn.mapping.Constellation("qam", m)
mapper = Mapper(constellation=constellation)
const = constellation.points.numpy()  # for numpy demap

# For ESN bit detection in numpy
def esn_bit_hat(y_pilot_nld, x_pilot_cp, y_data_nld, b_data, ebno_db):
    batch_size = y_pilot_nld.shape[0]
    b_hat = np.zeros(b_data.shape, dtype=np.float32)
    pi = 10**(ebno_db / 10) * No * N
    for i in range(batch_size):
        y_p = np.transpose(y_pilot_nld[i])  # (time, N_r)
        x_p = np.transpose(x_pilot_cp[i])  # (time, N_t)
        y_d = np.transpose(y_data_nld[i])
        var_x_i = 10**(ebno_db / 10) * No * N
        input_scaling = inputScaler / np.sqrt(var_x_i) * np.ones(nInputUnits)
        input_shift = inputOffset / inputScaler * np.ones(nInputUnits)
        teacher_scaling = teacherScaling
        teacher_shift = np.zeros(nOutputUnits)
        feedback_scaling = feedbackScaler * np.ones(nOutputUnits)
        esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                  spectral_radius=spectralRadius, sparsity=1 - min(0.2, 1),
                  input_shift=input_shift, input_scaling=input_scaling,
                  teacher_scaling=teacher_scaling, teacher_shift=teacher_shift,
                  feedback_scaling=feedback_scaling, silent=True)
        [_, _, trained_esn, delay, _, delay_min, delay_max, n_forget_points, _] = HelpFunc.trainMIMOESN(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_p, x_p)
        esn_input_test = np.zeros((N + delay_max + CyclicPrefixLen, nInputUnits))
        esn_input_test[:, 0] = np.append(y_d[:, 0].real, np.zeros(delay_max))
        esn_input_test[:, 1] = np.append(y_d[:, 0].imag, np.zeros(delay_max))
        esn_input_test[:, 2] = np.append(y_d[:, 1].real, np.zeros(delay_max))
        esn_input_test[:, 3] = np.append(y_d[:, 1].imag, np.zeros(delay_max))
        x_hat_temp = trained_esn.predict(esn_input_test, n_forget_points)
        x_hat_esn_0 = x_hat_temp[delay[0] - delay_min : delay[0] - delay_min + N, 0] + 1j * x_hat_temp[delay[1] - delay_min : delay[1] - delay_min + N, 1]
        x_hat_esn_1 = x_hat_temp[delay[2] - delay_min : delay[2] - delay_min + N, 2] + 1j * x_hat_temp[delay[3] - delay_min : delay[3] - delay_min + N, 3]
        x_hat_esn = np.column_stack((x_hat_esn_0, x_hat_esn_1))
        X_hat = np.zeros((N, N_t), dtype=complex)
        for ii in range(N_t):
            X_hat[:, ii] = 1 / N * np.fft.fft(x_hat_esn[:, ii]) / np.sqrt(pi)
        Const = HelpFunc.UnitQamConstellation(m)
        RxBits = np.zeros((N * m, N_t), dtype=int)
        for ii in range(N):
            for iii in range(N_t):
                ThisQamIdx = np.argmin(np.abs(Const - X_hat[ii, iii]))
                ThisBits = np.array(list(format(ThisQamIdx, '0{}b'.format(m))), dtype=int)[::-1]
                RxBits[ii * m : (ii + 1) * m, iii] = ThisBits
        b_hat[i] = RxBits.T
    return b_hat

# For LS bit hat in numpy (similar for MMSE, perfect)
def ls_bit_hat(y_data_nld, x_ls, y_ls, b_data, ebno_db):
    batch_size = y_data_nld.shape[0]
    b_hat = np.zeros(b_data.shape, dtype=np.float32)
    pi = 10**(ebno_db / 10) * No * N
    mmse_scaler = No / pi
    r_h = np.diag(IsiMagnitude)
    mmse_bold_td = np.linalg.inv(r_h) * mmse_scaler / (N / 2) + np.eye(IsiDuration)
    for i in range(batch_size):
        y_d = np.transpose(y_data_nld[i])
        Y = np.zeros((N, N_r), dtype=complex)
        for r in range(N_r):
            Y[:, r] = 1 / N * np.fft.fft(y_d[:, r][CyclicPrefixLen:]) 
        # LS est from y_ls
        y_l = np.transpose(y_ls[i])
        Y_l = np.zeros((N, N_r), dtype=complex)
        for r in range(N_r):
            Y_l[:, r] = 1 / N * np.fft.fft(y_l[:, r][CyclicPrefixLen:]) / np.sqrt(pi)
        Ci_LS_Pilots = np.zeros((N_r, N_t, N // N_t), dtype=complex)
        for r in range(N_r):
            for t in range(N_t):
                idx = np.arange(t, N, N_t)
                Ci_LS_Pilots[r, t, :] = Y_l[idx, r] / x_ls[i, t, idx]
        Ci_LS = np.zeros((N_r, N_t, N), dtype=complex)
        for r in range(N_r):
            for t in range(N_t):
                if t == 0:
                    x_points = np.append(np.arange(t, N, N_t), N-1)
                    vals = np.append(Ci_LS_Pilots[r, t, :], Ci_LS_Pilots[r, t, -1])
                else:
                    x_points = np.append(0, np.arange(t, N, N_t))
                    vals = np.append(Ci_LS_Pilots[r, t, 0], Ci_LS_Pilots[r, t, :])
                f = np.interp(np.arange(N), x_points, vals.real) + 1j * np.interp(np.arange(N), x_points, vals.imag)
                Ci_LS[r, t, :] = f
        # for detection
        X_hat = np.zeros((N, N_t), dtype=complex)
        for ii in range(N):
            H_temp = Ci_LS[:, :, ii]
            Y_temp = Y[ii, :]
            X_hat[ii, :] = np.linalg.solve(H_temp, Y_temp) / np.sqrt(pi)
        Const = HelpFunc.UnitQamConstellation(m)
        RxBits = np.zeros((N * m, N_t), dtype=int)
        for ii in range(N):
            for iii in range(N_t):
                ThisQamIdx = np.argmin(np.abs(Const - X_hat[ii, iii]))
                ThisBits = np.array(list(format(ThisQamIdx, '0{}b'.format(m))), dtype=int)[::-1]
                RxBits[ii * m : (ii + 1) * m, iii] = ThisBits
        b_hat[i] = RxBits.T
    return b_hat

# Model
class MIMOOFDMModel(tf.keras.Model):
    def __init__(self, mode='esn'):
        super().__init__()
        self.mode = mode
        self.binary_source = BinarySource()
        self.mapper = Mapper(constellation=constellation)

    def nonlinear_pa(self, x_time):
        a_clip = tf.sqrt(10**(self.ebno_db / 10) * No * N) * 10**(ClipLeveldB / 20)
        den = tf.pow(1 + tf.pow(tf.abs(x_time) / a_clip, 2 * p_smooth), 1 / (2 * p_smooth))
        return x_time / tf.cast(den, tf.complex64)

    def apply_channel(self, x_time):
        batch_size = tf.shape(x_time)[0]
        def conv_per_batch(args):
            x, h = args
            y = tf.zeros([N_r, N + CyclicPrefixLen], dtype=tf.complex64)
            for r in range(N_r):
                for t in range(N_t):
                    h_flip = tf.reverse(h[r, t, :], axis=[0])[tf.newaxis, :, tf.newaxis]
                    x_t = x[t, :][tf.newaxis, :, tf.newaxis]
                    pad = tf.zeros([IsiDuration - 1], dtype=tf.complex64)[tf.newaxis, :, tf.newaxis]
                    x_pad = tf.concat([pad, x_t], axis=1)
                    conv = tf.nn.conv1d(x_pad, h_flip, stride=1, padding='VALID')
                    y = tf.tensor_scatter_nd_add(y, [[r]], conv[0, :, 0][tf.newaxis, :])
            return y
        y_time = tf.map_fn(conv_per_batch, (x_time, self.c), fn_output_signature=tf.TensorSpec(shape=[N_r, N + CyclicPrefixLen], dtype=tf.complex64))
        return y_time

    def call(self, batch_size, ebno_db):
        self.ebno_db = ebno_db
        # Pilot
        b_pilot = self.binary_source([batch_size, N_t, N * m_pilot])
        x_pilot = tf.reshape(self.mapper(tf.reshape(b_pilot, [batch_size * N_t, N * m_pilot])), [batch_size, N_t, N])
        x_time = tf.cast(N, tf.complex64) * tf.signal.ifft(x_pilot)
        x_cp = tf.concat([x_time[:, :, -CyclicPrefixLen:], x_time], axis=2)

        pi = tf.pow(10.0, ebno_db / 10) * No * N
        x_cp = x_cp * tf.cast(tf.sqrt(pi), tf.complex64)

        # Data
        b_data = self.binary_source([batch_size, N_t, N * m])
        x_data = tf.reshape(self.mapper(tf.reshape(b_data, [batch_size * N_t, N * m])), [batch_size, N_t, N])
        x_time_data = tf.cast(N, tf.complex64) * tf.signal.ifft(x_data)
        x_cp_data = tf.concat([x_time_data[:, :, -CyclicPrefixLen:], x_time_data], axis=2)
        x_cp_data = x_cp_data * tf.cast(tf.sqrt(pi), tf.complex64)

        # Channel
        self.c = tf.complex(tf.random.normal([batch_size, N_r, N_t, IsiDuration]) / math.sqrt(2), tf.random.normal([batch_size, N_r, N_t, IsiDuration]) / math.sqrt(2))
        self.c = self.c * tf.sqrt(tf.constant(IsiMagnitude, dtype=tf.float32))

        # PA
        x_cp_nld = self.nonlinear_pa(x_cp)
        x_cp_data_nld = self.nonlinear_pa(x_cp_data)

        # Noise
        noise_p = tf.complex(tf.random.normal([batch_size, N_r, N + CyclicPrefixLen]), tf.random.normal([batch_size, N_r, N + CyclicPrefixLen])) * math.sqrt((N + CyclicPrefixLen) * No / 2)
        noise_d = tf.complex(tf.random.normal([batch_size, N_r, N + CyclicPrefixLen]), tf.random.normal([batch_size, N_r, N + CyclicPrefixLen])) * math.sqrt((N + CyclicPrefixLen) * No / 2)

        # Apply channel
        y_pilot_nld = self.apply_channel(x_cp_nld) + noise_p
        y_data_nld = self.apply_channel(x_cp_data_nld) + noise_d

        # For LS
        x_ls = tf.zeros_like(x_pilot)
        num_sub = N // 2
        batch_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, num_sub])
        ant_indices = tf.zeros([batch_size, num_sub], tf.int32)
        sub_indices = tf.tile(tf.range(0, N, 2)[tf.newaxis, :], [batch_size, 1])
        indices = tf.stack([batch_indices, ant_indices, sub_indices], axis=-1)
        indices = tf.reshape(indices, [-1, 3])
        values = tf.reshape(x_pilot[:, 0, 0::2], [-1])
        x_ls = tf.tensor_scatter_nd_update(x_ls, indices, values)

        ant_indices = tf.ones([batch_size, num_sub], tf.int32)
        sub_indices = tf.tile(tf.range(1, N, 2)[tf.newaxis, :], [batch_size, 1])
        indices = tf.stack([batch_indices, ant_indices, sub_indices], axis=-1)
        indices = tf.reshape(indices, [-1, 3])
        values = tf.reshape(x_pilot[:, 1, 1::2], [-1])
        x_ls = tf.tensor_scatter_nd_update(x_ls, indices, values)

        x_time_ls = tf.cast(N, tf.complex64) * tf.signal.ifft(x_ls)
        x_cp_ls = tf.concat([x_time_ls[:, :, -CyclicPrefixLen:], x_time_ls], axis=2) * tf.cast(tf.sqrt(pi), tf.complex64)
        y_cp_ls = self.apply_channel(x_cp_ls) + noise_p
        y_ls = tf.signal.fft(y_cp_ls[:, :, CyclicPrefixLen:]) / tf.cast(N, tf.complex64)

        if self.mode == 'esn':
            b_hat = tf.py_function(esn_bit_hat, [y_pilot_nld, x_cp, y_data_nld, b_data, ebno_db], tf.float32)
            b_hat.set_shape(b_data.shape)
        elif self.mode == 'ls':
            b_hat = tf.py_function(ls_bit_hat, [y_data_nld, x_ls, y_cp_ls, b_data, ebno_db], tf.float32)
            b_hat.set_shape(b_data.shape)
        # Add for 'mmse' and 'perfect' similarly
        else:
            # Placeholder for other modes
            b_hat = tf.zeros_like(b_data)
        return b_data, b_hat

# Run simulation
modes = ['esn', 'ls', 'mmse', 'perfect']  # Implement mmse and perfect similarly
bers = {}
for mode in modes:
    bers[mode] = []
    model = MIMOOFDMModel(mode)
    for ebno in EbNoDB:
        ber, _ = sim_ber(model, ebno_dbs=[ebno], batch_size=64, num_target_bit_errors=1000, max_mc_iter=100)
        bers[mode].append(ber[0])

# Plot
for mode in modes:
    plt.semilogy(EbNoDB, bers[mode], label=mode.upper())
plt.xlabel('Eb/No [dB]')
plt.ylabel('BER')
plt.grid(True)
plt.legend()
plt.show()

# Save
with open('./BERvsEBNo_esn.pkl', 'wb') as f:
    pickle.dump({'EBN0': EbNoDB, 'BER': bers['esn']}, f)