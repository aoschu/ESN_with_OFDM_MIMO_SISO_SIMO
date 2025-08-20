import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import tensorflow as tf

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc import LDPC5GEncoder
from sionna.fec.ldpc import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper, Constellation

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber
from sionna.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator, OFDMDemodulator, LMMSEEqualizer
import scipy.interpolate as interp

from sionna.signal import fft, ifft

"""
This code implements the MIMO-OFDM simulation with ESN using Sionna, with LDPC encoding and CDL channel model.
Adapted to use standard Sionna components where possible, with correct shape handling for MIMO.
LDPC parameters: rate 1/2, k=128, n=256.
Pilots uncoded for simplicity.
"""

# Physical parameters
W = 2 * 1.024e6
f_D = 100
No = 0.00001
IsiDuration = 8
CyclicPrefixLen = IsiDuration - 1
EbNoDB = np.arange(25, 31, 5)
N = 64
N_t = 2
N_r = 2
m = 4
m_pilot = 4
NumOfdmSymbols = 1000
Ptotal = 10 ** (EbNoDB / 10) * No * N

# CDL parameters
carrier_frequency = 2.6e9
delay_spread = 300e-9
cdl_model = "B"
direction = "uplink"
min_speed = 10

# Antenna arrays for 2x2 MIMO (dual polarization)
ut_array = AntennaArray(num_rows=1, num_cols=1, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_frequency)
bs_array = AntennaArray(num_rows=1, num_cols=1, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_frequency)

# Stream management
num_streams_per_tx = N_t
rx_tx_association = np.array([[1]])
sm = StreamManagement(rx_tx_association, num_streams_per_tx)

# LDPC parameters
k = 128
n = N * m
encoder = LDPC5GEncoder(k, n)
decoder = LDPC5GDecoder(encoder, num_iter=20, hard_out=True)

# Power Amplifier parameters
p_smooth = 1
ClipLeveldB = 3

# Secondary parameters
Subcarrier_Spacing = W / N
tau_c = 0.5 / f_D
T_OFDM_Total = (N + CyclicPrefixLen) / W
L = np.floor(tau_c / T_OFDM_Total).astype(int)
Pi = Ptotal / N
NumInfoBitsPerSymbol = k * num_streams_per_tx

# Constellations
constellation = Constellation("qam", m)
constellation_pilot = Constellation("qam", m_pilot)

# Sionna setup
binary_source = BinarySource()
mapper = Mapper(constellation=constellation)
mapper_pilot = Mapper(constellation=constellation_pilot)
demapper = Demapper("app", constellation=constellation)

rg = ResourceGrid(num_ofdm_symbols=1, fft_size=N, subcarrier_spacing=Subcarrier_Spacing, num_tx=1, num_streams_per_tx=num_streams_per_tx, cyclic_prefix_length=CyclicPrefixLen, pilot_pattern="empty", dc_null=False)

rg_mapper = ResourceGridMapper(rg)
rg_mapper_pilot = ResourceGridMapper(rg)

ofdm_mod = OFDMModulator(rg.cyclic_prefix_length)
ofdm_demod = OFDMDemodulator(rg.fft_size, -rg.cyclic_prefix_length, rg.cyclic_prefix_length)

lmmse_equ = LMMSEEqualizer(rg, sm)

# PA nonlinearity
def pa_nonlinearity(x, clip_level_db, p_smooth=1):
    a_clip = tf.math.sqrt(tf.math.reduce_variance(x)) * 10 ** (clip_level_db / 20)
    abs_x = tf.abs(x)
    return x / tf.pow(1 + tf.pow(abs_x / a_clip, 2 * p_smooth), 1 / (2 * p_smooth))

# Custom ESN layer
class ESNReservoir(tf.keras.layers.Layer):
    def __init__(self, units, spectral_radius=0.9, connectivity=1.0, leaky=1.0, activation='tanh', input_scaling=0.005, bias_scaling=0.0):
        super().__init__()
        self.units = units
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.leaky = leaky
        self.activation = tf.keras.activations.get(activation)
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

    def build(self, input_shape):
        _, _, input_dim = input_shape
        self.W_in = self.add_weight("W_in", shape=(input_dim, self.units), initializer="uniform", trainable=False)
        self.W_in.assign(tf.random.uniform((input_dim, self.units), -self.input_scaling, self.input_scaling))
        self.b = self.add_weight("b", shape=(self.units,), initializer="uniform", trainable=False)
        self.b.assign(tf.random.uniform((self.units,), -self.bias_scaling, self.bias_scaling))
        W_rec = tf.random.normal((self.units, self.units))
        if self.connectivity < 1:
            mask = tf.random.uniform((self.units, self.units)) < self.connectivity
            W_rec = W_rec * tf.cast(mask, W_rec.dtype)
        rho = tf.reduce_max(tf.abs(tf.linalg.eigvals(W_rec)))
        W_rec *= self.spectral_radius / rho
        self.W_rec = self.add_weight("W_rec", shape=(self.units, self.units), trainable=False)
        self.W_rec.assign(W_rec)

    def call(self, inputs):
        batch_size, time_steps, _ = tf.shape(inputs)
        state = tf.zeros((batch_size, self.units))
        states = []
        for t in tf.range(time_steps):
            update = self.activation(tf.matmul(inputs[:, t, :], self.W_in) + tf.matmul(state, self.W_rec) + self.b)
            state = (1 - self.leaky) * state + self.leaky * update
            states.append(state)
        return tf.stack(states, axis=1)

# ESN params
nInternalUnits = 64
spectralRadius = 0.9
leaky = 1.0
connectivity = min(0.2 * nInternalUnits, 1) / nInternalUnits if nInternalUnits > 0 else 1.0
input_scaling = 0.005
bias_scaling = 0.0
teacher_scaling = 5e-7

# Simulation
BER_ESN = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_LMMSE = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))

NMSE_ESN_Training = np.zeros(len(EbNoDB))
NMSE_ESN_Testing = np.zeros(len(EbNoDB))

# Instantiate CDL
cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=min_speed)

for jj, ebno_db in enumerate(EbNoDB):
    print(f"EbNoDB = {ebno_db}")
    no = ebnodb2no(ebno_db, m, k/n, rg)

    var_x = 10 ** (ebno_db / 10) * No * N
    input_scaling_snr = input_scaling / np.sqrt(var_x)

    total_ber_esn = 0
    total_ber_ls = 0
    total_ber_lmmse = 0
    total_ber_perfect = 0
    total_bits = 0
    nmse_count = 0
    nmse_train = 0
    nmse_test = 0

    esn = ESNReservoir(units=nInternalUnits, spectral_radius=spectralRadius, connectivity=connectivity, leaky=leaky, input_scaling=input_scaling_snr, bias_scaling=bias_scaling)

    temp = CyclicPrefixLen / 9
    IsiMagnitude = np.exp(-np.arange(IsiDuration) / temp)
    IsiMagnitude /= np.sum(IsiMagnitude)
    r_h = tf.linalg.diag(IsiMagnitude.astype(np.float32))
    mmse_scaler = No / Pi[jj]
    mmse_bold_td = tf.linalg.inv(r_h + mmse_scaler * tf.eye(IsiDuration))

    l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
    l_tot = l_max - l_min + 1

    time_channel = ApplyTimeChannel(rg.num_time_samples, l_tot, add_awgn=False)

    for kk in range(1, NumOfdmSymbols + 1):
        # Generate CIR with CDL
        a, tau = cdl(batch_size=1, num_time_steps=rg.num_time_samples + l_tot - 1, sampling_frequency=rg.bandwidth)

        h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)

        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

        if kk % L == 1:  # Pilot (uncoded)
            # --- Generate Pilot Symbols ---
            bits_pilot = binary_source([1, num_streams_per_tx, N * m_pilot])
            s_pilot = mapper_pilot(bits_pilot)  # [1, num_streams_per_tx, N]

            # --- Create Alternating Mask ---
            # s_pilot is shape [1, 2, N] for 2x2 MIMO

            # Generate even/odd mask for subcarrier index (N-length)
            mask_even = tf.cast(tf.reshape(tf.cast(np.arange(N) % 2 == 0, tf.float32), [1, N, 1]), tf.complex64)  # [1, N, 1]
            mask_odd = tf.cast(1.0 - mask_even, tf.complex64)  # [1, N, 1]

            # Stack for two streams (stream 0 gets even, stream 1 gets odd)
            # Result: [1, N, 2]
            mask_combined = tf.concat([mask_even, mask_odd], axis=-1)

            # Transpose to match s_pilot shape: [1, 2, N]
            mask_combined = tf.transpose(mask_combined, [0, 2, 1])

            # --- Apply Mask ---
            s_pilot_ls = s_pilot * mask_combined  # masked pilot for LS estimation

            # --- Map to Resource Grid and Scale Power ---
            x_freq = rg_mapper(s_pilot) * tf.sqrt(Pi[jj])
            x_freq_ls = rg_mapper(s_pilot_ls) * tf.sqrt(Pi[jj])


            x_time = ofdm_mod(x_freq)
            x_time_ls = ofdm_mod(x_freq_ls)

            x_time_nld = pa_nonlinearity(x_time, ClipLeveldB, p_smooth)

            y_time_ls = time_channel([x_time_ls, h_time])
            y_time_nld = time_channel([x_time_nld, h_time])

            noise = complex_normal(y_time_ls.shape) * np.sqrt(No / 2)
            y_time_ls += noise
            y_time_nld += noise

            y_freq_ls = ofdm_demod(y_time_ls)

            # Custom LS and LMMSE estimation
            h_ls = tf.zeros([1, 1, N_r, 1, N_t, N, 1], dtype=tf.complex64)
            h_lmmse = tf.zeros_like(h_ls)
            err_var_ls = no
            err_var_lmmse = no * 0.1

            for nnn in range(N_r):
                for mmm in range(N_t):
                    pilot_indices = np.arange(mmm, N, N_t)
                    h_ls_pilots = y_freq_ls[0, 0, nnn, pilot_indices, 0] / x_freq_ls[0, 0, mmm, pilot_indices, 0]

                    # LS interpolation
                    h_ls_pilots_np = h_ls_pilots.numpy()
                    if mmm == 0:
                        x_interp = np.append(pilot_indices, N - 1)
                        y_interp = np.append(h_ls_pilots_np, h_ls_pilots_np[-1])
                    else:
                        x_interp = np.append(0, pilot_indices)
                        y_interp = np.append(h_ls_pilots_np[0], h_ls_pilots_np)
                    tmpf = interp.interp1d(x_interp, y_interp, kind='linear')
                    h_ls_np = tmpf(np.arange(N))
                    h_ls[0, 0, nnn, 0, mmm, :, 0] = tf.convert_to_tensor(h_ls_np, dtype=tf.complex64)

                    # LMMSE
                    c_ls = ifft(h_ls_pilots)[:IsiDuration]
                    c_mmse = tf.linalg.solve(mmse_bold_td, c_ls[:, tf.newaxis])[:, 0]
                    c_mmse_pad = tf.pad(c_mmse, [[0, N - IsiDuration]])
                    h_lmmse[0, 0, nnn, 0, mmm, :, 0] = fft(c_mmse_pad)

            # ESN input/target
            y_real = tf.real(y_time_nld[0, 0, 0, :])
            y_imag = tf.imag(y_time_nld[0, 0, 0, :])
            y_real2 = tf.real(y_time_nld[0, 0, 1, :])
            y_imag2 = tf.imag(y_time_nld[0, 0, 1, :])
            esn_input = tf.stack([y_real, y_imag, y_real2, y_imag2], axis=-1)[tf.newaxis, :, :]

            x_real = tf.real(x_time[0, 0, 0, :])
            x_imag = tf.imag(x_time[0, 0, 0, :])
            x_real2 = tf.real(x_time[0, 0, 1, :])
            x_imag2 = tf.imag(x_time[0, 0, 1, :])
            esn_target = tf.stack([x_real, x_imag, x_real2, x_imag2], axis=-1)[tf.newaxis, :, :]

            states = esn(esn_input)

            nForgetPoints = 0
            states_train = states[:, nForgetPoints:, :]
            target_train = esn_target[:, nForgetPoints:, :]

            states_flat = tf.reshape(states_train, [-1, nInternalUnits])
            target_flat = tf.reshape(target_train, [-1, 4])

            reg = teacher_scaling * tf.eye(nInternalUnits)
            states_reg = tf.matmul(tf.transpose(states_flat), states_flat) + reg
            states_target = tf.matmul(tf.transpose(states_flat), target_flat)
            w_out = tf.transpose(tf.linalg.solve(states_reg, states_target))

            pred_train = tf.matmul(states_flat, w_out)
            nmse_train += tf.reduce_mean(tf.norm(target_flat - pred_train, axis=0)**2 / tf.norm(target_flat, axis=0)**2)

        else:  # Data symbol (coded)
            u = binary_source([1, num_streams_per_tx, k])

            bits = encoder(u)

            s = mapper(bits)

            x_freq = rg_mapper(s) * tf.sqrt(Pi[jj])

            x_time = ofdm_mod(x_freq)

            x_time_nld = pa_nonlinearity(x_time, ClipLeveldB, p_smooth)

            y_time_nld = time_channel([x_time_nld, h_time])
            noise = complex_normal(y_time_nld.shape) * np.sqrt(No / 2)
            y_time_nld += noise

            y_freq = ofdm_demod(y_time_nld)

            # Baselines using LMMSEEqualizer
            x_hat_perfect, no_post_perfect = lmmse_equ([y_freq, h_freq, no])
            llr_perfect = demapper([x_hat_perfect, no_post_perfect])

            x_hat_ls, no_post_ls = lmmse_equ([y_freq, h_ls, no])
            llr_ls = demapper([x_hat_ls, no_post_ls])

            x_hat_lmmse, no_post_lmmse = lmmse_equ([y_freq, h_lmmse, no])
            llr_lmmse = demapper([x_hat_lmmse, no_post_lmmse])

            u_hat_perfect = decoder(llr_perfect)
            u_hat_ls = decoder(llr_ls)
            u_hat_lmmse = decoder(llr_lmmse)

            # ESN
            y_real = tf.real(y_time_nld[0, 0, 0, :])
            y_imag = tf.imag(y_time_nld[0, 0, 0, :])
            y_real2 = tf.real(y_time_nld[0, 0, 1, :])
            y_imag2 = tf.imag(y_time_nld[0, 0, 1, :])
            esn_input = tf.stack([y_real, y_imag, y_real2, y_imag2], axis=-1)[tf.newaxis, :, :]

            states = esn(esn_input)

            states_test = states[:, nForgetPoints:, :]

            pred_flat = tf.matmul(tf.reshape(states_test, [-1, nInternalUnits]), w_out)

            pred = tf.reshape(pred_flat, [1, -1, 4])

            x_hat_real = pred[0, :, 0] + 1j * pred[0, :, 1]
            x_hat_real2 = pred[0, :, 2] + 1j * pred[0, :, 3]

            x_hat_time_tmp = tf.stack([x_hat_real, x_hat_real2], axis=0)[tf.newaxis, tf.newaxis, :, :]

            x_hat_time = tf.transpose(x_hat_time_tmp, [0, 1, 3, 2])  # [1,1,time,N_t] -> transpose to [1,1,N_t,time]

            x_target = x_time[ :, 0, :, :]

            nmse_test += tf.reduce_mean(tf.norm(x_hat_time - x_target, axis=(2,3))**2 / tf.norm(x_target, axis=(2,3))**2)
            nmse_count += 1

            x_hat_no_cp = x_hat_time[:, :, :, CyclicPrefixLen:]
            x_hat_freq = tf.signal.fft(x_hat_no_cp, axis=3) / tf.sqrt(tf.cast(N, tf.complex64))
            x_hat_freq = x_hat_freq / tf.sqrt(Pi[jj])
            x_hat_freq = tf.expand_dims(x_hat_freq, -1)  # [1,1,N_t,N,1]

            # Demap per stream with small noise variance
            llr_esn_list = []
            small_no = 1e-10
            for stream in range(N_t):
                y_stream = x_hat_freq[0, 0, stream, :, 0][tf.newaxis, :, tf.newaxis, tf.newaxis]
                llr_stream = demapper([y_stream, small_no])
                llr_esn_list.append(llr_stream)
            llr_esn = tf.concat(llr_esn_list, axis=1)

            u_hat_esn = decoder(llr_esn)

            # BER
            ber_esn = tf.reduce_mean(tf.cast(u != u_hat_esn, tf.float32))
            ber_ls = tf.reduce_mean(tf.cast(u != u_hat_ls, tf.float32))
            ber_lmmse = tf.reduce_mean(tf.cast(u != u_hat_lmmse, tf.float32))
            ber_perfect = tf.reduce_mean(tf.cast(u != u_hat_perfect, tf.float32))

            total_ber_esn += ber_esn * NumInfoBitsPerSymbol
            total_ber_ls += ber_ls * NumInfoBitsPerSymbol
            total_ber_lmmse += ber_lmmse * NumInfoBitsPerSymbol
            total_ber_perfect += ber_perfect * NumInfoBitsPerSymbol
            total_bits += NumInfoBitsPerSymbol

    BER_ESN[jj] = total_ber_esn / total_bits
    BER_LS[jj] = total_ber_ls / total_bits
    BER_LMMSE[jj] = total_ber_lmmse / total_bits
    BER_Perfect[jj] = total_ber_perfect / total_bits

    NMSE_ESN_Training[jj] = nmse_train / (NumOfdmSymbols / L)
    NMSE_ESN_Testing[jj] = nmse_test / nmse_count

# Plot and save
import matplotlib.pyplot as plt
plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS')
plt.semilogy(EbNoDB, BER_ESN, 'gd--', label='ESN')
plt.semilogy(EbNoDB, BER_LMMSE, 'rs-.', label='LMMSE')
plt.semilogy(EbNoDB, BER_Perfect, 'b*:', label='Perfect')
plt.legend()
plt.grid(True)
plt.title('BER vs SNR - ESN vs Baselines with LDPC and CDL')
plt.xlabel('Eb/N0 [dB]')
plt.ylabel('Bit Error Rate (BER)')
plt.tight_layout()
plt.savefig("ber_vs_snr_esn_sionna_ldpc_cdl.png", dpi=300)
plt.show()

import pickle
with open('./BERvsEBNo_esn_sionna_ldpc_cdl.pkl', 'wb') as f:
    pickle.dump({'EBN0': EbNoDB, 'BER_ESN': BER_ESN}, f)

import csv
with open('ber_results_sionna_ldpc_cdl.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["EbNoDB (dB)", "BER_ESN", "BER_LS", "BER_LMMSE", "BER_Perfect"])
    for i in range(len(EbNoDB)):
        writer.writerow([EbNoDB[i], BER_ESN[i], BER_LS[i], BER_LMMSE[i], BER_Perfect[i]])