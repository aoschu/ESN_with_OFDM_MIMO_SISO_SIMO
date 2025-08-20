import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import pinv as sc_pinv
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# Parameters (user-defined; change as needed)
N_t = 2  # Number of transmitters
N_r = 2  # Number of receivers
N_c = 32  # Number of subcarriers
N_cp = 8  # Cyclic prefix length
L = 4  # Channel taps
N_u = 100  # Reservoir size increased from 16
mod_order = 4  # Modulation order (4-QAM = QPSK)
bits_per_symbol = int(np.log2(mod_order))
ebn0_db_list = np.arange(0, 21, 2)  # Eb/N0 range extended to 20 dB
num_trials = 10  # Number of trials (small for testing)
N_sym = 14  # Symbols per frame
N_pilot = 8  # Pilot symbols increased from 4

# 4-QAM (QPSK) constellation
const = np.array([ -1-1j, -1+1j, 1-1j, 1+1j ]) / np.sqrt(2)

# Function to find closest symbol (fixed to handle multi-dimensional inputs)
def closest_symbol(s):
    original_shape = s.shape
    s_flat = s.flatten()
    dist = np.abs(const[:, np.newaxis] - s_flat)
    idx = np.argmin(dist, axis=0)
    result = const[idx]
    return result.reshape(original_shape)

# Bit mapping for QPSK
def symbol_to_bits(sym):
    re = np.real(sym) * np.sqrt(2)  # Denormalize
    im = np.imag(sym) * np.sqrt(2)
    re_bit = '0' if re < 0 else '1'
    im_bit = '0' if im < 0 else '1'
    return re_bit + im_bit

def get_bits(symbols):
    bits = ''
    for s in symbols.flatten():
        s_closest = closest_symbol(s)
        bits += symbol_to_bits(s_closest)
    return bits

# Calculate BER
def calculate_ber(pred_symbols, true_symbols):
    true_bits = get_bits(true_symbols)
    pred_bits = get_bits(pred_symbols)
    errors = sum(b1 != b2 for b1, b2 in zip(true_bits, pred_bits))
    return errors / len(true_bits)

# Class for ESN (Reservoir Computing) - Support cascading
class ESN:
    def __init__(self, input_size, reservoir_size):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.W_in = np.random.uniform(-0.5, 0.5, (reservoir_size, input_size))
        W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        mask = np.random.rand(reservoir_size, reservoir_size) < 0.1
        W_res = W_res * mask
        rho = np.max(np.abs(np.linalg.eigvals(W_res)))
        self.W_res = W_res / rho * 0.9 if rho > 0 else W_res
        self.W_out = None

    def get_states(self, y_real_sym):
        N = y_real_sym.shape[0]
        r = np.zeros(self.reservoir_size)
        states = np.zeros((N, self.reservoir_size))
        for t in range(N):
            u = y_real_sym[t]
            r = np.tanh(self.W_res @ r + self.W_in @ u)
            states[t] = r
        return states

    def train(self, y_real, x_real):
        R = []
        for s in range(N_pilot):
            states = self.get_states(y_real[s])
            R.append(states)
        R = np.vstack(R)
        T = []
        for s in range(N_pilot):
            T.append(x_real[s])
        T = np.vstack(T)
        self.W_out = sc_pinv(R) @ T

    def predict(self, y_real_sym, show_states=False):
        states = self.get_states(y_real_sym)
        z = states @ self.W_out
        return z

# Binary Classifier for frequency domain (input size 2 for re+im)
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# PAM Classifier (simplified for QPSK: 1 binary per dimension for sign)
class PamClassifier:
    def __init__(self):
        self.model = BinaryClassifier()  # Only 1 for boundary at 0

    def train(self, z_input_2d, true_levels):
        boundary = 0
        label_bin = (true_levels > boundary).astype(np.float32)
        label_t = torch.tensor(label_bin, dtype=torch.float32).view(-1, 1)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.001)
        loss_fn = nn.BCELoss()
        for epoch in range(2000):  # Increased from 800
            out = self.model(z_input_2d)
            loss = loss_fn(out, label_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_probs(self, z_input_2d):
        p = self.model(z_input_2d).detach().numpy().squeeze()
        return p

# Detection for QPSK (sign decision)
def pam_detect(probs):
    level = (2 * (probs > 0.5).astype(float) - 1) / np.sqrt(2)
    return level

# Simplified MMNet
class MMNet(nn.Module):
    def __init__(self, N_r, N_t):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(N_r * 2, N_t * 2) for _ in range(3)])  # Reduced layers

    def forward(self, y_real):
        x = torch.zeros(y_real.shape[0], N_t * 2)
        for layer in self.layers:
            x = layer(y_real) + x
        return x

# Rapp PA
def rapp_pa(x, p=3, sat=1, gain=1):
    a = np.abs(x)
    return gain * x / (1 + (a / sat)**(2 * p))**(1 / (2 * p))

# Simplified Sphere Decoder (bottom-up)
def sphere_decoder(y_k, h_k, const):
    Q, R = np.linalg.qr(h_k)
    y_tilde = Q.conj().T @ y_k.squeeze()
    m = h_k.shape[1]
    best_x = np.zeros(m, dtype=complex)
    best_dist = np.inf
    
    current_x = np.zeros(m, dtype=complex)
    
    def dfs(level, partial_sum):
        nonlocal best_x, best_dist
        if level == -1:
            if partial_sum < best_dist:
                best_dist = partial_sum
                best_x = current_x.copy()
            return
        r_sum = 0
        for j in range(level + 1, m):
            r_sum += R[level, j] * current_x[j]
        est = (y_tilde[level] - r_sum) / R[level, level]
        sorted_const = sorted(const, key=lambda c: np.abs(c - est))
        for c in sorted_const:
            current_x[level] = c
            inc = np.abs(y_tilde[level] - r_sum - R[level, level] * c)**2
            if partial_sum + inc < best_dist:
                dfs(level - 1, partial_sum + inc)
    
    dfs(m - 1, 0)
    return best_x if best_dist < np.inf else closest_symbol(np.linalg.pinv(h_k) @ y_k).squeeze()

# BER storage
ber_lmmse = np.zeros((3, len(ebn0_db_list)))
ber_sd = np.zeros((3, len(ebn0_db_list)))
ber_mmnet = np.zeros((3, len(ebn0_db_list)))
ber_rc_net = np.zeros((3, len(ebn0_db_list)))
ber_rc_struct = np.zeros((3, len(ebn0_db_list)))

# Main simulation loop
for scen_idx in range(3):  # 0: linear with CSI, 1: linear no CSI, 2: non-linear no CSI
    has_csi = (scen_idx == 0)
    is_nonlinear = (scen_idx == 2)
    for i, ebn0_db in enumerate(ebn0_db_list):
        ber_lmmse_trial = 0
        ber_sd_trial = 0
        ber_mmnet_trial = 0
        ber_net_trial = 0
        ber_struct_trial = 0
        for trial in range(num_trials):
            h = (np.random.randn(N_r, N_t, L) + 1j * np.random.randn(N_r, N_t, L)) / np.sqrt(2 * L)
            H_k = np.zeros((N_c, N_r, N_t), dtype=complex)
            for k in range(N_c):
                for l in range(L):
                    H_k[k] += h[:,:,l] * np.exp(-1j * 2 * np.pi * k * l / N_c)
            E_b_N0 = 10 ** (ebn0_db / 10)
            N0 = 1 / E_b_N0
            sigma = np.sqrt(N0 / 2)
            X = np.random.choice(const, size=(N_sym, N_c, N_t))
            x_time = ifft(X, axis=1) * np.sqrt(N_c)
            if is_nonlinear:
                x_time = rapp_pa(x_time)
            x_cp = np.concatenate((x_time[:, -N_cp:, :], x_time), axis=1)
            y_cp = np.zeros((N_sym, N_c + N_cp + L - 1, N_r), dtype=complex)
            for s in range(N_sym):
                for r in range(N_r):
                    for t in range(N_t):
                        y_cp[s, :, r] += np.convolve(x_cp[s, :, t], h[r, t, :])
            noise = (np.random.randn(*y_cp.shape) + 1j * np.random.randn(*y_cp.shape)) * sigma
            y_cp += noise
            y_time = y_cp[:, N_cp : N_cp + N_c, :]
            y_real = np.concatenate((np.real(y_time), np.imag(y_time)), axis=-1)
            x_real = np.concatenate((np.real(x_time), np.imag(x_time)), axis=-1)
            input_size = 2 * N_r
            # Cascaded ESN: Two instances
            esn1 = ESN(input_size, N_u)
            esn1.train(y_real, x_real)
            # Generate intermediate for pilots
            intermediate = []
            for s in range(N_pilot):
                intermediate.append(esn1.predict(y_real[s]))
            intermediate = np.array(intermediate)
            esn2 = ESN(2 * N_t, N_u)  # Input to second is z from first
            esn2.train(intermediate, x_real)
            data_start = N_pilot
            pred_symbols_struct = np.zeros((N_sym - data_start, N_c, N_t), dtype=complex)
            pred_symbols_net = np.zeros((N_sym - data_start, N_c, N_t), dtype=complex)
            pred_symbols_lmmse = np.zeros((N_sym - data_start, N_c, N_t), dtype=complex)
            pred_symbols_sd = np.zeros((N_sym - data_start, N_c, N_t), dtype=complex)
            pred_symbols_mmnet = np.zeros((N_sym - data_start, N_c, N_t), dtype=complex)
            true_symbols = X[data_start:, :, :]
            z_pilot = np.zeros((N_pilot, N_c, 2 * N_t))
            for s in range(N_pilot):
                z1 = esn1.predict(y_real[s])
                z_pilot[s] = esn2.predict(z1)
            z_pilot_complex = z_pilot[:, :, 0:N_t] + 1j * z_pilot[:, :, N_t:]
            Z_pilot = fft(z_pilot_complex, axis=1) / np.sqrt(N_c)
            z_input_2d = torch.tensor(np.stack((np.real(Z_pilot).flatten(), np.imag(Z_pilot).flatten()), axis=1), dtype=torch.float32)
            x_re = np.real(X[0:N_pilot]).flatten() * np.sqrt(2)
            x_im = np.imag(X[0:N_pilot]).flatten() * np.sqrt(2)
            pam_re = PamClassifier()
            pam_re.train(z_input_2d, x_re)
            pam_im = PamClassifier()
            pam_im.train(z_input_2d, x_im)
            mmnet = MMNet(N_r, N_t)
            optimizer_mm = optim.Adam(mmnet.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            y_pilot_freq = fft(y_time[0:N_pilot], axis=1) / np.sqrt(N_c)
            y_pilot_real = np.concatenate((np.real(y_pilot_freq), np.imag(y_pilot_freq)), axis=-1).reshape(N_pilot * N_c, -1)
            x_pilot_real = np.concatenate((np.real(X[0:N_pilot]), np.imag(X[0:N_pilot])), axis=-1).reshape(N_pilot * N_c, -1)
            y_pilot_t = torch.tensor(y_pilot_real, dtype=torch.float32)
            x_pilot_t = torch.tensor(x_pilot_real, dtype=torch.float32)
            for epoch in range(500):  # Increased from 100
                out = mmnet(y_pilot_t)
                loss = loss_fn(out, x_pilot_t)
                optimizer_mm.zero_grad()
                loss.backward()
                optimizer_mm.step()
            
            # LS Channel estimation for no CSI scenarios
            H_k_est = np.zeros((N_c, N_r, N_t), dtype=complex)
            for k in range(N_c):
                Y_k = y_pilot_freq[:, k, :].T  # N_r x N_pilot
                X_k = X[0:N_pilot, k, :].T  # N_t x N_pilot
                H_k_est[k] = Y_k @ np.linalg.pinv(X_k, rcond=1e-10)
            
            for s in range(data_start, N_sym):
                z1 = esn1.predict(y_real[s])
                z_s = esn2.predict(z1)
                z_complex = z_s[:, 0:N_t] + 1j * z_s[:, N_t:]
                Z = fft(z_complex, axis=0) / np.sqrt(N_c)
                pred_symbols_net[s - data_start] = closest_symbol(Z)
                z_input_data_2d = torch.tensor(np.stack((np.real(Z).flatten(), np.imag(Z).flatten()), axis=1), dtype=torch.float32)
                probs_re = pam_re.predict_probs(z_input_data_2d)
                probs_im = pam_im.predict_probs(z_input_data_2d)
                level_re = pam_detect(probs_re)
                level_im = pam_detect(probs_im)
                pred_symbols_struct[s - data_start] = level_re.reshape(N_c, N_t) + 1j * level_im.reshape(N_c, N_t)
                Y = fft(y_time[s], axis=0) / np.sqrt(N_c)
                h_use = H_k if has_csi else H_k_est
                for k in range(N_c):
                    h = h_use[k]
                    inv = np.linalg.inv(h @ np.conj(h).T + N0 * np.eye(N_r) + 1e-10 * np.eye(N_r))
                    w = np.conj(h).T @ inv
                    y_k = Y[k, :].T
                    x_hat = w @ y_k
                    pred_symbols_lmmse[s - data_start, k, :] = closest_symbol(x_hat)
                for k in range(N_c):
                    y_k = Y[k, :].T[:, np.newaxis]
                    h_k = h_use[k]
                    pred_symbols_sd[s - data_start, k, :] = sphere_decoder(y_k, h_k, const)
                y_real_s = np.concatenate((np.real(Y), np.imag(Y)), axis=-1).reshape(N_c, -1)
                out = mmnet(torch.tensor(y_real_s, dtype=torch.float32)).detach().numpy()
                re = out[:, :N_t]
                im = out[:, N_t:]
                pred_symbols_mmnet[s - data_start] = closest_symbol(re + 1j * im)
            ber_struct = calculate_ber(pred_symbols_struct, true_symbols)
            ber_net = calculate_ber(pred_symbols_net, true_symbols)
            ber_lmmse_val = calculate_ber(pred_symbols_lmmse, true_symbols)
            ber_sd_val = calculate_ber(pred_symbols_sd, true_symbols)
            ber_mmnet_val = calculate_ber(pred_symbols_mmnet, true_symbols)
            ber_struct_trial += ber_struct
            ber_net_trial += ber_net
            ber_lmmse_trial += ber_lmmse_val
            ber_sd_trial += ber_sd_val
            ber_mmnet_trial += ber_mmnet_val
        ber_rc_struct[scen_idx, i] = ber_struct_trial / num_trials
        ber_rc_net[scen_idx, i] = ber_net_trial / num_trials
        ber_lmmse[scen_idx, i] = ber_lmmse_trial / num_trials
        ber_sd[scen_idx, i] = ber_sd_trial / num_trials
        ber_mmnet[scen_idx, i] = ber_mmnet_trial / num_trials
        print(f"Scenario {scen_idx}, Eb/N0 {ebn0_db}: Avg BER RC-Struct {ber_rc_struct[scen_idx, i]}, RCNet {ber_rc_net[scen_idx, i]}, LMMSE {ber_lmmse[scen_idx, i]}, SD {ber_sd[scen_idx, i]}, MMNet {ber_mmnet[scen_idx, i]}")

# Plot BER curves with paper-like colors
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['(a) Linear with CSI', '(b) Linear without CSI', '(c) Non-linear without CSI']
colors = ['blue', 'green', 'orange', 'red', 'purple']  # From paper image
for idx in range(3):
    axs[idx].semilogy(ebn0_db_list, ber_lmmse[idx], 'o-', color=colors[0], label='LMMSE+LMMSE-CSI')
    axs[idx].semilogy(ebn0_db_list, ber_sd[idx], 's-', color=colors[1], label='SD+LMMSE-CSI')
    axs[idx].semilogy(ebn0_db_list, ber_mmnet[idx], '^-', color=colors[2], label='MMNet')
    axs[idx].semilogy(ebn0_db_list, ber_rc_net[idx], 'x-', color=colors[3], label='RCNet')
    axs[idx].semilogy(ebn0_db_list, ber_rc_struct[idx], '*-', color=colors[4], label='RC-Struct')
    axs[idx].set_xlabel('Eb/N0 (dB)')
    axs[idx].set_ylabel('Raw BER')
    axs[idx].set_title(titles[idx])
    axs[idx].grid(True)
    axs[idx].legend()
plt.tight_layout()
plt.show()

# Constellation diagram (for last true vs predicted of RC-Struct in last scenario)
true_const = true_symbols[-1].flatten()
pred_const = pred_symbols_struct[-1].flatten()
plt.scatter(np.real(true_const), np.imag(true_const), label='True Symbols', marker='o')
plt.scatter(np.real(pred_const), np.imag(pred_const), label='Predicted (RC-Struct)', marker='x')
plt.title('Constellation Diagram (True vs Predicted)')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.legend()
plt.grid(True)
plt.show()
print("Constellation diagram plotted")