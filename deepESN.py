import numpy as np
from scipy.linalg import pinv  # For pseudoinverse
import matplotlib.pyplot as plt

# ESN Class based on the paper (with delay support)
class ESN:
    def __init__(self, in_size, res_size, out_size, spectral_radius=0.98, input_scaling=0.5, 
                 feedback_scaling=0.5, leak_rate=0.5, connectivity=0.2, delay=0):
        self.in_size = in_size
        self.res_size = res_size
        self.out_size = out_size
        self.leak_rate = leak_rate
        self.delay = delay  # Output delay per paper Section III-C
        
        # Random input weights
        self.W_in = input_scaling * 2 * (np.random.rand(res_size, in_size) - 0.5)
        
        # Sparse reservoir weights
        W = 2 * (np.random.rand(res_size, res_size) - 0.5)
        mask = np.random.rand(res_size, res_size) < connectivity
        self.W = W * mask
        
        # Scale to spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(self.W)))
        if rho > 0:
            self.W *= spectral_radius / rho
        
        # Feedback weights
        self.W_fb = feedback_scaling * 2 * (np.random.rand(res_size, out_size) - 0.5)
        
        self.W_out = None
        self.x = np.zeros(res_size)
    
    def reset(self):
        self.x = np.zeros(self.res_size)
    
    def train(self, U, Y, washout=100):
        # U: T x in_size, Y: T x out_size
        T = U.shape[0]
        states = []
        x = np.zeros(self.res_size)
        y_prev = np.zeros(self.out_size)  # Initial feedback
        
        for t in range(T):
            preact = np.dot(self.W_in, U[t]) + np.dot(self.W, x) + np.dot(self.W_fb, y_prev)
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(preact)
            y_prev = Y[t]  # Teacher forcing
            if t >= washout:
                ext_state = np.concatenate((U[t], x))
                states.append(ext_state)
        
        if len(states) == 0:
            raise ValueError("No states collected after washout")
        
        S = np.vstack(states[:-self.delay] if self.delay > 0 else states)  # Align to shifted targets
        Tgt = Y[washout + self.delay:] if self.delay > 0 else Y[washout:]  # Shift target by delay
        
        # Least squares: S @ W_out.T = Tgt => W_out.T = pinv(S) @ Tgt
        self.W_out = (pinv(S) @ Tgt).T  # out x (in + res)
    
    def predict(self, U):
        # U: T x in_size
        T = U.shape[0]
        Y_hat = np.zeros((T + self.delay, self.out_size))  # Pad for delay
        y_prev = np.zeros(self.out_size)
        
        for t in range(T):
            preact = np.dot(self.W_in, U[t]) + np.dot(self.W, self.x) + np.dot(self.W_fb, y_prev)
            self.x = (1 - self.leak_rate) * self.x + self.leak_rate * np.tanh(preact)
            ext_state = np.concatenate((U[t], self.x))
            y_hat = np.dot(self.W_out, ext_state)
            Y_hat[t + self.delay] = y_hat  # Shift by delay
            y_prev = y_hat  # Feedback during prediction
        
        return Y_hat[self.delay:self.delay + T]  # Return shifted output

# Rapp model for nonlinear PA
def rapp_pa(s_time, ibo_db=3.0, p=1):
    # IBO in dB, p=shape
    A_sat = 10**(ibo_db / 20.0)  # Assuming E[|s|^2] = 1 after normalization
    abs_s = np.abs(s_time)
    g = abs_s / (1 + (abs_s / A_sat)**(2*p))**(1/(2*p))
    u = g * (s_time / (abs_s + 1e-10))  # Avoid division by zero
    return u

# Generate QPSK symbols (frequency domain)
def generate_qpsk_symbols(N):
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    idx = np.random.randint(0, 4, N)
    S_freq = const[idx]
    return S_freq

# Compute BER for detected symbols
def compute_ber(S_freq_true, S_freq_hat):
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    dist = np.abs(S_freq_hat[:, None] - const[None, :])
    detected = np.argmin(dist, axis=1)
    
    bits_true = np.zeros(2 * len(S_freq_true), dtype=int)
    bits_hat = np.zeros(2 * len(S_freq_true), dtype=int)
    
    for i, c in enumerate(const):
        mask = (S_freq_true == c)
        bits_true[2*mask.nonzero()[0]] = (i >> 1) & 1
        bits_true[2*mask.nonzero()[0] + 1] = i & 1
    
    for j in range(len(detected)):
        bits_hat[2*j] = (detected[j] >> 1) & 1
        bits_hat[2*j + 1] = detected[j] & 1
    
    errors = np.sum(bits_true != bits_hat)
    return errors / len(bits_true)

# LS Channel Estimation (Frequency Domain)
def ls_channel_est(Y_freq, S_freq_train):
    H_est = Y_freq / (S_freq_train + 1e-10)  # Element-wise
    return H_est

# LMMSE Channel Estimation (improved)
def lmmse_channel_est(Y_freq, S_freq_train, snr_lin, R_hh):
    H_ls = ls_channel_est(Y_freq, S_freq_train)
    inv_term = np.linalg.inv(R_hh + (1/snr_lin) * np.eye(len(R_hh)))
    W = inv_term @ R_hh
    H_est = W @ H_ls
    return H_est

# Simulation parameters (simplified for SISO)
N = 512  # subcarriers
CP_len = 128  # CP length
M = 10  # Channel taps (delay spread < CP)
snr_dbs = np.arange(0, 31, 5)  # SNR range
num_symbols_per_block = 19  # Block fading duration
num_blocks = 1000  # Increased for Monte Carlo accuracy

# Exponential PDP for channel
tau_rms = 2.0
pdp = np.exp(-np.arange(M) / tau_rms)
pdp /= np.sum(pdp)

# Improved R_hh for LMMSE (diagonal with real FFT of PDP)
R_hh_fft = np.fft.fft(np.concatenate((pdp, np.zeros(N - M))))
R_hh = np.diag(np.real(R_hh_fft))  # Real part for PSD approximation

ber_esn = np.zeros(len(snr_dbs))
ber_ls = np.zeros(len(snr_dbs))
ber_lmmse = np.zeros(len(snr_dbs))

for snr_idx, snr_db in enumerate(snr_dbs):
    snr_lin = 10**(snr_db / 10.0)
    ber_esn_block = []
    ber_ls_block = []
    ber_lmmse_block = []
    
    for block in range(num_blocks):
        # Random channel
        h = np.sqrt(pdp / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
        
        # Training symbol
        S_freq_train = generate_qpsk_symbols(N)
        s_time_train = np.fft.ifft(S_freq_train) * np.sqrt(N)
        s_time_train /= np.sqrt(np.mean(np.abs(s_time_train)**2))  # Normalize power to 1
        s_cp_train = np.concatenate((s_time_train[-CP_len:], s_time_train))
        u_train = rapp_pa(s_cp_train, ibo_db=3.0, p=1)
        y_cp = np.convolve(u_train, h)[:len(s_cp_train)] + \
               (np.random.randn(len(s_cp_train)) + 1j * np.random.randn(len(s_cp_train))) * np.sqrt(0.5 / snr_lin)
        x_time_train = y_cp[CP_len:]
        
        U_train = np.column_stack((x_time_train.real, x_time_train.imag))
        Y_train = np.column_stack((s_time_train.real, s_time_train.imag))
        
        # ESN setup with delay selection
        res_size = 64  # Optimal from paper
        best_mse = float('inf')
        best_esn = None
        for d in [0, 1, 2]:  # Try delays as per paper Section III-C
            esn = ESN(in_size=2, res_size=res_size, out_size=2, spectral_radius=0.98, connectivity=0.2, delay=d)
            esn.train(U_train, Y_train, washout=100)
            # Predict on training to compute MSE
            Y_hat_train = esn.predict(U_train)
            s_hat_train = Y_hat_train[:, 0] + 1j * Y_hat_train[:, 1]
            mse = np.mean(np.abs(s_time_train - s_hat_train)**2)
            if mse < best_mse:
                best_mse = mse
                best_esn = esn
        esn = best_esn  # Use best ESN
        
        # For LS/LMMSE, FFT to freq domain
        Y_freq_train = np.fft.fft(x_time_train)
        
        # Estimate H
        H_ls = ls_channel_est(Y_freq_train, S_freq_train)
        H_lmmse = lmmse_channel_est(Y_freq_train, S_freq_train, snr_lin, R_hh)
        
        # Test symbols
        block_ber_esn = 0
        block_ber_ls = 0
        block_ber_lmmse = 0
        num_test = num_symbols_per_block - 1
        
        for sym in range(num_test):
            S_freq_test = generate_qpsk_symbols(N)
            s_time_test = np.fft.ifft(S_freq_test) * np.sqrt(N)
            s_time_test /= np.sqrt(np.mean(np.abs(s_time_test)**2))  # Normalize power to 1
            s_cp_test = np.concatenate((s_time_test[-CP_len:], s_time_test))
            u_test = rapp_pa(s_cp_test, ibo_db=3.0, p=1)
            y_cp_test = np.convolve(u_test, h)[:len(s_cp_test)] + \
                        (np.random.randn(len(s_cp_test)) + 1j * np.random.randn(len(s_cp_test))) * np.sqrt(0.5 / snr_lin)
            x_time_test = y_cp_test[CP_len:]
            
            # ESN
            esn.reset()
            U_test = np.column_stack((x_time_test.real, x_time_test.imag))
            Y_hat = esn.predict(U_test)
            s_hat_time = Y_hat[:, 0] + 1j * Y_hat[:, 1]
            S_hat_esn = np.fft.fft(s_hat_time) / np.sqrt(N)
            ber_esn_sym = compute_ber(S_freq_test, S_hat_esn)
            block_ber_esn += ber_esn_sym
            
            # LS and LMMSE (with MMSE equalization)
            Y_freq_test = np.fft.fft(x_time_test)
            S_hat_ls = np.conj(H_ls) * Y_freq_test / (np.abs(H_ls)**2 + 1/snr_lin)
            S_hat_lmmse = np.conj(H_lmmse) * Y_freq_test / (np.abs(H_lmmse)**2 + 1/snr_lin)
            
            ber_ls_sym = compute_ber(S_freq_test, S_hat_ls)
            ber_lmmse_sym = compute_ber(S_freq_test, S_hat_lmmse)
            
            block_ber_ls += ber_ls_sym
            block_ber_lmmse += ber_lmmse_sym
        
        ber_esn_block.append(block_ber_esn / num_test)
        ber_ls_block.append(block_ber_ls / num_test)
        ber_lmmse_block.append(block_ber_lmmse / num_test)
    
    ber_esn[snr_idx] = np.mean(ber_esn_block)
    ber_ls[snr_idx] = np.mean(ber_ls_block)
    ber_lmmse[snr_idx] = np.mean(ber_lmmse_block)
    
    print(f"SNR {snr_db} dB - ESN BER: {ber_esn[snr_idx]:.4f}, LS BER: {ber_ls[snr_idx]:.4f}, LMMSE BER: {ber_lmmse[snr_idx]:.4f}")

# Plot
plt.figure()
plt.semilogy(snr_dbs, ber_esn, 'o-', label='ESN')
plt.semilogy(snr_dbs, ber_ls, 's-', label='LS')
plt.semilogy(snr_dbs, ber_lmmse, 'd-', label='LMMSE')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR for Nonlinear SISO-OFDM')
plt.legend()
plt.grid(True)
plt.show()