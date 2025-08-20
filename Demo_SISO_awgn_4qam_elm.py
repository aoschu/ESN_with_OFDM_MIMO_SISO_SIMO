import numpy as np
import math
from scipy import signal
from HelpFunc import HelpFunc
from pyESN import ESN
import matplotlib.pyplot as plt
import pickle

"""
SISO | AWGN (H=1) | 4-QAM (QPSK) + ESN vs ELM (same hidden size) | no LDPC
- Baselines: Perfect/LS/MMSE
- ML: ESN (matched SNR / fixed-train SNR), ELM (matched / fixed-train)
- Saves BER & capacity curves.
"""

# --------------------
# Physical / noise
# --------------------
W = 2*1.024e6
No = 1e-5                         # two-sided PSD
EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)

# --------------------
# SISO OFDM setup
# --------------------
N = 512                           # subcarriers
m = 2                             # 4-QAM (QPSK)
m_pilot = 2
NumOfdmSymbols = 400
Ptotal = 10**(EbNoDB/10)*No*N     # per-OFDM-symbol power
Pi = Ptotal/N                     # per-subcarrier power (vector over SNRs)

# AWGN channel: H=1, no multipath
IsiDuration = 1
CyclicPrefixLen = max(IsiDuration-1, 0)

# --------------------
# PA model
# --------------------
p_smooth = 1
ClipLeveldB = 3

# --------------------
# QAM helpers
# --------------------
Const = np.array(HelpFunc.UnitQamConstellation(m)).astype(complex)
ConstPilot = np.array(HelpFunc.UnitQamConstellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))

def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]

def hard_bits_from_syms(Xhat, Const, m):
    RxBits = np.zeros((len(Xhat)*m, 1), dtype=int)
    for ii, sym in enumerate(Xhat):
        idx = int(np.argmin(np.abs(Const - sym)))
        RxBits[m*ii:m*(ii+1), 0] = bits_to_grayvec(idx, m)
    return RxBits

def add_cp(x, cp):
    return np.r_[x[-cp:], x] if cp > 0 else x

def remove_cp(x_cp, cp):
    return x_cp[cp:] if cp > 0 else x_cp

# --------------------
# ESN/ELM sizes (same hidden size)
# --------------------
var_x = np.float_power(10, (EbNoDB/10)) * No * N
nInputUnits = 2
nOutputUnits = 2
nInternalUnits = 100            # ESN reservoir size
n_hidden_elm = nInternalUnits   # ELM hidden = ESN reservoir (matching dimension)

inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 0.0000005
spectralRadius = 0.9

teacherShift = np.zeros(nOutputUnits)
feedbackScaling = feedbackScaler*np.ones(nOutputUnits)

Min_Delay = 0
Max_Delay = 2                   # tiny window is enough for AWGN
DelayFlag = 0

# Fixed training SNR for mismatch study
TRAIN_EBNO_FIXED_DB = 12
Pi_train_fixed = (10**(TRAIN_EBNO_FIXED_DB/10)*No*N)/N
var_x_train_fixed = 10**(TRAIN_EBNO_FIXED_DB/10)*No*N

# --------------------
# Simple ELM (single hidden layer)
# --------------------
class ELM:
    def __init__(self, n_inputs, n_hidden, n_outputs, activation='tanh', reg=1e-4, seed=42):
        self.n_in = n_inputs
        self.n_h = n_hidden
        self.n_out = n_outputs
        self.activation = activation
        self.reg = reg
        rng = np.random.default_rng(seed)
        # Xavier-ish init
        self.W = rng.standard_normal((n_inputs, n_hidden)) / np.sqrt(n_inputs)
        self.b = rng.standard_normal((n_hidden,)) / np.sqrt(n_inputs)
        self.Beta = np.zeros((n_hidden, n_outputs))

    def _act(self, Z):
        if self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'relu':
            return np.maximum(0.0, Z)
        else:
            return Z  # linear

    def fit(self, X, T):
        # X: [Tsteps, n_in], T: [Tsteps, n_out]
        H = self._act(X @ self.W + self.b)
        # Ridge regression: Beta = (H^T H + reg I)^-1 H^T T
        HT = H.T
        A = HT @ H + self.reg * np.eye(self.n_h)
        self.Beta = np.linalg.solve(A, HT @ T)

    def predict(self, X):
        H = self._act(X @ self.W + self.b)
        return H @ self.Beta

# --------------------
# Holders
# --------------------
BER_ESN_matched      = np.zeros(len(EbNoDB))
BER_ESN_trainFixed   = np.zeros(len(EbNoDB))
BER_ELM_matched      = np.zeros(len(EbNoDB))
BER_ELM_trainFixed   = np.zeros(len(EbNoDB))
BER_Perfect          = np.zeros(len(EbNoDB))
BER_LS               = np.zeros(len(EbNoDB))
BER_MMSE             = np.zeros(len(EbNoDB))
Capacity_bits_per_sc = np.zeros(len(EbNoDB))

NumBitsPerSymbol = m*N

# --------------------
# Main SNR loop
# --------------------
for jj, ebno_db in enumerate(EbNoDB):
    print(f'EbNoDB = {ebno_db}')
    A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)

    # Input scaling (match vs fixed-train)
    inputScaling_matched   = (inputScaler/(var_x[jj]**0.5)) * np.ones(nInputUnits)
    inputShift_matched     = (inputOffset/inputScaler) * np.ones(nInputUnits)
    teacherScaling         = teacherScalingBase * np.ones(nOutputUnits)
    inputScaling_trainFix  = (inputScaler/(var_x_train_fixed**0.5)) * np.ones(nInputUnits)
    inputShift_trainFix    = (inputOffset/inputScaler) * np.ones(nInputUnits)

    # Reset BER counters
    TotErr_ESN_m = TotErr_ESN_f = 0
    TotErr_ELM_m = TotErr_ELM_f = 0
    TotErr_LS = TotErr_MMSE = TotErr_Perfect = 0
    TotBits = 0

    # AWGN channel
    c0 = np.array([1.0 + 0j])
    H_true = np.ones(N, dtype=complex)
    snr_sc = (Pi[jj]/No) * (np.abs(H_true)**2)
    Capacity_bits_per_sc[jj] = float(np.mean(np.log2(1 + snr_sc)))

    # ---- Pilot (matched SNR) ----
    TxBitsPilot = (np.random.rand(N*m_pilot, 1) > 0.5).astype(np.int32)
    X_p = np.zeros((N,), dtype=complex)
    for ii in range(N):
        idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), 0])[0])
        X_p[ii] = ConstPilot[idx]

    x_temp = N * np.fft.ifft(X_p)
    x_CP_p = add_cp(x_temp, CyclicPrefixLen) * (Pi[jj]**0.5)
    x_CP_p_NLD = x_CP_p / ((1 + (np.abs(x_CP_p)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
    y_time_p = signal.lfilter(c0, np.array([1]), x_CP_p_NLD)
    noise_p = math.sqrt(len(y_time_p)*No/2) * (np.random.randn(len(y_time_p)) + 1j*np.random.randn(len(y_time_p)))
    y_time_p = y_time_p + noise_p

    y_noCP_p = remove_cp(y_time_p, CyclicPrefixLen)
    Y_p = (1/N) * np.fft.fft(y_noCP_p)

    # LS/MMSE (trivial under AWGN but kept)
    H_LS = (Y_p / X_p) / (Pi[jj]**0.5)
    R_h = np.diag([1.0])
    MMSE_bold_TD = np.linalg.inv(R_h) * (No/Pi[jj])/(N/2) + np.eye(IsiDuration)
    c_LS_trunc = np.fft.ifft(H_LS)[:IsiDuration]
    c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS_trunc)
    H_MMSE = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])

    # ---- Pilot (fixed-train SNR for mismatch) ----
    x_CP_pf = add_cp(x_temp, CyclicPrefixLen) * (Pi_train_fixed**0.5)
    A_Clip_train = np.sqrt(var_x_train_fixed) * np.float_power(10, ClipLeveldB/20)
    x_CP_pf_NLD = x_CP_pf / ((1 + (np.abs(x_CP_pf)/A_Clip_train)**(2*p_smooth))**(1/(2*p_smooth)))
    y_time_pf = signal.lfilter(c0, np.array([1]), x_CP_pf_NLD) + noise_p  # reuse same noise

    # ---- Train ESN (matched & fixed) ----
    def build_delay_lut():
        if DelayFlag:
            dl = []
            for d0 in range(Min_Delay, Max_Delay+1):
                for d1 in range(Min_Delay, Max_Delay+1):
                    dl.append([d0, d1])
            return np.array(dl, dtype=int)
        else:
            dl = np.zeros(((Max_Delay + 1 - Min_Delay), 2), dtype=int)
            for d in range(Min_Delay, Max_Delay+1):
                dl[d - Min_Delay, :] = d
            return dl

    def train_esn_from_pilot(y_cp, x_cp, inputScaling_used, inputShift_used):
        Delay_LUT = build_delay_lut()
        Delay_Max_vec = np.amax(Delay_LUT, axis=1)
        Delay_Min_vec = np.amin(Delay_LUT, axis=1)
        NMSE_list = np.zeros(Delay_LUT.shape[0])

        esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                  spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                  input_shift=inputShift_used, input_scaling=inputScaling_used,
                  teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                  feedback_scaling=feedbackScaling)

        x_ref = remove_cp(x_cp, CyclicPrefixLen)

        for didx in range(Delay_LUT.shape[0]):
            d0, d1 = Delay_LUT[didx, :]
            dmax = Delay_Max_vec[didx]; dmin = Delay_Min_vec[didx]

            ESN_input = np.zeros((N + dmax + CyclicPrefixLen, nInputUnits))
            ESN_output = np.zeros((N + dmax + CyclicPrefixLen, nOutputUnits))
            ESN_input[:, 0] = np.r_[y_cp.real, np.zeros(dmax)]
            ESN_input[:, 1] = np.r_[y_cp.imag, np.zeros(dmax)]
            ESN_output[d0:(d0+N+CyclicPrefixLen), 0] = x_cp.real
            ESN_output[d1:(d1+N+CyclicPrefixLen), 1] = x_cp.imag

            nForget_tmp = int(dmin + CyclicPrefixLen)
            esn.fit(ESN_input, ESN_output, nForget_tmp)
            x_hat_tmp = esn.predict(ESN_input, nForget_tmp, continuation=False)

            x_hat_td = (x_hat_tmp[d0 - dmin : d0 - dmin + N + 1, 0]
                        + 1j * x_hat_tmp[d1 - dmin : d1 - dmin + N + 1, 1])
            x_hat_td = x_hat_td[:N]  # ensure length N
            NMSE_list[didx] = np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2

        didx_best = int(np.argmin(NMSE_list))
        Delay_sel = Delay_LUT[didx_best, :]
        Delay_Min_sel = int(Delay_Min_vec[didx_best])
        Delay_Max_sel = int(Delay_Max_vec[didx_best])

        # Final training
        ESN_input = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nInputUnits))
        ESN_output = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nOutputUnits))
        ESN_input[:, 0] = np.r_[y_cp.real, np.zeros(Delay_Max_sel)]
        ESN_input[:, 1] = np.r_[y_cp.imag, np.zeros(Delay_Max_sel)]
        ESN_output[Delay_sel[0]:(Delay_sel[0]+N+CyclicPrefixLen), 0] = x_cp.real
        ESN_output[Delay_sel[1]:(Delay_sel[1]+N+CyclicPrefixLen), 1] = x_cp.imag

        nForget_final = int(Delay_Min_sel + CyclicPrefixLen)
        esn.fit(ESN_input, ESN_output, nForget_final)
        return esn, Delay_sel, Delay_Min_sel, Delay_Max_sel, nForget_final

    # ---- Train ELM (matched & fixed) ----
    def train_elm_from_pilot(y_cp, x_cp, inputScaling_used, inputShift_used, reg=1e-4):
        Delay_LUT = build_delay_lut()
        Delay_Max_vec = np.amax(Delay_LUT, axis=1)
        Delay_Min_vec = np.amin(Delay_LUT, axis=1)
        NMSE_list = np.zeros(Delay_LUT.shape[0])

        elm = ELM(n_inputs=nInputUnits, n_hidden=n_hidden_elm, n_outputs=nOutputUnits,
                  activation='tanh', reg=reg, seed=123)

        x_ref = remove_cp(x_cp, CyclicPrefixLen)

        def preprocess(U):
            # mimic ESN's scaling direction: scaled = input_scaling * (u + input_shift)
            return (U + inputShift_used) * inputScaling_used

        for didx in range(Delay_LUT.shape[0]):
            d0, d1 = Delay_LUT[didx, :]
            dmax = Delay_Max_vec[didx]; dmin = Delay_Min_vec[didx]

            X_in = np.zeros((N + dmax + CyclicPrefixLen, nInputUnits))
            T_out = np.zeros((N + dmax + CyclicPrefixLen, nOutputUnits))
            X_in[:, 0] = np.r_[y_cp.real, np.zeros(dmax)]
            X_in[:, 1] = np.r_[y_cp.imag, np.zeros(dmax)]
            T_out[d0:(d0+N+CyclicPrefixLen), 0] = x_cp.real
            T_out[d1:(d1+N+CyclicPrefixLen), 1] = x_cp.imag

            Xs = preprocess(X_in)
            elm.fit(Xs, T_out)
            y_hat = elm.predict(Xs)

            x_hat_td = (y_hat[d0 - dmin : d0 - dmin + N + 1, 0]
                        + 1j * y_hat[d1 - dmin : d1 - dmin + N + 1, 1])
            x_hat_td = x_hat_td[:N]
            NMSE_list[didx] = np.linalg.norm(x_hat_td - x_ref)**2 / np.linalg.norm(x_ref)**2

        didx_best = int(np.argmin(NMSE_list))
        Delay_sel = Delay_LUT[didx_best, :]
        Delay_Min_sel = int(Delay_Min_vec[didx_best])
        Delay_Max_sel = int(Delay_Max_vec[didx_best])

        # Final train
        X_in = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nInputUnits))
        T_out = np.zeros((N + Delay_Max_sel + CyclicPrefixLen, nOutputUnits))
        X_in[:, 0] = np.r_[y_cp.real, np.zeros(Delay_Max_sel)]
        X_in[:, 1] = np.r_[y_cp.imag, np.zeros(Delay_Max_sel)]
        T_out[Delay_sel[0]:(Delay_sel[0]+N+CyclicPrefixLen), 0] = x_cp.real
        T_out[Delay_sel[1]:(Delay_sel[1]+N+CyclicPrefixLen), 1] = x_cp.imag

        Xs = (X_in + inputShift_used) * inputScaling_used
        elm.fit(Xs, T_out)
        return elm, Delay_sel, Delay_Min_sel, Delay_Max_sel

    # Train both model families on pilots
    esn_matched,  Delay_m, Delay_Min_m, Delay_Max_m, nForget_m = \
        train_esn_from_pilot(y_time_p, x_CP_p, inputScaling_matched,  inputShift_matched)
    esn_trainFixed, Delay_f, Delay_Min_f, Delay_Max_f, nForget_f = \
        train_esn_from_pilot(y_time_pf, x_CP_pf, inputScaling_trainFix, inputShift_trainFix)

    elm_matched,  EDelay_m, EDelay_Min_m, EDelay_Max_m = \
        train_elm_from_pilot(y_time_p,  x_CP_p, inputScaling_matched,  inputShift_matched,  reg=1e-4)
    elm_trainFixed, EDelay_f, EDelay_Min_f, EDelay_Max_f = \
        train_elm_from_pilot(y_time_pf, x_CP_pf, inputScaling_trainFix, inputShift_trainFix, reg=1e-4)

    # ---- Data loop ----
    for kk in range(2, NumOfdmSymbols+1):
        TxBits = (np.random.rand(N*m, 1) > 0.5).astype(np.int32)
        X = np.zeros((N,), dtype=complex)
        for ii in range(N):
            idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), 0])[0])
            X[ii] = Const[idx]

        x_temp = N * np.fft.ifft(X)
        x_CP = add_cp(x_temp, CyclicPrefixLen) * (Pi[jj]**0.5)
        x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

        y_time = signal.lfilter(c0, np.array([1]), x_CP_NLD)
        noise = math.sqrt(len(y_time)*No/2) * (np.random.randn(len(y_time)) + 1j*np.random.randn(len(y_time)))
        y_time = y_time + noise

        y_noCP = remove_cp(y_time, CyclicPrefixLen)
        Y = (1/N) * np.fft.fft(y_noCP)

        # ESN predicts time-domain clean x; then FFT -> freq symbols
        ESN_input_m = np.zeros((N + Delay_Max_m + CyclicPrefixLen, nInputUnits))
        ESN_input_m[:, 0] = np.r_[y_time.real, np.zeros(Delay_Max_m)]
        ESN_input_m[:, 1] = np.r_[y_time.imag, np.zeros(Delay_Max_m)]
        x_hat_m_tmp = esn_matched.predict(ESN_input_m, nForget_m, continuation=False)
        x_hat_td_m = (x_hat_m_tmp[Delay_m[0] - Delay_Min_m : Delay_m[0] - Delay_Min_m + N + 1, 0]
                      + 1j * x_hat_m_tmp[Delay_m[1] - Delay_Min_m : Delay_m[1] - Delay_Min_m + N + 1, 1])[:N]
        X_hat_ESN_m = (1/N) * np.fft.fft(x_hat_td_m) / math.sqrt(Pi[jj])

        ESN_input_f = np.zeros((N + Delay_Max_f + CyclicPrefixLen, nInputUnits))
        ESN_input_f[:, 0] = np.r_[y_time.real, np.zeros(Delay_Max_f)]
        ESN_input_f[:, 1] = np.r_[y_time.imag, np.zeros(Delay_Max_f)]
        x_hat_f_tmp = esn_trainFixed.predict(ESN_input_f, nForget_f, continuation=False)
        x_hat_td_f = (x_hat_f_tmp[Delay_f[0] - Delay_Min_f : Delay_f[0] - Delay_Min_f + N + 1, 0]
                      + 1j * x_hat_f_tmp[Delay_f[1] - Delay_Min_f : Delay_f[1] - Delay_Min_f + N + 1, 1])[:N]
        X_hat_ESN_f = (1/N) * np.fft.fft(x_hat_td_f) / math.sqrt(Pi[jj])

        # ELM predictions (same input/target layout)
        def preprocess(U, scale, shift):
            return (U + shift) * scale

        ELM_input_m = np.zeros((N + EDelay_Max_m + CyclicPrefixLen, nInputUnits))
        ELM_input_m[:, 0] = np.r_[y_time.real, np.zeros(EDelay_Max_m)]
        ELM_input_m[:, 1] = np.r_[y_time.imag, np.zeros(EDelay_Max_m)]
        y_hat_m = elm_matched.predict(preprocess(ELM_input_m, inputScaling_matched, inputShift_matched))
        x_hat_td_elm_m = (y_hat_m[EDelay_m[0] - EDelay_Min_m : EDelay_m[0] - EDelay_Min_m + N + 1, 0]
                          + 1j * y_hat_m[EDelay_m[1] - EDelay_Min_m : EDelay_m[1] - EDelay_Min_m + N + 1, 1])[:N]
        X_hat_ELM_m = (1/N) * np.fft.fft(x_hat_td_elm_m) / math.sqrt(Pi[jj])

        ELM_input_f = np.zeros((N + EDelay_Max_f + CyclicPrefixLen, nInputUnits))
        ELM_input_f[:, 0] = np.r_[y_time.real, np.zeros(EDelay_Max_f)]
        ELM_input_f[:, 1] = np.r_[y_time.imag, np.zeros(EDelay_Max_f)]
        y_hat_f = elm_trainFixed.predict(preprocess(ELM_input_f, inputScaling_trainFix, inputShift_trainFix))
        x_hat_td_elm_f = (y_hat_f[EDelay_f[0] - EDelay_Min_f : EDelay_f[0] - EDelay_Min_f + N + 1, 0]
                          + 1j * y_hat_f[EDelay_f[1] - EDelay_Min_f : EDelay_f[1] - EDelay_Min_f + N + 1, 1])[:N]
        X_hat_ELM_f = (1/N) * np.fft.fft(x_hat_td_elm_f) / math.sqrt(Pi[jj])

        # Baselines (trivial under AWGN)
        X_hat_Perfect = (Y / H_true) / math.sqrt(Pi[jj])
        X_hat_LS      = (Y / H_LS)   / math.sqrt(Pi[jj])
        X_hat_MMSE    = (Y / H_MMSE) / math.sqrt(Pi[jj])

        # Bit decisions
        RxBits_ESN_m   = hard_bits_from_syms(X_hat_ESN_m, Const, m)
        RxBits_ESN_f   = hard_bits_from_syms(X_hat_ESN_f, Const, m)
        RxBits_ELM_m   = hard_bits_from_syms(X_hat_ELM_m, Const, m)
        RxBits_ELM_f   = hard_bits_from_syms(X_hat_ELM_f, Const, m)
        RxBits_LS      = hard_bits_from_syms(X_hat_LS, Const, m)
        RxBits_MMSE    = hard_bits_from_syms(X_hat_MMSE, Const, m)
        RxBits_Perfect = hard_bits_from_syms(X_hat_Perfect, Const, m)

        # Accumulate errors
        TotErr_ESN_m += int(np.sum(TxBits != RxBits_ESN_m))
        TotErr_ESN_f += int(np.sum(TxBits != RxBits_ESN_f))
        TotErr_ELM_m += int(np.sum(TxBits != RxBits_ELM_m))
        TotErr_ELM_f += int(np.sum(TxBits != RxBits_ELM_f))
        TotErr_LS    += int(np.sum(TxBits != RxBits_LS))
        TotErr_MMSE  += int(np.sum(TxBits != RxBits_MMSE))
        TotErr_Perfect += int(np.sum(TxBits != RxBits_Perfect))
        TotBits      += NumBitsPerSymbol

    # BER per SNR
    BER_ESN_matched[jj]    = TotErr_ESN_m / max(TotBits, 1)
    BER_ESN_trainFixed[jj] = TotErr_ESN_f / max(TotBits, 1)
    BER_ELM_matched[jj]    = TotErr_ELM_m / max(TotBits, 1)
    BER_ELM_trainFixed[jj] = TotErr_ELM_f / max(TotBits, 1)
    BER_LS[jj]             = TotErr_LS / max(TotBits, 1)
    BER_MMSE[jj]           = TotErr_MMSE / max(TotBits, 1)
    BER_Perfect[jj]        = TotErr_Perfect / max(TotBits, 1)

# --------------------
# Save results
# --------------------
results_ber = {
    "EBN0": EbNoDB,
    "BER": {
        "ESN_matched": BER_ESN_matched,
        "ESN_trainFixed": BER_ESN_trainFixed,
        "ELM_matched": BER_ELM_matched,
        "ELM_trainFixed": BER_ELM_trainFixed,
        "LS": BER_LS,
        "MMSE": BER_MMSE,
        "Perfect": BER_Perfect
    },
    "meta": {
        "TRAIN_EBNO_FIXED_DB": TRAIN_EBNO_FIXED_DB,
        "N": int(N), "m": int(m),
        "IsiDuration": int(IsiDuration),
        "channel": "AWGN (H=1)",
        "hidden_ESN": int(nInternalUnits),
        "hidden_ELM": int(n_hidden_elm)
    }
}
with open("./BERvsEBNo_ESN_ELM_QPSK_awgn.pkl", "wb") as f:
    pickle.dump(results_ber, f)

results_channel = {
    "EBN0": EbNoDB,
    "capacity_bits_per_sc": Capacity_bits_per_sc,
    "absHk_stats": {
        "mean": np.ones_like(EbNoDB, dtype=float),
        "p10": np.ones_like(EbNoDB, dtype=float),
        "p90": np.ones_like(EbNoDB, dtype=float)
    },
    "notes": "SISO AWGN: |H_k|=1; capacity = log2(1+SNR)."
}
with open("./channel_metrics_siso_QPSK_awgn.pkl", "wb") as f:
    pickle.dump(results_channel, f)

# --------------------
# Plots
# --------------------
plt.figure()
plt.semilogy(EbNoDB, BER_MMSE, 'rs-.', label='MMSE')
plt.semilogy(EbNoDB, BER_LS, 'o-', label='LS')
plt.semilogy(EbNoDB, BER_ESN_matched, 'gd--', label='ESN (matched)')
plt.semilogy(EbNoDB, BER_ESN_trainFixed, 'b^:', label=f'ESN (train @{TRAIN_EBNO_FIXED_DB} dB)')
plt.semilogy(EbNoDB, BER_ELM_matched, 'm*--', label='ELM (matched)')
plt.semilogy(EbNoDB, BER_ELM_trainFixed, 'c>:', label=f'ELM (train @{TRAIN_EBNO_FIXED_DB} dB)')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.title('SISO | 4-QAM | ESN vs ELM under AWGN + PA')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
plt.show()
plt.savefig("./BERvsEBNo_ESN_ELM_QPSK_awgn.png", dpi=150)

plt.figure()
plt.plot(EbNoDB, Capacity_bits_per_sc, 'm.-', label='Avg. capacity per subcarrier')
plt.grid(True, ls=':'); plt.legend()
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Capacity [bits/s/Hz per subcarrier]')
plt.title('SISO capacity | AWGN (H=1)')
plt.tight_layout()
plt.show()
