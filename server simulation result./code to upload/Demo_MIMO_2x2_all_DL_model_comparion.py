#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_dl_models.py  — upgraded DL models + input/target normalization

- FNN/CNN/RNN made deeper but sized around ESN's internal state (~100 units)
- Feature-wise standardization for inputs (and targets for learned models)
- De-normalization applied at inference before time→freq conversion

Original baseline: see prior version.  (AOSCHU request)
"""

import os
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate

# Local utilities
from HelpFunc import HelpFunc
from pyESN import ESN  # used only to invoke HelpFunc.trainMIMOESN similarly to the baseline

# ----------------------
# Model Implementations
# ----------------------

class ELM:
    """Extreme Learning Machine (single hidden layer, ridge regression readout)."""
    def __init__(self, n_hidden=200, activation="tanh", alpha=1e-3, seed=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.b = None
        self.W_out = None

    def _act(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return x  # linear

    def fit(self, X, Y):
        # X: (T, in_dim), Y: (T, out_dim)
        T, in_dim = X.shape
        _, out_dim = Y.shape
        self.W = self.rng.normal(0.0, 1.0, size=(in_dim, self.n_hidden))
        self.b = self.rng.normal(0.0, 1.0, size=(self.n_hidden,))
        H = self._act(X @ self.W + self.b)
        I = np.eye(self.n_hidden)
        self.W_out = np.linalg.solve(H.T @ H + self.alpha * I, H.T @ Y)

    def predict(self, X):
        H = self._act(X @ self.W + self.b)
        return H @ self.W_out


def build_mlp(input_dim, output_dim, state_dim=100, dropout=0.1):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(2*state_dim, activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Dense(state_dim, activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Dense(state_dim, activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Dense(output_dim, activation="linear"),
    ], name="FNN")


def build_cnn(time_steps, feat_dim, output_dim, state_dim=100, dropout=0.1):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=(time_steps, feat_dim)),
        layers.Conv1D(state_dim, kernel_size=7, padding="same", activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Conv1D(state_dim, kernel_size=5, padding="same", activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Conv1D(state_dim, kernel_size=3, padding="same", activation="relu"),
        layers.LayerNormalization(),
        layers.Dropout(dropout),
        layers.Conv1D(output_dim, kernel_size=1, padding="same", activation="linear"),
    ], name="CNN1D")


def build_rnn(time_steps, feat_dim, output_dim, state_dim=100, dropout=0.1):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.Input(shape=(time_steps, feat_dim)),
        layers.LSTM(state_dim, return_sequences=True, dropout=dropout, recurrent_dropout=0.0),
        layers.LayerNormalization(),
        layers.LSTM(state_dim, return_sequences=True, dropout=dropout, recurrent_dropout=0.0),
        layers.LayerNormalization(),
        layers.Dense(output_dim, activation="linear"),
    ], name="RNN_LSTM")


# ----------------------
# Simulation Parameters
# ----------------------

def get_default_params():
    params = dict()
    # Physical parameters
    params["W"] = 2 * 1.024e6
    params["f_D"] = 100
    params["No"] = 1e-5
    params["IsiDuration"] = 8
    params["cFlag"] = False

    # SNR sweep
    params["EbNoDB"] = np.arange(0, 30 + 1, 3).astype(np.int32)

    # MIMO
    params["N_t"] = 2
    params["N_r"] = 2

    # OFDM/Design
    params["N"] = 512
    params["m"] = 4
    params["m_pilot"] = 4
    params["NumOfdmSymbols"] = 400

    # PA
    params["p_smooth"] = 1
    params["ClipLeveldB"] = 3

    return params


# ----------------------
# Utility helpers
# ----------------------

def unit_qam_constellation(m):
    return HelpFunc.UnitQamConstellation(m)

def semilogy_ber(ebn0, curves, title, save_path):
    plt.figure(figsize=(7,5), dpi=120)
    for label, y in curves.items():
        plt.semilogy(ebn0, y, marker='o', linewidth=1.5, label=label)
    plt.grid(True, which='both', linestyle=':')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


# ----------------------
# Main experiment
# ----------------------

def run_experiment(args):
    # Unpack parameters
    p = get_default_params()
    W = p["W"]; f_D = p["f_D"]; No = p["No"]; IsiDuration = p["IsiDuration"]
    EbNoDB = p["EbNoDB"]; N_t = p["N_t"]; N_r = p["N_r"]; N = p["N"]
    m = p["m"]; m_pilot = p["m_pilot"]; NumOfdmSymbols = p["NumOfdmSymbols"]
    p_smooth = p["p_smooth"]; ClipLeveldB = p["ClipLeveldB"]

    Subcarrier_Spacing = W / N
    Ptotal = (10**(EbNoDB/10)) * No * N

    # Timing
    T_OFDM = N / W
    T_OFDM_Total = (N + IsiDuration - 1) / W
    tau_c = 0.5 / f_D
    L = math.floor(tau_c / T_OFDM_Total)

    # Constellations
    Const = unit_qam_constellation(m)
    ConstPilot = unit_qam_constellation(m_pilot)

    # Helpers
    PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
    PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
    CyclicPrefixLen = IsiDuration - 1

    # One-sided exponential power delay profile (normalized)
    temp = CyclicPrefixLen / 9
    IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen + 1)) / temp)
    IsiMagnitude = IsiMagnitude / np.sum(IsiMagnitude)

    # ESN params (baseline; used to call HelpFunc.trainMIMOESN)
    var_x = np.float_power(10, (EbNoDB/10)) * No * N
    nInputUnits = N_t * 2
    nOutputUnits = N_t * 2
    nInternalUnits = 100  # ESN state size; we'll size DL models around this
    STATE_DIM = nInternalUnits

    inputScaler = 0.005
    inputOffset = 0.0
    feedbackScaler = 0.0
    teacherScaling = 0.0000005 * np.ones(N_t * 2)
    spectralRadius = 0.9
    teacherShift = np.zeros(N_t * 2)
    feedbackScaling = feedbackScaler * np.ones(N_t * 2)
    Min_Delay = 0
    Max_Delay = math.ceil(IsiDuration/2) + 2
    DelayFlag = 0

    # Correlation and MMSE scaler
    R_h = np.zeros((IsiDuration, IsiDuration))
    for ii in range(IsiDuration):
        R_h[ii, ii] = IsiMagnitude[ii]

    # Storage
    models = ["ESN", "ELM", "FNN", "CNN", "RNN", "LS", "MMSE", "Perfect"]
    BER = {k: np.zeros(len(EbNoDB)) for k in models}
    NMSE_train = {k: np.zeros(len(EbNoDB)) for k in ["ESN", "ELM", "FNN", "CNN", "RNN"]}
    NMSE_test = {k: np.zeros(len(EbNoDB)) for k in ["ESN", "ELM", "FNN", "CNN", "RNN"]}

    rng = np.random.default_rng(123)

    # Main SNR loop
    for jj, eb in enumerate(EbNoDB):
        print(f"[SNR] Eb/N0 = {eb} dB")
        Pi = (10**(eb/10)) * No  # power per subcarrier (equal allocation)
        A_Clip = np.sqrt(var_x[jj]) * np.float_power(10, ClipLeveldB/20)
        inputScaling = inputScaler / (var_x[jj]**0.5) * np.ones(N_t * 2)
        inputShift = inputOffset / inputScaler * np.ones(N_t * 2)

        # Accumulators
        TotalBerNum = {k: 0 for k in models}
        TotalBerDen = 0

        # Channel correlation inverse term for MMSE (time-domain)
        MMSE_bold_TD = np.dot(np.linalg.inv(R_h), (No/Pi)/(N/2)) + np.eye(IsiDuration)

        # Placeholders for trained models (per SNR)
        trained_esn = None
        elm = None; mlp = None; cnn = None; rnn = None
        Delay = None; Delay_Min = None; Delay_Max = Max_Delay; nForgetPoints = None
        Ci = [[None] * N_t for _ in range(N_r)]  # true FR (for Perfect/LS/MMSE equalizers)

        # Normalization scalers (filled during pilot training)
        X_mu = None; X_sigma = None; Y_mu = None; Y_sigma = None

        for kk in range(1, NumOfdmSymbols + 1):
            # (Re)draw channel every L symbols
            if (np.remainder(kk, L) == 1):
                c = [[None] * N_t for _ in range(N_r)]
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        c[nnn][mmm] = rng.normal(size=IsiDuration)/(2**0.5) + 1j * rng.normal(size=IsiDuration)/(2**0.5)
                        c[nnn][mmm] = c[nnn][mmm] * (IsiMagnitude**0.5)
                        Ci[nnn][mmm] = np.fft.fft(np.append(c[nnn][mmm], np.zeros(N - len(c[nnn][mmm]))))

                # ---------- Pilot symbol (for training) ----------
                TxBits = (rng.random(size=(N*m_pilot, N_t)) > 0.5).astype(np.int32)

                Xf = np.zeros((N, N_t), dtype=np.complex128)
                x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)
                for ii in range(N):
                    for iii in range(N_t):
                        idx = int(np.matmul(PowersOfTwoPilot[:, :m_pilot], TxBits[m_pilot*ii + np.arange(m_pilot), iii])[0])
                        Xf[ii, iii] = ConstPilot[idx]

                # Time-domain signals with CP
                for iii in range(N_t):
                    x_temp = N * np.fft.ifft(Xf[:, iii])
                    x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                    x_CP[:, iii] = x_CP[:, iii] * (Pi**0.5)

                # Nonlinear PA
                x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

                y_CP_NLD = np.zeros((N + CyclicPrefixLen, N_r), dtype=np.complex128)
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        y_CP_NLD[:, nnn] += signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
                    noise = math.sqrt(y_CP_NLD.shape[0]*No/2) * np.matmul(rng.normal(size=(y_CP_NLD.shape[0], 2)),
                                                                         np.array([[1], [1j]])).reshape(-1)
                    y_CP_NLD[:, nnn] += noise

                # ---- Call HelpFunc to prepare ESN training pairs ----
                esn = ESN(n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                          spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                          input_shift=inputShift, input_scaling=inputScaling,
                          teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                          feedback_scaling=feedbackScaling)

                (ESN_input, ESN_output, trained_esn, Delay, Delay_Idx,
                 Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN_train) = HelpFunc.trainMIMOESN(
                    esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r,
                    IsiDuration, y_CP_NLD, x_CP)

                NMSE_train["ESN"][jj] += NMSE_ESN_train

                # Prepare arrays for other models (drop initial forget points)
                X_tr = ESN_input[nForgetPoints:, :]
                Y_tr = ESN_output[nForgetPoints:, :]

                # -------- Standardization (inputs & targets) --------
                X_mu, X_sigma = standardize_fit(X_tr)
                X_tr_n = standardize_apply(X_tr, X_mu, X_sigma)

                Y_mu, Y_sigma = standardize_fit(Y_tr)
                Y_tr_n = standardize_apply(Y_tr, Y_mu, Y_sigma)

                # -------- Train ELM (normalized) --------
                elm = ELM(n_hidden=2*STATE_DIM, activation="tanh", alpha=1e-3, seed=42)
                elm.fit(X_tr_n, Y_tr_n)

                # -------- Train TF models (normalized) --------
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    tf.get_logger().setLevel('ERROR')
                    tf.random.set_seed(1234)

                    # FNN / MLP
                    mlp = build_mlp(input_dim=X_tr_n.shape[1], output_dim=Y_tr_n.shape[1],
                                    state_dim=STATE_DIM, dropout=0.1)
                    mlp.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
                    mlp.fit(X_tr_n, Y_tr_n, epochs=10, batch_size=1024, verbose=0)

                    # CNN/RNN expect 3D input (batch, time, feat)
                    X_tr_seq = X_tr_n[None, :, :]       # (1, T, F)
                    Y_tr_seq = Y_tr_n[None, :, :]       # (1, T, O)

                    # CNN
                    cnn = build_cnn(time_steps=X_tr_n.shape[0], feat_dim=X_tr_n.shape[1],
                                    output_dim=Y_tr_n.shape[1], state_dim=STATE_DIM, dropout=0.1)
                    cnn.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
                    cnn.fit(X_tr_seq, Y_tr_seq, epochs=10, batch_size=1, verbose=0)

                    # RNN
                    rnn = build_rnn(time_steps=X_tr_n.shape[0], feat_dim=X_tr_n.shape[1],
                                    output_dim=Y_tr_n.shape[1], state_dim=STATE_DIM, dropout=0.1)
                    rnn.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
                    rnn.fit(X_tr_seq, Y_tr_seq, epochs=10, batch_size=1, verbose=0)

                except Exception as e:
                    print("[WARN] TensorFlow not available or training failed:", e)
                    mlp = None; cnn = None; rnn = None

                # store Ci for equalizers already set above
                # Done with training part for this channel coherence block

            else:
                # ---------- Data symbol (for evaluation) ----------
                TxBits = (rng.random(size=(N*m, N_t)) > 0.5).astype(np.int32)
                Xf = np.zeros((N, N_t), dtype=np.complex128)
                x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)
                for ii in range(N):
                    for iii in range(N_t):
                        idx = int(np.matmul(PowersOfTwo[:, :m], TxBits[m*ii + np.arange(m), iii])[0])
                        Xf[ii, iii] = Const[idx]

                for iii in range(N_t):
                    x_temp = N * np.fft.ifft(Xf[:, iii])
                    x_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                    x_CP[:, iii] = x_CP[:, iii] * (Pi**0.5)

                # Nonlinear PA
                x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

                Y_NLD = np.zeros((N, N_r), dtype=np.complex128)
                y_CP_NLD = np.zeros((N + CyclicPrefixLen, N_r), dtype=np.complex128)

                for nnn in range(N_r):
                    for mmm in range(N_t):
                        y_CP_NLD[:, nnn] += signal.lfilter(c[nnn][mmm], np.array([1]), x_CP_NLD[:, mmm])
                    noise = math.sqrt(y_CP_NLD.shape[0]*No/2) * np.matmul(rng.normal(size=(y_CP_NLD.shape[0], 2)),
                                                                         np.array([[1], [1j]])).reshape(-1)
                    y_CP_NLD[:, nnn] += noise
                    Y_NLD[:, nnn] = (1/N) * np.fft.fft(y_CP_NLD[IsiDuration-1:, nnn])

                # Build ESN-style input for inference
                ESN_input_test = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
                ESN_input_test[:, 0] = np.append(y_CP_NLD[:, 0].real, np.zeros(Delay_Max))
                ESN_input_test[:, 1] = np.append(y_CP_NLD[:, 0].imag, np.zeros(Delay_Max))
                ESN_input_test[:, 2] = np.append(y_CP_NLD[:, 1].real, np.zeros(Delay_Max))
                ESN_input_test[:, 3] = np.append(y_CP_NLD[:, 1].imag, np.zeros(Delay_Max))

                # --- ESN prediction (unchanged scaling) ---
                nForgetPoints_inf = Delay_Min + CyclicPrefixLen
                xhat_esn_temp = trained_esn.predict(ESN_input_test, nForgetPoints_inf, continuation=False)

                x_hat_ESN_0 = xhat_esn_temp[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0] \
                              + 1j * xhat_esn_temp[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1]
                x_hat_ESN_1 = xhat_esn_temp[Delay[2] - Delay_Min : Delay[2] - Delay_Min + N + 1, 2] \
                              + 1j * xhat_esn_temp[Delay[3] - Delay_Min : Delay[3] - Delay_Min + N + 1, 3]
                x_hat_ESN = np.hstack((x_hat_ESN_0.reshape(-1,1), x_hat_ESN_1.reshape(-1,1)))

                x_true_td = x_CP[IsiDuration-1:, :]
                NMSE_test["ESN"][jj] += (
                    np.linalg.norm(x_hat_ESN[:,0] - x_true_td[:,0])**2 / np.linalg.norm(x_true_td[:,0])**2
                    + np.linalg.norm(x_hat_ESN[:,1] - x_true_td[:,1])**2 / np.linalg.norm(x_true_td[:,1])**2
                )
                X_hat_ESN = np.zeros_like(Xf, dtype=np.complex128)
                for ii in range(N_t):
                    X_hat_ESN[:, ii] = (1/N) * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi)

                # --- ELM / FNN / CNN / RNN predictions (with normalize -> denormalize) ---
                def predict_and_build_xhat(model, kind):
                    if model is None or X_mu is None:
                        return None
                    X_te_sl = ESN_input_test[nForgetPoints:, :]
                    X_te_n = standardize_apply(X_te_sl, X_mu, X_sigma)

                    if kind == "ELM":
                        # ELM is a numpy class; no verbose kwarg
                        Y_pred_n = model.predict(X_te_n)
                    elif kind == "MLP":
                        Y_pred_n = model.predict(X_te_n, verbose=0)
                    else:
                        X_seq = X_te_n[None, :, :]
                        Y_pred_n = model.predict(X_seq, verbose=0)[0]


                    Y_pred = Y_pred_n * Y_sigma + Y_mu  # de-normalize

                    x0 = Y_pred[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0] \
                         + 1j * Y_pred[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1]
                    x1 = Y_pred[Delay[2] - Delay_Min : Delay[2] - Delay_Min + N + 1, 2] \
                         + 1j * Y_pred[Delay[3] - Delay_Min : Delay[3] - Delay_Min + N + 1, 3]
                    xhat_td = np.hstack((x0.reshape(-1,1), x1.reshape(-1,1)))

                    # NMSE
                    NMSE_tmp = (
                        np.linalg.norm(xhat_td[:,0] - x_true_td[:,0])**2 / np.linalg.norm(x_true_td[:,0])**2
                        + np.linalg.norm(xhat_td[:,1] - x_true_td[:,1])**2 / np.linalg.norm(x_true_td[:,1])**2
                    )

                    Xhat = np.zeros_like(Xf, dtype=np.complex128)
                    for ii in range(N_t):
                        Xhat[:, ii] = (1/N) * np.fft.fft(xhat_td[:, ii]) / math.sqrt(Pi)
                    return Xhat, NMSE_tmp

                # ELM
                elm_res = predict_and_build_xhat(elm, "ELM")
                if elm_res is not None:
                    X_hat_ELM, nmse = elm_res
                    NMSE_test["ELM"][jj] += nmse
                else:
                    X_hat_ELM = None

                # FNN
                if 'mlp' in locals() and mlp is not None:
                    fnn_res = predict_and_build_xhat(mlp, "MLP")
                    if fnn_res is not None:
                        X_hat_FNN, nmse = fnn_res
                        NMSE_test["FNN"][jj] += nmse
                    else:
                        X_hat_FNN = None
                else:
                    X_hat_FNN = None

                # CNN
                if 'cnn' in locals() and cnn is not None:
                    cnn_res = predict_and_build_xhat(cnn, "CNN")
                    if cnn_res is not None:
                        X_hat_CNN, nmse = cnn_res
                        NMSE_test["CNN"][jj] += nmse
                    else:
                        X_hat_CNN = None
                else:
                    X_hat_CNN = None

                # RNN
                if 'rnn' in locals() and rnn is not None:
                    rnn_res = predict_and_build_xhat(rnn, "RNN")
                    if rnn_res is not None:
                        X_hat_RNN, nmse = rnn_res
                        NMSE_test["RNN"][jj] += nmse
                    else:
                        X_hat_RNN = None
                else:
                    X_hat_RNN = None

                # ---------- Equalizers (Perfect / LS / MMSE) ----------
                # Build LS pilot-based FR estimates first (same as baseline pipeline)
                X_LS = np.copy(Xf)
                X_LS[np.arange(1, len(X_LS), 2), 0] = 0
                X_LS[np.arange(0, len(X_LS), 2), 1] = 0
                x_LS_CP = np.zeros_like(x_CP, dtype=np.complex128)
                for iii in range(N_t):
                    x_temp = N * np.fft.ifft(X_LS[:, iii])
                    x_LS_CP[:, iii] = np.append(x_temp[(-1 - CyclicPrefixLen + 1): len(x_temp)], x_temp)
                    x_LS_CP[:, iii] = x_LS_CP[:, iii] * (Pi**0.5)
                Y_LS = np.zeros((N, N_r), dtype=np.complex128)
                y_LS_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=np.complex128)
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        y_LS_CP[:, nnn] += signal.lfilter(c[nnn][mmm], np.array([1]), x_LS_CP[:, mmm])
                    y_LS_CP[:, nnn] += math.sqrt(y_CP_NLD.shape[0] * No / 2) * np.matmul(
                        rng.normal(size=(y_CP_NLD.shape[0], 2)), np.array([[1], [1j]])).reshape(-1)
                    Y_LS[:, nnn] = (1/N) * np.fft.fft(y_LS_CP[IsiDuration-1:, nnn])
                Y_LS = Y_LS / (Pi**0.5)

                Ci_LS = [[None] * N_t for _ in range(N_r)]
                Ci_MMSE = [[None] * N_t for _ in range(N_r)]
                Ci_LS_Pilots = [[None] * N_t for _ in range(N_r)]
                for nnn in range(N_r):
                    for mmm in range(N_t):
                        Ci_LS_Pilots[nnn][mmm] = Y_LS[np.arange(mmm, len(Y_LS), 2), nnn] / X_LS[np.arange(mmm, len(X_LS), 2), mmm]
                        c_LS = np.fft.ifft(Ci_LS_Pilots[nnn][mmm])
                        c_LS = np.delete(c_LS, np.arange(IsiDuration, len(c_LS)))
                        c_MMSE = np.linalg.solve(MMSE_bold_TD, c_LS)
                        Ci_MMSE[nnn][mmm] = np.fft.fft(np.append(c_MMSE, np.zeros(N-IsiDuration)))
                        if mmm == 0:
                            tmpf = interpolate.interp1d(np.append(np.arange(mmm, N-1, N_t), N-1),
                                 np.append(Ci_LS_Pilots[nnn][mmm], Ci_LS_Pilots[nnn][mmm][-1]))
                            Ci_LS[nnn][mmm] = tmpf(np.arange(N))
                        else:
                            tmpf = interpolate.interp1d(np.append(0, np.arange(mmm, N, N_t)),
                                 np.append(Ci_LS_Pilots[nnn][mmm][0], Ci_LS_Pilots[nnn][mmm]))
                            Ci_LS[nnn][mmm] = tmpf(np.arange(N))

                # Solve equalization per-subcarrier
                X_hat_Perfect = np.zeros_like(Xf, dtype=np.complex128)
                X_hat_LS = np.zeros_like(Xf, dtype=np.complex128)
                X_hat_MMSE = np.zeros_like(Xf, dtype=np.complex128)
                for ii in range(N):
                    Y_temp = Y_NLD[ii, :].T
                    H_temp = np.zeros((N_r, N_t), dtype=np.complex128)
                    H_temp_LS = np.zeros((N_r, N_t), dtype=np.complex128)
                    H_temp_MMSE = np.zeros((N_r, N_t), dtype=np.complex128)
                    for nnn in range(N_r):
                        for mmm in range(N_t):
                            H_temp[nnn, mmm] = Ci[nnn][mmm][ii]
                            H_temp_LS[nnn, mmm] = Ci_LS[nnn][mmm][ii]
                            H_temp_MMSE[nnn, mmm] = Ci_MMSE[nnn][mmm][ii]
                    X_hat_Perfect[ii, :] = np.linalg.solve(H_temp, Y_temp) / math.sqrt(Pi)
                    X_hat_LS[ii, :] = np.linalg.solve(H_temp_LS, Y_temp) / math.sqrt(Pi)
                    X_hat_MMSE[ii, :] = np.linalg.solve(H_temp_MMSE, Y_temp) / math.sqrt(Pi)

                # -------------- Bit decisions & BER accumulation --------------
                def hard_bits_from_symbols(X_hat):
                    RxBits = np.zeros_like(TxBits)
                    for ii in range(N):
                        for iii in range(N_t):
                            qidx = np.argmin(np.abs(Const - X_hat[ii, iii]))
                            bits = list(format(qidx, 'b').zfill(m))
                            bits = np.array([int(b) for b in bits])[::-1]
                            RxBits[m*ii : m*(ii+1), iii] = bits
                    return RxBits

                # Perfect/LS/MMSE
                RxBits_Perfect = hard_bits_from_symbols(X_hat_Perfect)
                RxBits_LS = hard_bits_from_symbols(X_hat_LS)
                RxBits_MMSE = hard_bits_from_symbols(X_hat_MMSE)

                # ESN & learned models
                RxBits_ESN = hard_bits_from_symbols(X_hat_ESN)
                if X_hat_ELM is not None: RxBits_ELM = hard_bits_from_symbols(X_hat_ELM)
                if X_hat_FNN is not None: RxBits_FNN = hard_bits_from_symbols(X_hat_FNN)
                if X_hat_CNN is not None: RxBits_CNN = hard_bits_from_symbols(X_hat_CNN)
                if X_hat_RNN is not None: RxBits_RNN = hard_bits_from_symbols(X_hat_RNN)

                # Accumulate
                TotalBerNum["Perfect"] += np.sum(TxBits != RxBits_Perfect)
                TotalBerNum["LS"] += np.sum(TxBits != RxBits_LS)
                TotalBerNum["MMSE"] += np.sum(TxBits != RxBits_MMSE)
                TotalBerNum["ESN"] += np.sum(TxBits != RxBits_ESN)
                if X_hat_ELM is not None: TotalBerNum["ELM"] += np.sum(TxBits != RxBits_ELM)
                if X_hat_FNN is not None: TotalBerNum["FNN"] += np.sum(TxBits != RxBits_FNN)
                if X_hat_CNN is not None: TotalBerNum["CNN"] += np.sum(TxBits != RxBits_CNN)
                if X_hat_RNN is not None: TotalBerNum["RNN"] += np.sum(TxBits != RxBits_RNN)
                TotalBerDen += m * N * N_t

        # Store BER for this SNR
        for k in models:
            if TotalBerDen > 0:
                BER[k][jj] = TotalBerNum[k] / TotalBerDen

        # NMSE averages
        data_count = NumOfdmSymbols - math.ceil(NumOfdmSymbols / L)
        if data_count <= 0: data_count = 1
        for k in ["ESN", "ELM", "FNN", "CNN", "RNN"]:
            NMSE_test[k][jj] /= max(1, data_count)
        NMSE_train["ESN"][jj] /= max(1, math.ceil(NumOfdmSymbols / L))

    # -------- Save and Plot --------
    out_ber = {"EBN0": EbNoDB}
    for k in models:
        out_ber[k] = BER[k]
    with open("BER_compare.pkl", "wb") as f:
        pickle.dump(out_ber, f)

    semilogy_ber(EbNoDB,
                 {"Perfect": BER["Perfect"],
                  "LS": BER["LS"],
                  "MMSE": BER["MMSE"],
                  "ESN": BER["ESN"],
                  "ELM": BER["ELM"],
                  "FNN": BER["FNN"],
                  "CNN": BER["CNN"],
                  "RNN": BER["RNN"]},
                 "BER vs. Eb/N0 (2x2 OFDM MIMO, Nonlinear PA)",
                 "BER_compare.png")

    out_nmse = {"EBN0": EbNoDB, "train": NMSE_train, "test": NMSE_test}
    with open("NMSE_compare.pkl", "wb") as f:
        pickle.dump(out_nmse, f)

    print("[DONE] Saved: BER_compare.pkl, BER_compare.png, NMSE_compare.pkl")

def parse_args():
    ap = argparse.ArgumentParser()
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
