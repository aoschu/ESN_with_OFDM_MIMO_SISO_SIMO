# helper_mimo_esn_generic.py
# Generic ESN trainer for arbitrary N_t x N_r MIMO.
# Stacks ALL Rx branches (Re/Im) as ESN inputs and ALL Tx branches (Re/Im) as teacher outputs.
# Returns the same tuple layout as your original HelpFunc.trainMIMOESN.

import numpy as np

def trainMIMOESN_generic(esn, DelayFlag, Min_Delay, Max_Delay,
                         CyclicPrefixLen, N, N_t, N_r, IsiDuration,
                         y_CP, x_CP):
    """Generic ESN trainer for MIMO-OFDM.

    Args:
        esn: pyESN.ESN instance (already initialized).
        DelayFlag: if 0, uses a single shared delay; if nonzero, scans delays in [Min_Delay, Max_Delay].
        Min_Delay, Max_Delay: integer delay bounds.
        CyclicPrefixLen: CP length in samples.
        N: FFT size (subcarriers).
        N_t: number of transmit antennas.
        N_r: number of receive antennas.
        IsiDuration: channel memory length (taps).
        y_CP: complex array of shape (N+CP, N_r), received pilot TD per Rx.
        x_CP: complex array of shape (N+CP, N_t), transmitted pilot TD per Tx.

    Returns:
        [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Minn, Delay_Maxx, nForgetPoints, NMSE_ESN]
    """

    def build_io_for_delay(d):
        """Build ESN input/output for a shared delay d (all outputs share same delay)."""
        T = N + CyclicPrefixLen
        T_pad = T + d
        X_in = np.zeros((T_pad, 2*N_r), dtype=float)
        for rx in range(N_r):
            X_in[:, 2*rx]   = np.r_[y_CP[:, rx].real, np.zeros(d)]
            X_in[:, 2*rx+1] = np.r_[y_CP[:, rx].imag, np.zeros(d)]
        X_out = np.zeros((T_pad, 2*N_t), dtype=float)
        for tx in range(N_t):
            X_out[d:d+T, 2*tx]   = x_CP[:, tx].real
            X_out[d:d+T, 2*tx+1] = x_CP[:, tx].imag
        return X_in, X_out

    def nmse_for_delay(d):
        """Train/predict at delay d and compute NMSE w.r.t. x_CP (tail aligns like original)."""
        X_in, X_out = build_io_for_delay(d)
        nForget = d + CyclicPrefixLen
        esn.fit(X_in, X_out, nForget)
        pred = esn.predict(X_in, nForget, continuation=False)

        nmse_sum = 0.0
        for tx in range(N_t):
            re_seq = pred[d : d + N + 1, 2*tx]
            im_seq = pred[d : d + N + 1, 2*tx+1]
            x_hat = re_seq + 1j*im_seq
            x_true = x_CP[IsiDuration-1:, tx]
            M = min(len(x_hat), len(x_true))
            if M > 0:
                nmse_sum += np.linalg.norm(x_hat[:M] - x_true[:M])**2 / (np.linalg.norm(x_true[:M])**2 + 1e-12)
        return nmse_sum, X_in, X_out, nForget

    # Choose delay
    if DelayFlag == 0:
        d = int((Min_Delay + Max_Delay)//2)
        nmse, ESN_input, ESN_output, nForgetPoints = nmse_for_delay(d)
        Delay = np.full(2*N_t, d, dtype=int)
        Delay_Idx = d - Min_Delay
        Delay_Minn = d
        Delay_Maxx = d
        NMSE_ESN = float(nmse)
    else:
        best_nmse = 1e9
        best_tuple = None
        Delay_Idx = 0
        for d in range(Min_Delay, Max_Delay+1):
            nmse, Xin, Xout, nF = nmse_for_delay(d)
            if nmse < best_nmse:
                best_nmse = nmse
                best_tuple = (Xin, Xout, nF, d)
                Delay_Idx = d - Min_Delay
        ESN_input, ESN_output, nForgetPoints, d_best = best_tuple
        Delay = np.full(2*N_t, int(d_best), dtype=int)
        Delay_Minn = int(d_best)
        Delay_Maxx = int(d_best)
        NMSE_ESN = float(best_nmse)

    # Final fit on chosen delay
    esn.fit(ESN_input, ESN_output, nForgetPoints)

    return [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Minn, Delay_Maxx, nForgetPoints, NMSE_ESN]
