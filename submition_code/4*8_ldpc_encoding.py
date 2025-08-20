

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Dict

from pyESN import ESN
from HelpFunc import HelpFunc


# ----------------------
# Utilities: QAM mapping and LLRs
# ----------------------

def bits_to_indices(bits: np.ndarray, m: int) -> np.ndarray:
    """
    Convert a 1D bit array (LSB-first within each group of m) to constellation indices.
    bits length must be a multiple of m.
    """
    assert bits.ndim == 1
    assert len(bits) % m == 0
    B = bits.reshape(-1, m)  # shape (num_symbols, m)
    # LSB-first binary to integer index
    powers = (2 ** np.arange(m)).astype(int)
    idx = (B * powers).sum(axis=1).astype(int)
    return idx


def indices_to_bits(idx: np.ndarray, m: int) -> np.ndarray:
    """
    Inverse of bits_to_indices, following LSB-first bit order.
    """
    out = np.zeros((len(idx), m), dtype=int)
    for i, v in enumerate(idx.astype(int)):
        b = [int(x) for x in np.binary_repr(v, width=m)][::-1]  # LSB-first
        out[i, :] = b
    return out.reshape(-1)


def map_bits_to_qam(bits: np.ndarray, const: np.ndarray, m: int) -> np.ndarray:
    """
    Map bits to QAM symbols using given constellation array,
    with index determined by LSB-first binary coding.
    """
    idx = bits_to_indices(bits, m)
    return const[idx]


def llr_maxlog_per_symbol(z: np.ndarray, const: np.ndarray, m: int, sigma2: float) -> np.ndarray:
    """
    Max-log LLRs for each symbol z against constellation 'const' with LSB-first labeling.
    Returns shape (len(z), m).
    """
    M = len(const)
    # Precompute bit labels for the constellation indices (LSB-first)
    lab = np.array([list(np.binary_repr(i, width=m))[::-1] for i in range(M)], dtype=int)
    llrs = np.zeros((len(z), m))
    for b in range(m):
        S0 = const[lab[:, b] == 0]
        S1 = const[lab[:, b] == 1]
        d0 = np.min(np.abs(z[:, None] - S0[None, :]) ** 2, axis=1)
        d1 = np.min(np.abs(z[:, None] - S1[None, :]) ** 2, axis=1)
        llrs[:, b] = (d1 - d0) / max(1e-12, sigma2)
    return llrs


# ----------------------
# Utilities: LDPC (rate 1/2) construction, encode, decode (min-sum)
# ----------------------

def ldpc_build_systematic(n: int, rate: float = 0.5, dv: int = 3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple systematic LDPC parity-check matrix H = [A | I], with A sparse (column weight ~ dv).
    Returns (H, A). H shape ((n-k) x n), A shape ((n-k) x k), k = int(n*rate).
    This is not a standards-compliant LDPC, but sufficient for experimentation.

    Encoding: for message m (k,), parity p = A @ m mod 2, codeword c = [m, p].
    """
    rng = np.random.default_rng(seed)
    k = int(n * rate)
    r = n - k
    A = np.zeros((r, k), dtype=np.uint8)
    # Assign dv ones per column at random check positions
    for col in range(k):
        rows = rng.choice(r, size=dv, replace=False)
        A[rows, col] = 1
    # Ensure every check has at least degree 2 (heuristic cleanup)
    for row in range(r):
        if A[row].sum() < 2:
            # flip two random bits in this row
            cols = rng.choice(k, size=2, replace=False)
            A[row, cols] ^= 1
    # H = [A | I_r]
    I = np.eye(r, dtype=np.uint8)
    H = np.concatenate([A, I], axis=1)
    return H, A


def ldpc_encode_systematic(A: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Encode message m (k,) into codeword c (n,) for H=[A|I].
    n = k + r where r = A.shape[0].
    """
    m = m.astype(np.uint8)
    p = (A @ m) % 2
    return np.concatenate([m, p])


def ldpc_decode_min_sum(H: np.ndarray, llr: np.ndarray, max_iter: int = 25, damping: float = 0.0) -> Tuple[np.ndarray, bool]:
    """
    Min-sum LDPC decoder for H (r x n), channel LLRs 'llr' of length n.
    Returns (decoded_bits, success_flag).

    Notes:
      - Damping in [0,1) can improve stability: m_new = (1-d)*m_old + d*m_new
    """
    H = H.astype(np.uint8)
    r, n = H.shape
    # Build adjacency lists
    check_neighbors = [np.where(H[i] == 1)[0] for i in range(r)]
    var_neighbors = [np.where(H[:, j] == 1)[0] for j in range(n)]

    # Initialize messages
    # Variable-to-check messages: start with channel LLR
    v2c = {}
    c2v = {}
    for i in range(r):
        for j in check_neighbors[i]:
            v2c[(i, j)] = llr[j]
            c2v[(i, j)] = 0.0

    for it in range(max_iter):
        # Check node update (min-sum)
        for i in range(r):
            neigh = check_neighbors[i]
            # Gather incoming v2c messages
            msgs = np.array([v2c[(i, j)] for j in neigh])
            signs = np.sign(msgs)
            abs_msgs = np.abs(msgs)
            prod_sign = np.prod(signs + (signs == 0))  # treat zero sign as +1
            min1_idx = np.argmin(abs_msgs)
            min1 = abs_msgs[min1_idx]
            # For strictly correct MS, also need second minimum; simplified version uses only min1
            for t, j in enumerate(neigh):
                # message to var j: sign product excluding j times minimum excluding j
                # Using single-min approximation: use min among others;
                # we approximate by min1 unless j was the argmin (then use min among others)
                if t == min1_idx:
                    # compute second min
                    if len(abs_msgs) > 1:
                        min2 = np.min(np.delete(abs_msgs, t))
                        m = min2
                    else:
                        m = min1
                else:
                    m = min1
                s = prod_sign * np.sign(v2c[(i, j)] if v2c[(i, j)] != 0 else 1.0)
                new_msg = s * m
                if damping > 0.0:
                    c2v[(i, j)] = (1 - damping) * c2v[(i, j)] + damping * new_msg
                else:
                    c2v[(i, j)] = new_msg

        # Variable node update
        L_post = np.copy(llr)
        for j in range(n):
            for i in var_neighbors[j]:
                L_post[j] += c2v[(i, j)]

        # Hard decision
        x_hat = (L_post < 0).astype(np.uint8)  # bit=1 if LLR<0
        syndrome = (H @ x_hat) % 2
        if syndrome.sum() == 0:
            return x_hat, True

        # Prepare next v2c
        for j in range(n):
            for i in var_neighbors[j]:
                new_msg = L_post[j] - c2v[(i, j)]
                if damping > 0.0:
                    v2c[(i, j)] = (1 - damping) * v2c[(i, j)] + damping * new_msg
                else:
                    v2c[(i, j)] = new_msg

    # Final decision if not converged
    x_hat = (L_post < 0).astype(np.uint8)
    syndrome = (H @ x_hat) % 2
    return x_hat, bool(syndrome.sum() == 0)


# ----------------------
# ESN I/O builders for general N_t x N_r
# ----------------------

def estimate_delays_per_tx(y_mix: np.ndarray, x_cp: np.ndarray, max_delay: int) -> np.ndarray:
    """
    Rough delay estimate per TX stream using cross-correlation against a received mixture.
    y_mix: 1D real-valued reference (e.g., sum of RX real parts)
    x_cp: 2D time-domain TX CP signals, shape (T_eff, N_t)
    Returns integer delays per TX of length N_t, clipped to [0, max_delay].
    """
    N_t = x_cp.shape[1]
    d_tx = np.zeros(N_t, dtype=int)
    for tx in range(N_t):
        a = y_mix
        b = x_cp[:, tx].real
        r = np.correlate(a, b, mode='full')
        lag = np.argmax(r) - (len(b) - 1)
        d_tx[tx] = int(np.clip(lag, 0, max_delay))
    return d_tx


def build_esn_training_pairs(y_cp: np.ndarray, x_cp: np.ndarray, max_delay: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Build ESN input/output for training from pilot signals.
    y_cp: time-domain RX signals with CP, shape (T_eff, N_r), complex
    x_cp: time-domain TX signals with CP, shape (T_eff, N_t), complex
    Returns:
      ESN_input: shape (T_eff + max(d), 2*N_r) real-valued
      ESN_output: shape (T_eff + max(d), 2*N_t) real-valued with delayed targets
      d_tx: array of delays per TX (len N_t)
      delay_min, delay_max
    """
    T_eff, N_r = y_cp.shape
    _, N_t = x_cp.shape

    # Reference for delay search: sum of RX real parts
    y_mix = y_cp.real.sum(axis=1)
    d_tx = estimate_delays_per_tx(y_mix, x_cp, max_delay)

    delay_min = int(np.min(d_tx))
    delay_max = int(np.max(d_tx))

    T_total = T_eff + delay_max
    esn_in = np.zeros((T_total, 2 * N_r))
    for rx in range(N_r):
        esn_in[:, 2 * rx] = np.concatenate([y_cp[:, rx].real, np.zeros(delay_max)])
        esn_in[:, 2 * rx + 1] = np.concatenate([y_cp[:, rx].imag, np.zeros(delay_max)])

    esn_out = np.zeros((T_total, 2 * N_t))
    for tx in range(N_t):
        d = d_tx[tx]
        esn_out[d: d + T_eff, 2 * tx] = x_cp[:, tx].real
        esn_out[d: d + T_eff, 2 * tx + 1] = x_cp[:, tx].imag

    return esn_in, esn_out, d_tx, delay_min, delay_max


def esn_inference(trained_esn: ESN, y_cp: np.ndarray, d_tx: np.ndarray, delay_min: int, delay_max: int, N: int, Pi: float) -> np.ndarray:
    """
    Run ESN inference on data RX time-domain y_cp (with CP).
    Returns estimated frequency-domain symbols per TX: X_hat (N x N_t).
    """
    T_eff, N_r = y_cp.shape
    N_t = len(d_tx)
    # Build test input
    esn_in = np.zeros((N + delay_max + (T_eff - (N)), 2 * N_r))  # T_eff = N+CP
    esn_in[:, 0:2*N_r:2] = np.vstack([y_cp.real, np.zeros((delay_max, N_r))])
    esn_in[:, 1:2*N_r:2] = np.vstack([y_cp.imag, np.zeros((delay_max, N_r))])

    n_forget = delay_min + (T_eff - N)  # CP length + delay_min
    xhat_td_all = trained_esn.predict(esn_in, n_forget, continuation=False)  # shape (T_total - n_forget, 2*N_t)

    # Recover per-TX time-domain signals without CP
    x_td = np.zeros((N, N_t), dtype=np.complex128)
    for tx in range(N_t):
        re = xhat_td_all[d_tx[tx] - delay_min: d_tx[tx] - delay_min + N + 1, 2 * tx]
        im = xhat_td_all[d_tx[tx] - delay_min: d_tx[tx] - delay_min + N + 1, 2 * tx + 1]
        x_td[:, tx] = re[:N] + 1j * im[:N]

    # FFT and power normalization back to frequency domain
    X_hat = np.zeros((N, N_t), dtype=np.complex128)
    for tx in range(N_t):
        X_hat[:, tx] = (1 / N) * np.fft.fft(x_td[:, tx]) / math.sqrt(Pi)
    return X_hat


# ----------------------
# Main MIMO-OFDM + ESN + LDPC simulation
# ----------------------

def run(args):
    # ---------- Parameters ----------
    N_t = 4
    N_r = 8
    N = args.N                  # OFDM subcarriers
    m = args.m                  # bits per QAM symbol (default 4 for 16-QAM)
    m_pilot = args.m_pilot
    IsiDuration = args.isi
    W = args.W
    f_D = args.fD
    No = args.No
    EbNoDB = np.array([args.snr_db], dtype=float)  # single SNR run for brevity
    NumOfdmSymbols = args.num_symbols
    p_smooth = 1
    ClipLeveldB = args.clip_db
    rate = 0.5  # LDPC rate
    dv = args.ldpc_dv
    max_bp_iter = args.ldpc_iter

    Subcarrier_Spacing = W / N
    T_OFDM = N / W
    T_OFDM_Total = (N + IsiDuration - 1) / W
    tau_c = 0.5 / f_D
    L = max(2, int(math.floor(tau_c / T_OFDM_Total)))  # coherence length in OFDM symbols

    # Constellations
    Const = HelpFunc.UnitQamConstellation(m)
    ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)

    PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
    PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))

    CyclicPrefixLen = IsiDuration - 1

    # One-sided exponential power delay profile (normalized)
    temp = CyclicPrefixLen / 9
    IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen + 1)) / temp)
    IsiMagnitude = IsiMagnitude / np.sum(IsiMagnitude)

    rng = np.random.default_rng(args.seed)

    # ESN hyperparameters
    nInputUnits = 2 * N_r
    nOutputUnits = 2 * N_t
    nInternalUnits = args.reservoir
    spectralRadius = args.spectral
    inputScaler_base = args.inpscale
    inputOffset = 0.0
    teacherScaling_base = args.teacher_scale
    feedbackScaler = 0.0
    sparsity = args.sparsity

    # LDPC block length matches bits per OFDM symbol per TX
    n_bits_per_symbol = m * N
    k_bits = int(n_bits_per_symbol * rate)
    n_bits = n_bits_per_symbol
    H, A = ldpc_build_systematic(n_bits, rate=rate, dv=dv, seed=args.seed)

    # Storage
    ber_tx = np.zeros((N_t,), dtype=float)

    for eb in EbNoDB:
        Pi = (10 ** (eb / 10)) * No  # per-subcarrier power
        var_x = (10 ** (eb / 10)) * No * N
        A_Clip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB / 20)

        # ESN scaling vectors depend on SNR
        inputScaling_vec = inputScaler_base / (var_x ** 0.5) * np.ones(nInputUnits)
        inputShift_vec = inputOffset * np.ones(nInputUnits)
        teacherScaling_vec = teacherScaling_base * np.ones(nOutputUnits)
        teacherShift_vec = np.zeros(nOutputUnits)
        feedbackScaling_vec = feedbackScaler * np.ones(nOutputUnits)

        # === Main loop over OFDM symbols ===
        symbol_idx = 0
        while symbol_idx < NumOfdmSymbols:
            # Draw a new channel every coherence block
            c = [[None] * N_t for _ in range(N_r)]
            for nr in range(N_r):
                for nt in range(N_t):
                    taps = rng.normal(size=IsiDuration) / np.sqrt(2) + 1j * rng.normal(size=IsiDuration) / np.sqrt(2)
                    c[nr][nt] = taps * (IsiMagnitude ** 0.5)

            # --------- Pilot (ESN training) ---------
            # Pilot bits and mapping per TX
            TxBits_pilot = (rng.random(size=(N * m_pilot, N_t)) > 0.5).astype(np.int32)
            X_pilot = np.zeros((N, N_t), dtype=np.complex128)
            x_cp_pilot = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int(np.matmul(PowersOfTwoPilot[:, :m_pilot], TxBits_pilot[m_pilot * ii + np.arange(m_pilot), tx])[0])
                    X_pilot[ii, tx] = ConstPilot[idx]
            for tx in range(N_t):
                x_temp = N * np.fft.ifft(X_pilot[:, tx])
                x_cp_pilot[:, tx] = np.concatenate([x_temp[-CyclicPrefixLen:], x_temp])
                x_cp_pilot[:, tx] = x_cp_pilot[:, tx] * (Pi ** 0.5)

            # PA nonlinearity
            x_cp_pilot_nld = x_cp_pilot / ((1 + (np.abs(x_cp_pilot) / A_Clip) ** (2 * p_smooth)) ** (1 / (2 * p_smooth)))

            # Pass through MIMO channel + AWGN
            y_cp_pilot = np.zeros((N + CyclicPrefixLen, N_r), dtype=np.complex128)
            for nr in range(N_r):
                for nt in range(N_t):
                    y_cp_pilot[:, nr] += signal.lfilter(c[nr][nt], [1.0], x_cp_pilot_nld[:, nt])
                noise = math.sqrt((N + CyclicPrefixLen) * No / 2) * (rng.normal(size=(N + CyclicPrefixLen,)) + 1j * rng.normal(size=(N + CyclicPrefixLen,)))
                y_cp_pilot[:, nr] += noise

            # Build ESN training pairs and train
            max_delay = int(math.ceil(IsiDuration / 2) + 2)
            esn_in, esn_out, d_tx, delay_min, delay_max = build_esn_training_pairs(y_cp_pilot, x_cp_pilot, max_delay)
            n_forget = delay_min + CyclicPrefixLen

            esn = ESN(
                n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                spectral_radius=spectralRadius, sparsity=sparsity, noise=0.001,
                input_shift=inputShift_vec, input_scaling=inputScaling_vec,
                teacher_forcing=True, feedback_scaling=feedbackScaling_vec,
                teacher_scaling=teacherScaling_vec, teacher_shift=teacherShift_vec,
                random_state=args.seed, silent=True
            )
            esn.fit(esn_in, esn_out, transient=n_forget)

            # --------- Next: Data symbols with LDPC ---------
            data_symbols_in_block = min(L - 1, NumOfdmSymbols - 1 - symbol_idx)
            for _ in range(data_symbols_in_block):
                # LDPC encoding per TX
                info_bits_tx = np.zeros((k_bits, N_t), dtype=np.uint8)
                code_bits_tx = np.zeros((n_bits, N_t), dtype=np.uint8)
                for tx in range(N_t):
                    m_bits = (rng.random(size=(k_bits,)) > 0.5).astype(np.uint8)
                    c_bits = ldpc_encode_systematic(A, m_bits)
                    info_bits_tx[:, tx] = m_bits
                    code_bits_tx[:, tx] = c_bits

                # Map to QAM
                X = np.zeros((N, N_t), dtype=np.complex128)
                for tx in range(N_t):
                    X[:, tx] = map_bits_to_qam(code_bits_tx[:, tx], Const, m)

                # OFDM mod + CP + power
                x_cp = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)
                for tx in range(N_t):
                    x_temp = N * np.fft.ifft(X[:, tx])
                    x_cp[:, tx] = np.concatenate([x_temp[-CyclicPrefixLen:], x_temp])
                    x_cp[:, tx] = x_cp[:, tx] * (Pi ** 0.5)

                # PA nonlinearity
                x_cp_nld = x_cp / ((1 + (np.abs(x_cp) / A_Clip) ** (2 * p_smooth)) ** (1 / (2 * p_smooth)))

                # Channel + AWGN
                y_cp = np.zeros((N + CyclicPrefixLen, N_r), dtype=np.complex128)
                Y_fd = np.zeros((N, N_r), dtype=np.complex128)
                for nr in range(N_r):
                    for nt in range(N_t):
                        y_cp[:, nr] += signal.lfilter(c[nr][nt], [1.0], x_cp_nld[:, nt])
                    noise = math.sqrt((N + CyclicPrefixLen) * No / 2) * (rng.normal(size=(N + CyclicPrefixLen,)) + 1j * rng.normal(size=(N + CyclicPrefixLen,)))
                    y_cp[:, nr] += noise
                    Y_fd[:, nr] = (1 / N) * np.fft.fft(y_cp[IsiDuration - 1:, nr])

                # ESN inference to recover per-TX symbols
                X_hat = esn_inference(esn, y_cp, d_tx, delay_min, delay_max, N, Pi)  # shape (N, N_t)

                # LLRs per TX stream (per subcarrier symbol)
                # Estimate effective noise variance for demapper (rough)
                # Here we use a simple fixed value based on No; for better results, estimate from residuals.
                sigma2 = No
                llrs_all = np.zeros((n_bits, N_t))
                for tx in range(N_t):
                    llr_sym = llr_maxlog_per_symbol(X_hat[:, tx], Const, m, sigma2)  # (N, m)
                    llrs_all[:, tx] = llr_sym.reshape(-1)

                # LDPC decode and BER
                for tx in range(N_t):
                    # Channel LLRs correspond to code bits; we pass them directly
                    decoded, ok = ldpc_decode_min_sum(H, llrs_all[:, tx], max_iter=max_bp_iter, damping=0.2)
                    # Extract first k_bits as info bits (systematic code: c=[m|p])
                    m_hat = decoded[:k_bits]
                    ber_tx[tx] += np.sum(m_hat != info_bits_tx[:, tx])

                symbol_idx += 1
                if symbol_idx >= NumOfdmSymbols:
                    break
            symbol_idx += 1  # count the pilot too

    # Normalize BER per TX
    denom = k_bits * (NumOfdmSymbols - NumOfdmSymbols // L)  # approx number of data symbols per TX
    denom = max(1, denom)
    ber_tx = ber_tx / denom

    # Report
    for tx in range(N_t):
        print(f"TX{tx}: LDPC-coded BER = {ber_tx[tx]:.4e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256, help="Number of OFDM subcarriers (try 512 for full runs)")
    ap.add_argument("--m", type=int, default=4, help="Bits per QAM symbol (e.g., 4 for 16-QAM)")
    ap.add_argument("--m_pilot", type=int, default=4, help="Bits per QAM symbol for pilot")
    ap.add_argument("--isi", type=int, default=8, help="Number of channel taps (including tap0). CP = isi-1")
    ap.add_argument("--W", type=float, default=2*1.024e6, help="Sampling rate (Hz)")
    ap.add_argument("--fD", type=float, default=100.0, help="Doppler (Hz)")
    ap.add_argument("--No", type=float, default=1e-5, help="Noise spectral density")
    ap.add_argument("--snr_db", type=float, default=18.0, help="Eb/N0 in dB")
    ap.add_argument("--num_symbols", type=int, default=40, help="Total OFDM symbols to simulate")
    ap.add_argument("--clip_db", type=float, default=3.0, help="PA clipping level in dB relative to average")
    # ESN
    ap.add_argument("--reservoir", type=int, default=100, help="ESN reservoir size")
    ap.add_argument("--spectral", type=float, default=0.9, help="ESN spectral radius")
    ap.add_argument("--inpscale", type=float, default=0.005, help="ESN input scaling base")
    ap.add_argument("--teacher_scale", type=float, default=5e-7, help="ESN teacher scaling base")
    ap.add_argument("--sparsity", type=float, default=0.2, help="ESN reservoir sparsity fraction")
    # LDPC
    ap.add_argument("--ldpc_dv", type=int, default=3, help="LDPC column weight for A (sparsity)")
    ap.add_argument("--ldpc_iter", type=int, default=25, help="LDPC min-sum max iterations")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
