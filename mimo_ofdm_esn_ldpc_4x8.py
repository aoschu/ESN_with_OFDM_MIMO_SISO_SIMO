
import math
import argparse
import numpy as np
from scipy import signal, interpolate
from typing import Tuple

# Constellation helper (consistent with your codebase)
from HelpFunc import HelpFunc


# ----------------------
# QAM mapping and LLRs
# ----------------------

def indices_to_bits(idx: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros((len(idx), m), dtype=int)
    for i, v in enumerate(idx.astype(int)):
        b = [int(x) for x in np.binary_repr(v, width=m)][::-1]  # LSB-first
        out[i, :] = b
    return out.reshape(-1)


def bits_to_indices(bits: np.ndarray, m: int) -> np.ndarray:
    assert bits.ndim == 1 and (len(bits) % m == 0)
    B = bits.reshape(-1, m)
    powers = (2 ** np.arange(m)).astype(int)  # LSB-first
    return (B * powers).sum(axis=1).astype(int)


def map_bits_to_qam(bits: np.ndarray, const: np.ndarray, m: int) -> np.ndarray:
    return const[bits_to_indices(bits, m)]


def llr_maxlog_per_symbol(z: np.ndarray, const: np.ndarray, m: int, sigma2: float) -> np.ndarray:
    """
    Max-log LLRs for each complex symbol z against constellation 'const' with LSB-first indexing.
    Returns (len(z), m).
    """
    M = len(const)
    labels = np.array([list(np.binary_repr(i, width=m))[::-1] for i in range(M)], dtype=int)
    llrs = np.zeros((len(z), m))
    for b in range(m):
        S0 = const[labels[:, b] == 0]
        S1 = const[labels[:, b] == 1]
        d0 = np.min(np.abs(z[:, None] - S0[None, :]) ** 2, axis=1)
        d1 = np.min(np.abs(z[:, None] - S1[None, :]) ** 2, axis=1)
        llrs[:, b] = (d1 - d0) / max(1e-12, sigma2)
    return llrs


# ----------------------
# LDPC (simple systematic, rate 1/2) + min-sum BP
# ----------------------

def ldpc_build_systematic(n: int, rate: float = 0.5, dv: int = 3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build parity-check H = [A | I_r] with column-weight~dv (random), n total bits, k=n*rate.
    Returns (H, A). Encoding: c = [m | (A @ m mod 2)].
    """
    rng = np.random.default_rng(seed)
    k = int(n * rate)
    r = n - k
    A = np.zeros((r, k), dtype=np.uint8)
    for col in range(k):
        rows = rng.choice(r, size=min(dv, r), replace=False)
        A[rows, col] = 1
    # Ensure each row has degree >= 2
    for row in range(r):
        if A[row].sum() < 2 and k >= 2:
            cols = rng.choice(k, size=2, replace=False)
            A[row, cols] ^= 1
    H = np.concatenate([A, np.eye(r, dtype=np.uint8)], axis=1)
    return H, A


def ldpc_encode_systematic(A: np.ndarray, m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    p = (A @ m) % 2
    return np.concatenate([m, p])


def ldpc_decode_min_sum(H: np.ndarray, llr: np.ndarray, max_iter: int = 30, damping: float = 0.15) -> Tuple[np.ndarray, bool]:
    """
    Min-sum LDPC decoder. H (r x n), llr (n,).
    Returns (decoded_bits, success_flag).
    """
    H = H.astype(np.uint8)
    r, n = H.shape
    check_neighbors = [np.where(H[i] == 1)[0] for i in range(r)]
    var_neighbors = [np.where(H[:, j] == 1)[0] for j in range(n)]
    # init
    v2c = {(i, j): llr[j] for i in range(r) for j in check_neighbors[i]}
    c2v = {(i, j): 0.0 for i in range(r) for j in check_neighbors[i]}

    for _ in range(max_iter):
        # check update
        for i in range(r):
            neigh = check_neighbors[i]
            msgs = np.array([v2c[(i, j)] for j in neigh])
            signs = np.sign(msgs); signs[signs == 0] = 1.0
            abs_msgs = np.abs(msgs)
            prod_sign = np.prod(signs)
            # find min1, min2
            if len(abs_msgs) == 0:
                continue
            min1_idx = np.argmin(abs_msgs)
            min1 = abs_msgs[min1_idx]
            min2 = np.min(np.delete(abs_msgs, min1_idx)) if len(abs_msgs) > 1 else min1
            for t, j in enumerate(neigh):
                m = min2 if t == min1_idx else min1
                s = prod_sign * np.sign(v2c[(i, j)])
                new_msg = s * m
                c2v[(i, j)] = (1 - damping) * c2v[(i, j)] + damping * new_msg

        # variable update
        L_post = np.copy(llr)
        for j in range(n):
            for i in var_neighbors[j]:
                L_post[j] += c2v[(i, j)]
        x_hat = (L_post < 0).astype(np.uint8)
        if np.all((H @ x_hat) % 2 == 0):
            return x_hat, True

        # next v2c
        for j in range(n):
            for i in var_neighbors[j]:
                new_msg = L_post[j] - c2v[(i, j)]
                v2c[(i, j)] = (1 - damping) * v2c[(i, j)] + damping * new_msg

    x_hat = (L_post < 0).astype(np.uint8)
    return x_hat, bool(np.all((H @ x_hat) % 2 == 0))


# ----------------------
# Channel, OFDM, Pilots
# ----------------------

def exp_pdp_taps(IsiDuration: int, cp_len: int) -> np.ndarray:
    temp = cp_len / 9.0
    pdp = np.exp(-np.arange(cp_len + 1) / temp)
    pdp /= pdp.sum()
    taps = np.sqrt(pdp[:IsiDuration])  # amplitude weighting
    return taps


def draw_mimo_channel(N_r: int, N_t: int, IsiDuration: int, taps_amp: np.ndarray, rng) -> np.ndarray:
    """
    Returns c[nr, nt, L] complex time-domain taps.
    """
    c = np.zeros((N_r, N_t, IsiDuration), dtype=np.complex128)
    for nr in range(N_r):
        for nt in range(N_t):
            w = (rng.normal(size=IsiDuration) + 1j * rng.normal(size=IsiDuration)) / np.sqrt(2.0)
            c[nr, nt, :] = w * taps_amp
    return c


def ofdm_tx_from_symbols(X: np.ndarray, Pi: float, cp_len: int) -> np.ndarray:
    """
    X: (N, N_t) frequency-domain symbols.
    Return time-domain with CP: (N+cp_len, N_t)
    """
    N, N_t = X.shape
    x_cp = np.zeros((N + cp_len, N_t), dtype=np.complex128)
    for tx in range(N_t):
        x_td = N * np.fft.ifft(X[:, tx])
        x_cp[:, tx] = np.concatenate([x_td[-cp_len:], x_td]) * (Pi ** 0.5)
    return x_cp


def pass_through_channel(x_cp: np.ndarray, c: np.ndarray, No: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_cp: (N+cp_len, N_t), c: (N_r, N_t, L)
    Returns (y_cp, Y_fd) with shapes (N+cp_len, N_r), (N, N_r) after CP removal + FFT.
    """
    T, N_t = x_cp.shape
    N_r, _, L = c.shape
    cp_len = T - int(T * 0 + 1)  # dummy to keep variable

    y_cp = np.zeros((T, N_r), dtype=np.complex128)
    for nr in range(N_r):
        for nt in range(N_t):
            y_cp[:, nr] += signal.lfilter(c[nr, nt, :], [1.0], x_cp[:, nt])
        noise = math.sqrt(T * No / 2.0) * (rng.normal(size=T) + 1j * rng.normal(size=T))
        y_cp[:, nr] += noise

    # CP removal + FFT
    N = T - (L - 1)  # not accurate; fix below with explicit cp_len from caller
    return y_cp, None  # Y_fd computed by caller (needs cp_len info)


def build_pilots_X(N: int, N_t: int, m_pilot: int, const_pilot: np.ndarray, pattern: str = "comb") -> np.ndarray:
    """
    Build orthogonal comb pilots across TX streams:
      TX k uses subcarriers i with i % N_t == k, others are zero there.
    Pilot symbols are all-ones from the constellation (or random).
    """
    Xp = np.zeros((N, N_t), dtype=np.complex128)
    # Choose a fixed pilot symbol (e.g., constellation[0]) for stability
    sym = const_pilot[0]
    for k in range(N_t):
        Xp[k::N_t, k] = sym
    return Xp


def ls_channel_estimate(Yp: np.ndarray, Xp: np.ndarray, N_r: int, N_t: int) -> np.ndarray:
    """
    LS H estimate on pilot tones, with linear interpolation across frequency.
    Returns H_hat of shape (N, N_r, N_t).
    """
    N = Yp.shape[0]
    H_hat = np.zeros((N, N_r, N_t), dtype=np.complex128)
    for nr in range(N_r):
        for nt in range(N_t):
            pilot_idx = np.arange(nt, N, N_t)
            # Avoid division by zero
            denom = Xp[pilot_idx, nt]
            denom[np.abs(denom) < 1e-12] = 1e-12
            Hp = Yp[pilot_idx, nr] / denom  # LS at pilot tones
            # Interpolate over frequency
            if len(pilot_idx) == 1:
                H_hat[:, nr, nt] = Hp[0]
            else:
                f = interpolate.interp1d(pilot_idx, Hp, kind="linear", fill_value="extrapolate", bounds_error=False)
                H_hat[:, nr, nt] = f(np.arange(N))
    return H_hat


# ----------------------
# Receiver: MMSE + LLR demapper
# ----------------------

def mmse_equalize_per_subcarrier(Y: np.ndarray, H: np.ndarray, No: float, Es: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each subcarrier i, compute linear MMSE estimate z = W Y with
      W = (H^H H + (No/Es) I)^-1 H^H
    Returns:
      Z: (N, N_t) equalized symbols per stream
      sigma2_eff: (N, N_t) post-equalization noise variance per stream, approx sigma^2 * ||w_i||^2
    """
    N, N_r = Y.shape
    _, _, N_t = H.shape
    Z = np.zeros((N, N_t), dtype=np.complex128)
    sigma2_eff = np.zeros((N, N_t), dtype=float)
    reg = No / max(1e-12, Es)
    I = np.eye(N_t, dtype=np.complex128)
    for i in range(N):
        Hi = H[i, :, :]  # (N_r, N_t)
        G = Hi.conj().T @ Hi + reg * I
        W = np.linalg.solve(G, Hi.conj().T)  # (N_t, N_r)
        Zi = W @ Y[i, :]
        Z[i, :] = Zi
        # per-stream noise variance approx
        Wi = W  # rows are stream equalizers
        sigma2_eff[i, :] = No * np.sum(np.abs(Wi) ** 2, axis=1).real
    return Z, sigma2_eff


# ----------------------
# Main simulation
# ----------------------

def run(args):
    rng = np.random.default_rng(args.seed)

    # System dims
    N_t = 4
    N_r = 8
    N = args.N
    m = args.m
    m_pilot = args.m_pilot
    IsiDuration = args.isi
    cp_len = IsiDuration - 1

    # Channel/time
    W = args.W
    f_D = args.fD
    No = args.No
    EbN0_dB = args.snr_db
    NumOfdmSymbols = args.num_symbols

    # Power per subcarrier and overall (matching your convention)
    Pi = (10 ** (EbN0_dB / 10.0)) * No  # per-subcarrier signal power
    var_x = (10 ** (EbN0_dB / 10.0)) * No * N  # total block power (for legacy scaling)

    # OFDM timing + coherence (block fading)
    T_OFDM = N / W
    T_OFDM_Total = (N + cp_len) / W
    tau_c = 0.5 / f_D
    L = max(2, int(math.floor(tau_c / T_OFDM_Total)))  # symbols per coherence block

    # Constellations
    Const = HelpFunc.UnitQamConstellation(m)
    ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)

    # LDPC parameters per TX per OFDM symbol
    n_bits = N * m
    k_bits = n_bits // 2
    H, A = ldpc_build_systematic(n_bits, rate=0.5, dv=args.ldpc_dv, seed=args.seed)

    # BER counters
    total_err_bits = np.zeros((N_t,), dtype=np.int64)
    total_info_bits = 0

    # PDP amplitude profile
    taps_amp = exp_pdp_taps(IsiDuration, cp_len)

    sym_idx = 0
    while sym_idx < NumOfdmSymbols:
        # Draw a new channel every coherence block
        c = draw_mimo_channel(N_r, N_t, IsiDuration, taps_amp, rng)

        # ---- Pilot symbol for channel estimation ----
        Xp = build_pilots_X(N, N_t, m_pilot, ConstPilot, pattern="comb")
        x_cp_pilot = ofdm_tx_from_symbols(Xp, Pi, cp_len)
        # Channel + noise on pilot
        y_cp_pilot = np.zeros((N + cp_len, N_r), dtype=np.complex128)
        for nr in range(N_r):
            for nt in range(N_t):
                y_cp_pilot[:, nr] += signal.lfilter(c[nr, nt, :], [1.0], x_cp_pilot[:, nt])
            noise = math.sqrt((N + cp_len) * No / 2.0) * (rng.normal(size=N + cp_len) + 1j * rng.normal(size=N + cp_len))
            y_cp_pilot[:, nr] += noise
        # CP removal + FFT
        Yp = np.zeros((N, N_r), dtype=np.complex128)
        for nr in range(N_r):
            Yp[:, nr] = (1.0 / N) * np.fft.fft(y_cp_pilot[cp_len:, nr])

        # LS + interpolation to get H_hat(f)
        H_hat = ls_channel_estimate(Yp, Xp, N_r, N_t)  # (N, N_r, N_t)

        # ---- Data symbols within the same coherence block ----
        data_in_block = min(L - 1, NumOfdmSymbols - sym_idx - 1)
        for _ in range(data_in_block):
            # LDPC encode per TX
            info_bits = (rng.random(size=(k_bits, N_t)) > 0.5).astype(np.uint8)
            code_bits = np.zeros((n_bits, N_t), dtype=np.uint8)
            for tx in range(N_t):
                code_bits[:, tx] = ldpc_encode_systematic(A, info_bits[:, tx])
            total_info_bits += k_bits * N_t

            # Map to QAM and OFDM
            X = np.zeros((N, N_t), dtype=np.complex128)
            for tx in range(N_t):
                X[:, tx] = map_bits_to_qam(code_bits[:, tx], Const, m)  # (N,)

            x_cp = ofdm_tx_from_symbols(X, Pi, cp_len)

            # Channel + AWGN
            y_cp = np.zeros((N + cp_len, N_r), dtype=np.complex128)
            for nr in range(N_r):
                for nt in range(N_t):
                    y_cp[:, nr] += signal.lfilter(c[nr, nt, :], [1.0], x_cp[:, nt])
                noise = math.sqrt((N + cp_len) * No / 2.0) * (rng.normal(size=N + cp_len) + 1j * rng.normal(size=N + cp_len))
                y_cp[:, nr] += noise

            # Remove CP + FFT
            Y = np.zeros((N, N_r), dtype=np.complex128)
            for nr in range(N_r):
                Y[:, nr] = (1.0 / N) * np.fft.fft(y_cp[cp_len:, nr])

            # Linear MMSE equalization using H_hat
            Z, sigma2_eff = mmse_equalize_per_subcarrier(Y, H_hat, No, Pi)

            # Max-log demapper to bit LLRs per TX
            llrs = np.zeros((n_bits, N_t))
            for tx in range(N_t):
                # Per-subcarrier variance vector -> repeat for m bits each
                llr_sym = llr_maxlog_per_symbol(Z[:, tx], Const, m, sigma2_eff[:, tx].mean())
                llrs[:, tx] = llr_sym.reshape(-1)

            # LDPC decoding and BER
            for tx in range(N_t):
                decoded, ok = ldpc_decode_min_sum(H, llrs[:, tx], max_iter=args.ldpc_iter, damping=0.15)
                m_hat = decoded[:k_bits]
                total_err_bits[tx] += np.sum(m_hat != info_bits[:, tx])

            sym_idx += 1
            if sym_idx >= NumOfdmSymbols:
                break

        # count the pilot
        sym_idx += 1

    # Report
    ber_tx = total_err_bits / max(1, total_info_bits)
    print("=== LDPC-coded BER (MMSE, 4x8 MIMO-OFDM) ===")
    for tx in range(N_t):
        print(f"TX{tx}: {ber_tx[tx]:.6e}")
    print(f"Average BER: {ber_tx.mean():.6e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256, help="OFDM subcarriers (try 512 for full runs)")
    ap.add_argument("--m", type=int, default=4, help="Bits per QAM symbol (e.g., 4 for 16-QAM)")
    ap.add_argument("--m_pilot", type=int, default=4, help="Bits per QAM symbol used to define pilot symbol")
    ap.add_argument("--isi", type=int, default=8, help="Channel taps (CP length = isi-1)")
    ap.add_argument("--W", type=float, default=2*1.024e6, help="Sampling rate (Hz)")
    ap.add_argument("--fD", type=float, default=100.0, help="Doppler (Hz) for coherence length")
    ap.add_argument("--No", type=float, default=1e-5, help="Noise spectral density")
    ap.add_argument("--snr_db", type=float, default=18.0, help="Eb/N0 in dB (single value)")
    ap.add_argument("--num_symbols", type=int, default=40, help="Total OFDM symbols to simulate (includes pilots)")
    ap.add_argument("--ldpc_dv", type=int, default=3, help="LDPC column weight for A")
    ap.add_argument("--ldpc_iter", type=int, default=30, help="LDPC min-sum iterations")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
