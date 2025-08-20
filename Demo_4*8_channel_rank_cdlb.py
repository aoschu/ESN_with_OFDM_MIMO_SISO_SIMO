# capacity_rank_cdlb.py
# -----------------------------------------------
# 4x8 MIMO, CDL-B (TDL equivalent), OFDM size N.
# Computes:
#   - Avg. per-subcarrier MIMO capacity vs Eb/N0
#   - Fraction of subcarriers with "usable rank >= 4"
#     where a stream is "usable" if (snr/Nt)*s_i^2 >= gamma_th
# Saves PNG and PKL to OUT_DIR.
# -----------------------------------------------

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# ========== Config ==========
Nt, Nr = 4, 8
N = 512                    # subcarriers (FFT size for H[k])
W_Hz = 2*1.024e6           # sampling rate
ISI_TAPS = 16              # discrete taps kept for TDL
DS_NS = 300.0              # delay spread (ns)
SEED = 1234

EbNo_dB = np.arange(-10, 31, 3)         # 0..30 dB, step 3
gamma_th = 1.0                         # SNR threshold (linear) for "usable" stream
OUT_DIR = "./results_channel_capacity_rank"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== CDL-B (TDL) profile (38.901 Table 7.7.2-2) ==========
TDLB_NORM_DELAYS = np.array([
    0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681,
    0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294,
    3.0219, 3.6187, 4.1067, 4.2790, 4.7834
], dtype=float)
TDLB_POW_DB = np.array([
     0.0,  -2.2,  -4.0,  -3.2,  -9.8,  -1.2,  -3.4,  -5.2,  -7.6,
    -3.0,  -8.9,  -9.0,  -4.8,  -5.7,  -7.5,  -1.9,  -7.6, -12.2,
    -9.8, -11.4, -14.9,  -9.2, -11.3
], dtype=float)

# ========== Helpers ==========
def gen_tdlb_impulse(L, fs_Hz, DS_ns, rng):
    """One complex impulse response (TDL-B), length L."""
    pow_lin = 10.0**(TDLB_POW_DB/10.0)
    pow_lin /= np.sum(pow_lin)

    delays_s = (TDLB_NORM_DELAYS * DS_ns) * 1e-9
    delays_samp = delays_s * fs_Hz

    h = np.zeros(L, dtype=np.complex128)
    for p, d in enumerate(delays_samp):
        i0 = int(np.floor(d)); frac = d - i0
        gp = (rng.standard_normal() + 1j*rng.standard_normal())/np.sqrt(2) * np.sqrt(pow_lin[p])
        if 0 <= i0 < L:
            h[i0] += gp*(1.0-frac)
        if 0 <= i0+1 < L:
            h[i0+1] += gp*frac
    pwr = np.sum(np.abs(h)**2)
    if pwr > 0: h /= np.sqrt(pwr)
    return h

def build_mimo_taps(Nr, Nt, L, fs_Hz, DS_ns, seed=None):
    rng = np.random.default_rng(seed)
    c = [[None]*Nt for _ in range(Nr)]
    for nr in range(Nr):
        for nt in range(Nt):
            c[nr][nt] = gen_tdlb_impulse(L, fs_Hz, DS_ns, rng)
    return c

def taps_to_H(c, N):
    """FFT taps -> H[k], shape (N, Nr, Nt)."""
    Nr, Nt = len(c), len(c[0])
    H = np.zeros((N, Nr, Nt), dtype=np.complex128)
    for nr in range(Nr):
        for nt in range(Nt):
            h = c[nr][nt]
            H[:, nr, nt] = np.fft.fft(np.r_[h, np.zeros(N-len(h))])
    return H

# ========== Build one block-fading channel realization ==========
c = build_mimo_taps(Nr, Nt, ISI_TAPS, W_Hz, DS_NS, seed=SEED)
H = taps_to_H(c, N)  # (N, Nr, Nt)

# Precompute singular values per subcarrier (independent of SNR)
Svals = np.zeros((N, min(Nr, Nt)))
for k in range(N):
    s = svd(H[k], compute_uv=False)
    Svals[k, :len(s)] = s[:min(Nr, Nt)]

# ========== Capacity & usable-rank sweep over Eb/N0 ==========
cap_avg = []       # average per-subcarrier capacity [bits/s/Hz]
frac_rank_ge4 = [] # fraction with >=4 usable streams

for eb in EbNo_dB:
    snr = 10.0**(eb/10.0)               # treat Eb/N0 ~ SNR (scaling not critical for trend)
    # Capacity with equal power: sum log2(1 + (snr/Nt)*s_i^2)
    term = (snr/Nt) * (Svals**2)
    Ck = np.sum(np.log2(1.0 + term), axis=1)  # per subcarrier
    cap_avg.append(np.mean(Ck))

    # "Usable" stream criterion: (snr/Nt)*s_i^2 >= gamma_th
    usable_streams = np.sum(term >= gamma_th, axis=1)
    frac_rank_ge4.append(np.mean(usable_streams >= 4))

cap_avg = np.array(cap_avg)
frac_rank_ge4 = np.array(frac_rank_ge4)

# ========== Plot & Save ==========
plt.figure(figsize=(8,6))
plt.plot(EbNo_dB, cap_avg, 'm.-', lw=2, ms=8, label='Avg. capacity per subcarrier')
plt.plot(EbNo_dB, frac_rank_ge4, 'c.-', lw=2, ms=8, label='Frac. rank ≥ 4')
plt.grid(True, which='both', ls=':')
plt.xlabel('E_b/N_0 [dB]')
plt.ylabel('Capacity [bits/s/Hz] / Fraction')
plt.title('4x8 MIMO: Capacity & usable rank (≥4)')
plt.legend()
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "capacity_rank_cdlb_4x8.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()

# Save data
data = {
    "EbNo_dB": EbNo_dB.tolist(),
    "capacity_avg": cap_avg.tolist(),
    "frac_rank_ge4": frac_rank_ge4.tolist(),
    "params": {
        "Nt": Nt, "Nr": Nr, "N": N, "W_Hz": W_Hz,
        "ISI_TAPS": ISI_TAPS, "DS_NS": DS_NS,
        "gamma_th": gamma_th, "seed": SEED
    }
}
with open(os.path.join(OUT_DIR, "capacity_rank_cdlb_4x8.pkl"), "wb") as f:
    pickle.dump(data, f)

print("Saved:")
print(" -", png_path)
print(" -", os.path.join(OUT_DIR, "capacity_rank_cdlb_4x8.pkl"))
