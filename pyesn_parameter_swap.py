

import os
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate

from HelpFunc import HelpFunc
from pyESN import ESN

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

    # SNR sweep (coarse by default; you can tighten in CLI)
    params["EbNoDB"] = np.arange(0, 30 + 1, 6).astype(np.int32)

    # MIMO
    params["N_t"] = 2
    params["N_r"] = 2

    # OFDM/Design
    params["N"] = 512
    params["m"] = 4
    params["m_pilot"] = 4
    params["NumOfdmSymbols"] = 500  # keep moderate by default

    # PA
    params["p_smooth"] = 1
    params["ClipLeveldB"] = 3

    return params


# ----------------------
# Plot helper
# ----------------------

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


# ----------------------
# Core experiment (one sweep family at a time)
# ----------------------

def run_sweep_family(family_name, values, fixed_cfg, args, p):
    """
    family_name: one of {"reservoir", "spectral", "inpscale", "sparsity", "teachers"}
    values: list of floats/ints for the chosen family
    fixed_cfg: dict with default ESN params
    p: system parameters
    Returns dict with {'EBN0': array, 'curves': {label: ber_array}, 'settings': list_of_dicts}
    """
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
    if L < 2: L = 2  # ensure at least Pilot/Data alternation

    # Constellations via HelpFunc
    Const = HelpFunc.UnitQamConstellation(m)
    ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)

    PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
    PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
    CyclicPrefixLen = IsiDuration - 1

    # One-sided exponential power delay profile (normalized)
    temp = CyclicPrefixLen / 9
    IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen + 1)) / temp)
    IsiMagnitude = IsiMagnitude / np.sum(IsiMagnitude)

    # Channel correlation (time-domain taps)
    R_h = np.zeros((IsiDuration, IsiDuration))
    for ii in range(IsiDuration):
        R_h[ii, ii] = IsiMagnitude[ii]

    # Prepare outputs
    curves = {}
    settings = []

    rng = np.random.default_rng(123)

    # Build label + config per value
    label_cfg = []
    for v in values:
        cfg = fixed_cfg.copy()
        if family_name == "reservoir":
            cfg["n_reservoir"] = int(v)
            label = f"N={int(v)}"
        elif family_name == "spectral":
            cfg["spectral_radius"] = float(v)
            label = f"rho={v}"
        elif family_name == "inpscale":
            cfg["input_scaling"] = float(v)
            label = f"inpscale={v}"
        elif family_name == "sparsity":
            cfg["sparsity"] = float(v)
            label = f"sparsity={v}"
        elif family_name == "teachers":
            cfg["teacher_scaling_val"] = float(v)
            label = f"teach={v:g}"
        else:
            raise ValueError("Unknown family")
        label_cfg.append((label, cfg))

    # For each config, compute BER curve across Eb/N0
    for label, cfg in label_cfg:
        print(f"[SWEEP:{family_name}] {label} -> cfg={cfg}")
        BER_curve = np.zeros(len(EbNoDB))

        # SNR loop
        for jj, eb in enumerate(EbNoDB):
            Pi = (10**(eb/10)) * No  # power per subcarrier
            var_x = np.float_power(10, (eb/10)) * No * N
            A_Clip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB/20)

            # ESN hyperparameters for this config
            nInputUnits = N_t * 2
            nOutputUnits = N_t * 2
            nInternalUnits = cfg["n_reservoir"]
            inputScaler = cfg["input_scaling"]
            inputOffset = 0.0
            feedbackScaler = 0.0
            teacherScaling = cfg["teacher_scaling_val"] * np.ones(N_t * 2)
            spectralRadius = cfg["spectral_radius"]
            teacherShift = np.zeros(N_t * 2)
            feedbackScaling = feedbackScaler * np.ones(N_t * 2)
            sparsity = cfg["sparsity"]

            # Delay & forget setup
            Min_Delay = 0
            Max_Delay = math.ceil(IsiDuration/2) + 2
            DelayFlag = 0

            # Accumulators
            TotalBerNum = 0
            TotalBerDen = 0

            # For interpolation in LS/MMSE (not used here; ESN only, but needed for ESN input preparation)
            # (We keep as in baseline to maintain timing/struct alignment)
            # Training and testing happen within one coherence block
            trained_esn = None
            Delay = None; Delay_Min = None; Delay_Max = Max_Delay; nForgetPoints = None

            for kk in range(1, NumOfdmSymbols + 1):
                # Re-draw channel every L symbols
                if (np.remainder(kk, L) == 1):
                    c = [[None] * N_t for _ in range(N_r)]
                    Ci = [[None] * N_t for _ in range(N_r)]
                    for nnn in range(N_r):
                        for mmm in range(N_t):
                            taps = rng.normal(size=IsiDuration)/(2**0.5) + 1j * rng.normal(size=IsiDuration)/(2**0.5)
                            c[nnn][mmm] = taps * (IsiMagnitude**0.5)
                            Ci[nnn][mmm] = np.fft.fft(np.append(c[nnn][mmm], np.zeros(N - len(c[nnn][mmm]))))

                    # ---------- Pilot symbol (training) ----------
                    TxBits = (rng.random(size=(N*m_pilot, N_t)) > 0.5).astype(np.int32)
                    X = np.zeros((N, N_t), dtype=np.complex128)
                    x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)

                    for ii in range(N):
                        for iii in range(N_t):
                            idx = int(np.matmul(PowersOfTwoPilot[:, :m_pilot], TxBits[m_pilot*ii + np.arange(m_pilot), iii])[0])
                            X[ii, iii] = ConstPilot[idx]

                    # Time-domain + CP
                    for iii in range(N_t):
                        x_temp = N * np.fft.ifft(X[:, iii])
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

                    # ESN training pairs via helper
                    inputScaling_vec = inputScaler / (var_x**0.5) * np.ones(N_t * 2)
                    inputShift_vec = inputOffset / max(1e-12, inputScaler) * np.ones(N_t * 2)

                    esn = ESN(
                        n_inputs=nInputUnits, n_outputs=nOutputUnits, n_reservoir=nInternalUnits,
                        spectral_radius=spectralRadius, sparsity=sparsity,
                        input_shift=inputShift_vec, input_scaling=inputScaling_vec,
                        teacher_scaling=teacherScaling, teacher_shift=teacherShift,
                        feedback_scaling=feedbackScaling
                    )

                    (ESN_input, ESN_output, trained_esn, Delay, Delay_Idx,
                     Delay_Min, Delay_Max, nForgetPoints, NMSE_ESN_train) = HelpFunc.trainMIMOESN(
                        esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r,
                        IsiDuration, y_CP_NLD, x_CP)

                else:
                    # ---------- Data symbol (testing) ----------
                    TxBits = (rng.random(size=(N*m, N_t)) > 0.5).astype(np.int32)
                    X = np.zeros((N, N_t), dtype=np.complex128)
                    x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=np.complex128)
                    for ii in range(N):
                        for iii in range(N_t):
                            idx = int(np.matmul(PowersOfTwo[:, :m], TxBits[m*ii + np.arange(m), iii])[0])
                            X[ii, iii] = Const[idx]

                    for iii in range(N_t):
                        x_temp = N * np.fft.ifft(X[:, iii])
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

                    # ESN-style test input
                    ESN_input_test = np.zeros((N + Delay_Max + CyclicPrefixLen, N_t * 2))
                    ESN_input_test[:, 0] = np.append(y_CP_NLD[:, 0].real, np.zeros(Delay_Max))
                    ESN_input_test[:, 1] = np.append(y_CP_NLD[:, 0].imag, np.zeros(Delay_Max))
                    ESN_input_test[:, 2] = np.append(y_CP_NLD[:, 1].real, np.zeros(Delay_Max))
                    ESN_input_test[:, 3] = np.append(y_CP_NLD[:, 1].imag, np.zeros(Delay_Max))

                    # ESN prediction
                    nForgetPoints_inf = Delay_Min + CyclicPrefixLen
                    xhat_esn_temp = trained_esn.predict(ESN_input_test, nForgetPoints_inf, continuation=False)

                    x_hat_ESN_0 = xhat_esn_temp[Delay[0] - Delay_Min : Delay[0] - Delay_Min + N + 1, 0] \
                                  + 1j * xhat_esn_temp[Delay[1] - Delay_Min : Delay[1] - Delay_Min + N + 1, 1]
                    x_hat_ESN_1 = xhat_esn_temp[Delay[2] - Delay_Min : Delay[2] - Delay_Min + N + 1, 2] \
                                  + 1j * xhat_esn_temp[Delay[3] - Delay_Min : Delay[3] - Delay_Min + N + 1, 3]
                    x_hat_ESN = np.hstack((x_hat_ESN_0.reshape(-1,1), x_hat_ESN_1.reshape(-1,1)))

                    # Map to freq domain for decisions
                    X_hat_ESN = np.zeros_like(X, dtype=np.complex128)
                    for ii in range(N_t):
                        X_hat_ESN[:, ii] = (1/N) * np.fft.fft(x_hat_ESN[:, ii]) / math.sqrt(Pi)

                    # Hard decisions
                    def hard_bits_from_symbols(X_hat):
                        RxBits = np.zeros_like(TxBits)
                        for ii in range(N):
                            for iii in range(N_t):
                                qidx = np.argmin(np.abs(Const - X_hat[ii, iii]))
                                bits = list(format(qidx, 'b').zfill(m))
                                bits = np.array([int(b) for b in bits])[::-1]
                                RxBits[m*ii : m*(ii+1), iii] = bits
                        return RxBits

                    RxBits_ESN = hard_bits_from_symbols(X_hat_ESN)

                    TotalBerNum += np.sum(TxBits != RxBits_ESN)
                    TotalBerDen += m * N * N_t

            BER_curve[jj] = TotalBerNum / max(1, TotalBerDen)

        curves[label] = BER_curve
        settings.append(cfg)

    # Save artifacts
    out = {"EBN0": p["EbNoDB"], "curves": curves, "settings": settings, "family": family_name}
    pkl_name = f"ESN_sweep_{family_name}.pkl"
    png_name = f"ESN_sweep_{family_name}.png"
    with open(pkl_name, "wb") as f:
        pickle.dump(out, f)
    semilogy_ber(p["EbNoDB"], curves, f"ESN {family_name} sweep (BER vs. Eb/N0)", png_name)
    print(f"[DONE] {family_name} -> saved {pkl_name}, {png_name}")
    return out


# ----------------------
# CLI
# ----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebn0", type=str, default=None, help="Comma-separated Eb/N0 dB list, e.g., 0,6,12,18,24,30")
    ap.add_argument("--num_symbols", type=int, default=None, help="Total OFDM symbols (per SNR) to simulate (default 800)")
    ap.add_argument("--reservoir", type=str, default="50,100,200,400", help="Reservoir sizes to sweep")
    ap.add_argument("--spectral", type=str, default="0.7,0.9,1.1", help="Spectral radius values to sweep")
    ap.add_argument("--input_scale", type=str, default="0.001,0.005,0.02", help="Input scaling values to sweep")
    ap.add_argument("--sparsity", type=str, default="0.1,0.3,0.5", help="Reservoir sparsity (0..1) values to sweep")
    ap.add_argument("--teacher_scale", type=str, default="5e-7,1e-6,5e-6", help="Teacher scaling magnitudes to sweep")
    return ap.parse_args()


def parse_list(s, cast):
    return [cast(x.strip()) for x in s.split(",") if len(x.strip()) > 0]


# ----------------------
# Main
# ----------------------

def main():
    args = parse_args()
    p = get_default_params()

    if args.ebn0 is not None:
        p["EbNoDB"] = np.array(parse_list(args.ebn0, float), dtype=np.float64)
    if args.num_symbols is not None:
        p["NumOfdmSymbols"] = int(args.num_symbols)

    # Fixed/default ESN config (used unless a family overrides its target hyperparameter)
    fixed_cfg = {
        "n_reservoir": 200,
        "spectral_radius": 0.9,
        "input_scaling": 0.005,
        "sparsity": 0.2,
        "teacher_scaling_val": 5e-7,
    }

    # Value lists for each family
    reservoir_values = parse_list(args.reservoir, int)
    spectral_values  = parse_list(args.spectral, float)
    inpscale_values  = parse_list(args.input_scale, float)
    sparsity_values  = parse_list(args.sparsity, float)
    teach_values     = parse_list(args.teacher_scale, float)

    # Run sweeps
    run_sweep_family("reservoir", reservoir_values, fixed_cfg, args, p)
    run_sweep_family("spectral", spectral_values, fixed_cfg, args, p)
    run_sweep_family("inpscale",  inpscale_values, fixed_cfg, args, p)
    run_sweep_family("sparsity",  sparsity_values, fixed_cfg, args, p)
    run_sweep_family("teachers",  teach_values, fixed_cfg, args, p)

if __name__ == "__main__":
    main()

