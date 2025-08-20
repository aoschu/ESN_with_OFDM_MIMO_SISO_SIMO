# Demo_MIMO_4x8_DeepESN_FAST_v2_fixed.py
import os, math, numpy as np, matplotlib.pyplot as plt
from scipy import signal, interpolate
from deepesn import DeepESN
from deepesn.configurations import config_MG

def unit_qam_constellation(Bi):
    EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2
    PamM = EvenSquareRoot
    PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32).reshape(1, -1)
    SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
    C = SquareMatrix + 1j * (SquareMatrix.T)
    C_tmp = np.zeros(C.shape[0]*C.shape[1], dtype=np.complex128)
    for i in range(C.shape[1]):
        for j in range(C.shape[0]):
            C_tmp[i*C.shape[0] + j] = C[j][i]
    C = C_tmp
    return C / math.sqrt(np.mean(np.abs(C) ** 2))

def bits_to_grayvec(idx, m):
    b = list(format(int(idx), 'b').zfill(m))
    return np.array([int(i) for i in b])[::-1]

def equalize_zf(Yk, Hk, power_scale):
    HH = Hk.conj().T
    G = HH @ Hk
    G += 1e-12 * np.eye(G.shape[0], dtype=G.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def equalize_mmse(Yk, Hk, power_scale, noise_over_power):
    HH = Hk.conj().T
    G = HH @ Hk + noise_over_power * np.eye(Hk.shape[1], dtype=Hk.dtype)
    Xhat = np.linalg.solve(G, HH @ Yk)
    return Xhat / power_scale

def hard_bits_from_syms(Xhat_matrix, Const, m):
    N, N_t = Xhat_matrix.shape
    RxBits = np.zeros((N*m, N_t), dtype=int)
    for ii in range(N):
        for tx in range(N_t):
            sym = Xhat_matrix[ii, tx]
            idx = int(np.argmin(np.abs(Const - sym)))
            RxBits[m*ii:m*(ii+1), tx] = bits_to_grayvec(idx, m)
    return RxBits

def normalize_output_matrix(Yout, Ny, Tlen, ctx):
    Y = np.asarray(Yout)
    if Y.ndim == 2:
        if Y.shape == (Ny, Tlen):   return Y
        if Y.shape == (Tlen, Ny):   return Y.T
        raise ValueError(f"[{ctx}] Unexpected 2D shape {Y.shape}, expected {(Ny,Tlen)} or {(Tlen,Ny)}.")
    if Y.ndim == 1:
        if Y.size == Ny*Tlen:       return Y.reshape(Ny, Tlen)
        if Y.size == Tlen:
            raise ValueError(f"[{ctx}] Model produced a single output channel (T,), "
                             f"but Ny={Ny} required. Check train_targets orientation.")
        raise ValueError(f"[{ctx}] Unexpected 1D size {Y.size}, cannot reshape to {(Ny,Tlen)}.")
    raise ValueError(f"[{ctx}] Unexpected output ndim={Y.ndim}.")

# FAST knobs
np.random.seed(7)
EbNoDB = np.array([0,6,12,18,24], dtype=int)
NumOfdmSymbols = 120
N = 256
Nl = 4
Nr_res = 120
TRANSIENT = 40

# System
N_t = 4; N_r = 8
m = 4; m_pilot = 4
W = 2*1.024e6; f_D = 100; No = 1e-5; IsiDuration = 8
ClipLeveldB = 3; p_smooth = 1
Ptotal = 10**(EbNoDB/10)*No*N; Pi = Ptotal/N
Const = np.array(unit_qam_constellation(m)).astype(complex)
ConstPilot = np.array(unit_qam_constellation(m_pilot)).astype(complex)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
PowersOfTwoPilot = np.power(2, np.arange(m_pilot)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1
T_OFDM_Total = (N+IsiDuration-1)/W; tau_c = 0.5/f_D
L = math.floor(tau_c/T_OFDM_Total)
temp = CyclicPrefixLen/9
IsiMagnitude = np.exp(-(np.arange(CyclicPrefixLen+1))/temp); IsiMagnitude /= np.sum(IsiMagnitude)

BER_DeepESN = np.zeros(len(EbNoDB))
BER_MMSE    = np.zeros(len(EbNoDB))
BER_ZF_LS   = np.zeros(len(EbNoDB))
BER_ZF_true = np.zeros(len(EbNoDB))

for jj, ebno_db in enumerate(EbNoDB):
    print(f"\n=== SNR {ebno_db} dB ===")
    var_x = 10**(ebno_db/10)*No*N
    A_Clip = np.sqrt(var_x) * np.float_power(10, ClipLeveldB/20)

    TotalErr_DeepESN = TotalErr_MMSE = TotalErr_ZF_LS = TotalErr_ZF_true = 0
    TotalBits = 0

    dummy_idx = list(range(N + CyclicPrefixLen))
    configs = config_MG(dummy_idx)
    Nu = 2*N_r; Ny = 2*N_t
    deepESN = DeepESN(Nu, Nr_res, Nl, configs)

    for kk in range(1, NumOfdmSymbols+1):
        redraw = (np.remainder(kk, L) == 1)
        if redraw:
            # channel
            c = [[None for _ in range(N_t)] for __ in range(N_r)]
            H_true = np.zeros((N, N_r, N_t), dtype=complex)
            for nr in range(N_r):
                for nt in range(N_t):
                    c0 = (np.random.randn(IsiDuration) + 1j*np.random.randn(IsiDuration))/np.sqrt(2)
                    c0 *= np.sqrt(IsiMagnitude[:IsiDuration])
                    c[nr][nt] = c0
                    H_true[:, nr, nt] = np.fft.fft(np.r_[c0, np.zeros(N - len(c0))])

            # pilots
            TxBitsPilot = (np.random.rand(N*m_pilot, N_t) > 0.5).astype(np.int32)
            X_p = np.zeros((N, N_t), dtype=complex)
            for ii in range(N):
                for tx in range(N_t):
                    idx = int((PowersOfTwoPilot @ TxBitsPilot[m_pilot*ii + np.arange(m_pilot), tx])[0])
                    X_p[ii, tx] = ConstPilot[idx]
            X_LS = np.zeros_like(X_p)
            for tx in range(N_t):
                X_LS[tx::N_t, tx] = X_p[tx::N_t, tx]

            x_temp = np.zeros((N, N_t), dtype=complex)
            x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
            x_LS_CP = np.zeros_like(x_CP)
            for tx in range(N_t):
                x_temp[:, tx] = N * np.fft.ifft(X_p[:, tx])
                x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
                x_temp_ls = N * np.fft.ifft(X_LS[:, tx])
                x_LS_CP[:, tx] = np.r_[x_temp_ls[-CyclicPrefixLen:], x_temp_ls] * (Pi[jj]**0.5)

            x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
            x_LS_CP_NLD = x_LS_CP / ((1 + (np.abs(x_LS_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))

            y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
            y_LS_CP = np.zeros_like(y_CP)
            for nr in range(N_r):
                for tx in range(N_t):
                    y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
                    y_LS_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_LS_CP_NLD[:, tx])
                noise = math.sqrt((N + CyclicPrefixLen)*No/2) * (np.random.randn(N + CyclicPrefixLen) + 1j*np.random.randn(N + CyclicPrefixLen))
                y_CP[:, nr] += noise
                y_LS_CP[:, nr] += noise

            Y_LS = (1/N) * np.fft.fft(y_LS_CP[CyclicPrefixLen:, :], axis=0)
            H_LS = np.zeros_like(H_true)
            H_MMSE = np.zeros_like(H_true)
            R_h = np.diag(IsiMagnitude[:IsiDuration]); MMSEScaler = (No/Pi[jj])/(N/2)
            for nr in range(N_r):
                for tx in range(N_t):
                    sc_idx = np.arange(tx, N, N_t)
                    denom = (X_LS[sc_idx, tx] * (Pi[jj]**0.5) + 1e-12)
                    Hls_sc = Y_LS[sc_idx, nr] / denom
                    tmpf = interpolate.interp1d(sc_idx, Hls_sc, kind='linear', bounds_error=False, fill_value='extrapolate')
                    Hls_full = tmpf(np.arange(N))
                    c_LS = np.fft.ifft(Hls_full); c_LS_trunc = c_LS[:IsiDuration]
                    c_MMSE = np.linalg.solve(np.dot(np.linalg.inv(R_h), MMSEScaler) + np.eye(IsiDuration), c_LS_trunc)
                    Hmmse_full = np.fft.fft(np.r_[c_MMSE, np.zeros(N-IsiDuration)])
                    H_LS[:, nr, tx] = Hls_full; H_MMSE[:, nr, tx] = Hmmse_full

            # DeepESN training (LIST API)
            Tpilot = N + CyclicPrefixLen
            U = np.zeros((2*N_r, Tpilot))
            for rx in range(N_r):
                U[2*rx, :]   = y_CP[:, rx].real
                U[2*rx+1, :] = y_CP[:, rx].imag
            D = np.zeros((2*N_t, Tpilot))
            for tx in range(N_t):
                D[2*tx,   :] = x_CP[:, tx].real
                D[2*tx+1, :] = x_CP[:, tx].imag

            states_list = deepESN.computeState([U], deepESN.IPconf.DeepIP)
            states = states_list[0]                       # (Nl*Nr_res, Tpilot)
            train_states_list  = [states[:, TRANSIENT:]]  # (features, time)
            train_targets_list = [D.T[TRANSIENT:, :]]     # **** FIX: (time, Ny)
            deepESN.trainReadout(train_states_list, train_targets_list, 1e-6)

        # DATA OFDM
        TxBits = (np.random.rand(N*m, N_t) > 0.5).astype(np.int32)
        X = np.zeros((N, N_t), dtype=complex)
        for ii in range(N):
            for tx in range(N_t):
                idx = int((PowersOfTwo @ TxBits[m*ii + np.arange(m), tx])[0])
                X[ii, tx] = Const[idx]
        x_temp = np.zeros((N, N_t), dtype=complex)
        x_CP = np.zeros((N + CyclicPrefixLen, N_t), dtype=complex)
        for tx in range(N_t):
            x_temp[:, tx] = N * np.fft.ifft(X[:, tx])
            x_CP[:, tx] = np.r_[x_temp[-CyclicPrefixLen:, tx], x_temp[:, tx]] * (Pi[jj]**0.5)
        x_CP_NLD = x_CP / ((1 + (np.abs(x_CP)/A_Clip)**(2*p_smooth))**(1/(2*p_smooth)))
        y_CP = np.zeros((N + CyclicPrefixLen, N_r), dtype=complex)
        for nr in range(N_r):
            for tx in range(N_t):
                y_CP[:, nr] += signal.lfilter(c[nr][tx], np.array([1]), x_CP_NLD[:, tx])
            noise = math.sqrt(len(y_CP[:, nr])*No/2) * (np.random.randn(len(y_CP[:, nr])) + 1j*np.random.randn(len(y_CP[:, nr])))
            y_CP[:, nr] += noise

        Y = (1/N) * np.fft.fft(y_CP[CyclicPrefixLen:, :], axis=0)

        # DeepESN inference
        Tdata = N + CyclicPrefixLen
        U_data = np.zeros((2*N_r, Tdata))
        for rx in range(N_r):
            U_data[2*rx, :]   = y_CP[:, rx].real
            U_data[2*rx+1, :] = y_CP[:, rx].imag
        states_data_list = deepESN.computeState([U_data], deepESN.IPconf.DeepIP)
        outputs_list = deepESN.computeOutput(states_data_list)
        Ydeep = normalize_output_matrix(outputs_list[0], Ny=2*N_t, Tlen=Tdata, ctx="computeOutput")

        X_hat_DeepESN = np.zeros((N, N_t), dtype=complex)
        for tx in range(N_t):
            xhat_td = Ydeep[2*tx, :] + 1j*Ydeep[2*tx+1, :]
            xhat_noCP = xhat_td[CyclicPrefixLen:]
            X_hat_DeepESN[:, tx] = (1/N) * np.fft.fft(xhat_noCP) / math.sqrt(Pi[jj])

        # Linear equalizers
        X_hat_MMSE = np.zeros((N, N_t), dtype=complex)
        X_hat_ZF_LS = np.zeros((N, N_t), dtype=complex)
        X_hat_ZF_true = np.zeros((N, N_t), dtype=complex)
        for k in range(N):
            Yk = Y[k, :].reshape(N_r, 1)
            Hk_true  = H_true[k, :, :]
            Hk_LS    = H_LS[k, :, :]
            Hk_MMSEf = H_MMSE[k, :, :]
            X_hat_ZF_true[k, :] = equalize_zf(Yk, Hk_true,  math.sqrt(Pi[jj])).reshape(-1)
            X_hat_ZF_LS[k, :]   = equalize_zf(Yk, Hk_LS,    math.sqrt(Pi[jj])).reshape(-1)
            X_hat_MMSE[k, :]    = equalize_mmse(Yk, Hk_MMSEf, math.sqrt(Pi[jj]), noise_over_power=(No/Pi[jj])).reshape(-1)

        RxBits_DeepESN = hard_bits_from_syms(X_hat_DeepESN, Const, m)
        RxBits_MMSE    = hard_bits_from_syms(X_hat_MMSE,    Const, m)
        RxBits_ZFLS    = hard_bits_from_syms(X_hat_ZF_LS,   Const, m)
        RxBits_ZFtrue  = hard_bits_from_syms(X_hat_ZF_true, Const, m)

        TotalErr_DeepESN += int(np.sum(TxBits != RxBits_DeepESN))
        TotalErr_MMSE    += int(np.sum(TxBits != RxBits_MMSE))
        TotalErr_ZF_LS   += int(np.sum(TxBits != RxBits_ZFLS))
        TotalErr_ZF_true += int(np.sum(TxBits != RxBits_ZFtrue))
        TotalBits        += N*m*N_t

    BER_DeepESN[jj] = TotalErr_DeepESN / max(TotalBits,1)
    BER_MMSE[jj]    = TotalErr_MMSE    / max(TotalBits,1)
    BER_ZF_LS[jj]   = TotalErr_ZF_LS   / max(TotalBits,1)
    BER_ZF_true[jj] = TotalErr_ZF_true / max(TotalBits,1)

outdir = "./results_4x8"; os.makedirs(outdir, exist_ok=True)
plt.figure(figsize=(9,6))
plt.semilogy(EbNoDB, BER_ZF_true, 'kx-', label='Perfect ZF (uncoded)')
plt.semilogy(EbNoDB, BER_MMSE,    'rs-.', label='MMSE (uncoded)')
plt.semilogy(EbNoDB, BER_ZF_LS,   'o-',   label='LS ZF (uncoded)')
plt.semilogy(EbNoDB, BER_DeepESN, 'g^--', label=f'DeepESN L{Nl}x{Nr_res} (uncoded)')
plt.legend(); plt.grid(True, which='both', ls=':')
plt.title('4x8 MIMO | DeepESN vs Linear Equalizers | Pre-LDPC (FAST v2 fixed)')
plt.xlabel('E_b/N_0 [dB]'); plt.ylabel('Bit Error Rate')
plt.tight_layout()
png1 = f"{outdir}/BER_DeepESN_preLDPC_FAST_v2_fixed.png"
plt.savefig(png1, dpi=150); plt.show()
print("Saved:", png1)
