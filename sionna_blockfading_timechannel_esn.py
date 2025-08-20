
import os
import math
import pickle
import numpy as np
import tensorflow as tf

from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.channel import RayleighBlockFading, GenerateTimeChannel, ApplyTimeChannel

from pyESN import ESN
from HelpFunc import HelpFunc

W = 2*1.024e6
f_D = 100
No = 1e-5
IsiDuration = 8
CP = IsiDuration - 1

N_t = 2
N_r = 2

N = 512
m = 4
NumOfdmSymbols = 1000
EbNoDB = np.arange(0, 30+1, 3).astype(np.int32)

nInternalUnits = 100
spectralRadius = 0.9
inputScaler = 0.005
inputOffset = 0.0
feedbackScaler = 0.0
teacherScalingBase = 5e-7

K_ldpc = 1024
N_ldpc = 2048

Min_Delay = 0
Max_Delay = math.ceil(IsiDuration/2) + 2

constel = Constellation('qam', num_bits_per_symbol=m, normalize=True)  # keep this
mapper   = Mapper('qam', m)            # remove normalize kwarg
demapper = Demapper('app', 'qam', m)   # pass positionally


CONST_PTS = constel.points.numpy()
M = CONST_PTS.shape[0]

def ofdm_modulate_fd(X_fd, cp_len):
    # NumPy IFFT + CP. X_fd is [N, N_t]. Returns [(N+cp), N_t].
    x_cp = np.zeros((N+cp_len, N_t), dtype=np.complex128)
    for tx in range(N_t):
        x_td = N*np.fft.ifft(X_fd[:, tx])
        x_cp[:, tx] = np.concatenate([x_td[-cp_len:], x_td])
    return x_cp

def ofdm_demodulate_td(y_td_cp, cp_len):
    # Remove CP + FFT. y_td_cp is [(N+cp), N_r]. Returns [N, N_r].
    Y = np.zeros((N, N_r), dtype=np.complex128)
    for rr in range(N_r):
        Y[:, rr] = (1.0/N) * np.fft.fft(y_td_cp[cp_len:, rr])
    return Y

def soft_clip(x, A, p=1):
    mag = np.abs(x)
    denom = np.power(1.0 + np.power(mag/A, 2*p), 1.0/(2*p))
    return x/denom

def sionna_map_bits_to_syms(bits_2d):
    # bits_2d [N*m, N_t] -> X [N, N_t]
    X = np.zeros((N, N_t), dtype=np.complex128)
    for tx in range(N_t):
        b = bits_2d[:, tx].reshape(N, m)
        b_tf = tf.convert_to_tensor(b, dtype=tf.int32)
        x_tf = mapper(b_tf)
        X[:, tx] = x_tf.numpy().astype(np.complex128).reshape(-1)
    return X

def sionna_app_llrs_awgn(Xeq, noise_var):
    # Xeq [N, N_t] -> LLRs [N*m, N_t]
    Xtf = tf.convert_to_tensor(Xeq, dtype=tf.complex64)
    Xtf = tf.reshape(Xtf, [1, N, N_t])
    nv = tf.constant(noise_var, dtype=tf.float32)
    L = demapper([Xtf, nv])
    L = tf.squeeze(L, axis=0).numpy()
    L = L.reshape(N*m, N_t)
    return L

def build_time_channel_layers():
    rb = RayleighBlockFading(num_rx=N_r, num_rx_ant=1, num_tx=N_t, num_tx_ant=1, dtype=tf.complex64)
    gen = GenerateTimeChannel(channel_model=rb,
                              bandwidth=W,
                              num_time_samples=N + CP,
                              l_min=0, l_max=0,
                              normalize_channel=True)
    app = ApplyTimeChannel(num_time_samples=N + CP, l_tot=1, add_awgn=True, dtype=tf.complex64)
    return gen, app

def hard_demap_nearest(X_est):
    # Nearest neighbor to normalized constellation points
    RxBits = np.zeros((N*m, N_t), dtype=np.int32)
    pts = CONST_PTS.astype(np.complex128)
    for k in range(N):
        for tx in range(N_t):
            d = np.abs(pts - X_est[k, tx])
            qi = int(np.argmin(d))
            bits = np.array(list(np.binary_repr(qi, width=m)), dtype=int)[::-1]
            RxBits[m*k:m*(k+1), tx] = bits
    return RxBits

def build_ldpc():
    enc = LDPC5GEncoder(k=K_ldpc, n=N_ldpc)
    dec = LDPC5GDecoder(enc, hard_out=True, num_iter=12)
    return enc, dec

def run_ldpc_chain(enc, dec, llr_streamwise, tx_bits_streamwise):
    llr = llr_streamwise.astype(np.float32)
    if llr.shape[0] < N_ldpc:
        pad = np.zeros((N_ldpc-llr.shape[0], llr.shape[1]), dtype=np.float32)
        llr = np.vstack([llr, pad])
    elif llr.shape[0] > N_ldpc:
        llr = llr[:N_ldpc, :]

    txb = tx_bits_streamwise.astype(np.int32)
    if txb.shape[0] < K_ldpc:
        padb = np.zeros((K_ldpc-txb.shape[0], txb.shape[1]), dtype=np.int32)
        tx_k = np.vstack([txb, padb])
    else:
        tx_k = txb[:K_ldpc, :]

    enc_bits = enc(tf.convert_to_tensor(tx_k.T, dtype=tf.float32))  # [N_t, N_ldpc]

    dec_in = tf.convert_to_tensor(llr.T, dtype=tf.float32)
    dec_out = dec(dec_in).numpy().T
    ber = np.mean(dec_out != tx_k)
    return ber

def main():
    BER_ESN = np.zeros_like(EbNoDB, dtype=float)
    BER_LS  = np.zeros_like(EbNoDB, dtype=float)
    BER_MMSE= np.zeros_like(EbNoDB, dtype=float)
    BER_PERF= np.zeros_like(EbNoDB, dtype=float)

    BER_ESN_DEC = np.zeros_like(EbNoDB, dtype=float)
    BER_LS_DEC  = np.zeros_like(EbNoDB, dtype=float)
    BER_MMSE_DEC= np.zeros_like(EbNoDB, dtype=float)
    BER_PERF_DEC= np.zeros_like(EbNoDB, dtype=float)

    enc, dec = build_ldpc()

    T_OFDM_Total = (N + CP)/W
    tau_c = 0.5/max(f_D, 1e-6)
    Lcoh = max(1, int(math.floor(tau_c / T_OFDM_Total)))

    for jj, ebn0_db in enumerate(EbNoDB):
        print(f'Eb/N0 = {ebn0_db} dB')
        var_x = (10.0**(ebn0_db/10.0)) * No * N
        A_Clip = np.sqrt(var_x) * (10.0**(3/20.0))
        Pi = var_x / N

        inputScaling = (inputScaler/np.sqrt(var_x))*np.ones(N_t*2)
        inputShift = (inputOffset/max(inputScaler,1e-12))*np.ones(N_t*2)
        teacherScaling = teacherScalingBase*np.ones(N_t*2)
        feedbackScaling = feedbackScaler*np.ones(N_t*2)

        TotalBitNum = 0
        Err_ESN = Err_LS = Err_MMSE = Err_PERF = 0
        TotalBERs_ESN_DEC = []
        TotalBERs_LS_DEC  = []
        TotalBERs_MMSE_DEC= []
        TotalBERs_PERF_DEC= []

        sym_idx = 0
        while sym_idx < NumOfdmSymbols:
            gen, app = build_time_channel_layers()
            h_time = gen(batch_size=1)  # [1, N_r, 1, N_t, 1, N+CP, 1]

            Ci_PERF = [[None]*N_t for _ in range(N_r)]
            H_const = h_time.numpy()[0, :, 0, :, 0, 0, 0]  # [N_r, N_t]
            for rr in range(N_r):
                for tx in range(N_t):
                    Ci_PERF[rr][tx] = np.ones(N, dtype=np.complex128) * H_const[rr, tx]

            # Pilot
            TxBitsPilot = (np.random.rand(N*m, N_t) > 0.5).astype(np.int32)
            Xp = sionna_map_bits_to_syms(TxBitsPilot)

            Xp_ls = Xp.copy()
            Xp_ls[1::2, 0] = 0
            Xp_ls[0::2, 1] = 0

            x_cp_ls = ofdm_modulate_fd(Xp_ls, CP)
            x_cp    = ofdm_modulate_fd(Xp,    CP)
            x_cp_nld = soft_clip(x_cp, A_Clip, 1)

            x_tf_ls = tf.convert_to_tensor(x_cp_ls.T[np.newaxis, :, np.newaxis, :], dtype=tf.complex64)
            no_tf = tf.constant(No, dtype=tf.float32)
            y_tf_ls = app([x_tf_ls, h_time, no_tf])
            y_cp_ls = y_tf_ls.numpy()[0, :, 0, :].T

            x_tf_nld = tf.convert_to_tensor(x_cp_nld.T[np.newaxis, :, np.newaxis, :], dtype=tf.complex64)
            y_tf_nld = app([x_tf_nld, h_time, no_tf])
            y_cp_nld = y_tf_nld.numpy()[0, :, 0, :].T

            Y_ls = ofdm_demodulate_td(y_cp_ls, CP)
            Y_ls = Y_ls / np.sqrt(Pi)

            Ci_LS = [[None]*N_t for _ in range(N_r)]
            for rr in range(N_r):
                for tx in range(N_t):
                    pilot_bins = Y_ls[tx::2, rr] / Xp_ls[tx::2, tx]
                    idx = np.arange(tx, N, 2)
                    x_anchor = idx
                    y_anchor = pilot_bins
                    if tx == 0:
                        x_anchor = np.concatenate([x_anchor, np.array([N-1])])
                        y_anchor = np.concatenate([y_anchor, y_anchor[-1:]])
                    else:
                        x_anchor = np.concatenate([np.array([0]), x_anchor])
                        y_anchor = np.concatenate([y_anchor[:1], y_anchor])
                    Ci_LS[rr][tx] = np.interp(np.arange(N), x_anchor, y_anchor)

            esn = ESN(n_inputs=N_t*2, n_outputs=N_t*2, n_reservoir=nInternalUnits,
                      spectral_radius=spectralRadius, sparsity=1 - min(0.2*nInternalUnits, 1),
                      input_shift=inputShift, input_scaling=inputScaling,
                      teacher_scaling=teacherScaling, teacher_shift=np.zeros(N_t*2),
                      feedback_scaling=feedbackScaling*np.ones(N_t*2))

            Delay_LUT = np.stack([np.array([d,d,d,d], dtype=int) for d in range(Min_Delay, Max_Delay+1)], axis=0)
            Delay_Max_vec = Delay_LUT.max(axis=1)
            Delay_Min_vec = Delay_LUT.min(axis=1)

            NMSE_per_delay = np.zeros(Delay_LUT.shape[0])
            for jjj in range(Delay_LUT.shape[0]):
                dvec = Delay_LUT[jjj]
                Dmax, Dmin = int(dvec.max()), int(dvec.min())

                X_in = np.zeros((N + Dmax + CP, N_t*2))
                X_out= np.zeros((N + Dmax + CP, N_t*2))

                X_in[:, 0] = np.concatenate([y_cp_nld[:, 0].real, np.zeros(Dmax)])
                X_in[:, 1] = np.concatenate([y_cp_nld[:, 0].imag, np.zeros(Dmax)])
                X_in[:, 2] = np.concatenate([y_cp_nld[:, 1].real, np.zeros(Dmax)])
                X_in[:, 3] = np.concatenate([y_cp_nld[:, 1].imag, np.zeros(Dmax)])

                X_out[dvec[0]:(dvec[0]+N+CP), 0] = x_cp[:, 0].real
                X_out[dvec[1]:(dvec[1]+N+CP), 1] = x_cp[:, 0].imag
                X_out[dvec[2]:(dvec[2]+N+CP), 2] = x_cp[:, 1].real
                X_out[dvec[3]:(dvec[3]+N+CP), 3] = x_cp[:, 1].imag

                nForget = Dmin + CP
                esn.fit(X_in, X_out, nForget)
                x_hat_temp = esn.predict(X_in, nForget, continuation=False)

                x_hat_0 = x_hat_temp[dvec[0]-Dmin : dvec[0]-Dmin + N + 1, 0] + 1j*x_hat_temp[dvec[1]-Dmin : dvec[1]-Dmin + N + 1, 1]
                x_hat_1 = x_hat_temp[dvec[2]-Dmin : dvec[2]-Dmin + N + 1, 2] + 1j*x_hat_temp[dvec[3]-Dmin : dvec[3]-Dmin + N + 1, 3]

                x_true = x_cp[CP:, :]
                nmse = (np.linalg.norm(x_hat_0 - x_true[:,0])**2 / np.linalg.norm(x_true[:,0])**2
                       +np.linalg.norm(x_hat_1 - x_true[:,1])**2 / np.linalg.norm(x_true[:,1])**2)
                NMSE_per_delay[jjj] = nmse

            best_idx = int(np.argmin(NMSE_per_delay))
            Delay = Delay_LUT[best_idx]
            Delay_Min = int(Delay.min())
            Delay_Max = int(Delay.max())
            nForgetPoints = Delay_Min + CP

            X_in = np.zeros((N + Delay_Max + CP, N_t*2))
            X_out= np.zeros((N + Delay_Max + CP, N_t*2))

            X_in[:, 0] = np.concatenate([y_cp_nld[:, 0].real, np.zeros(Delay_Max)])
            X_in[:, 1] = np.concatenate([y_cp_nld[:, 0].imag, np.zeros(Delay_Max)])
            X_in[:, 2] = np.concatenate([y_cp_nld[:, 1].real, np.zeros(Delay_Max)])
            X_in[:, 3] = np.concatenate([y_cp_nld[:, 1].imag, np.zeros(Delay_Max)])

            X_out[Delay[0]:(Delay[0]+N+CP), 0] = x_cp[:, 0].real
            X_out[Delay[1]:(Delay[1]+N+CP), 1] = x_cp[:, 0].imag
            X_out[Delay[2]:(Delay[2]+N+CP), 2] = x_cp[:, 1].real
            X_out[Delay[3]:(Delay[3]+N+CP), 3] = x_cp[:, 1].imag

            esn.fit(X_in, X_out, nForgetPoints)

            sym_idx += 1
            if sym_idx >= NumOfdmSymbols:
                break

            # Data symbols
            for _ in range(min(Lcoh-1, NumOfdmSymbols - sym_idx)):
                TxBits = (np.random.rand(N*m, N_t) > 0.5).astype(np.int32)
                Xd = sionna_map_bits_to_syms(TxBits)

                x_cp_d = ofdm_modulate_fd(Xd, CP)
                x_cp_d_nld = soft_clip(x_cp_d, A_Clip, 1)

                x_tf_d = tf.convert_to_tensor(x_cp_d_nld.T[np.newaxis, :, np.newaxis, :], dtype=tf.complex64)
                no_tf = tf.constant(No, dtype=tf.float32)
                y_tf_d = app([x_tf_d, h_time, no_tf])
                y_cp_d = y_tf_d.numpy()[0, :, 0, :].T

                Yd = ofdm_demodulate_td(y_cp_d, CP)

                Xhat_PERF = np.zeros_like(Xd, dtype=np.complex128)
                Xhat_LS   = np.zeros_like(Xd, dtype=np.complex128)
                Xhat_MMSE = np.zeros_like(Xd, dtype=np.complex128)

                for k in range(N):
                    H_perf = np.array([[Ci_PERF[rr][tx][k] for tx in range(N_t)] for rr in range(N_r)], dtype=np.complex128)
                    H_ls   = np.array([[Ci_LS[rr][tx][k]   for tx in range(N_t)] for rr in range(N_r)], dtype=np.complex128)
                    yk = Yd[k, :].reshape(-1,1)

                    Xhat_PERF[k,:] = (np.linalg.lstsq(H_perf, yk, rcond=None)[0].ravel()) / np.sqrt(Pi)
                    Xhat_LS[k,:]   = (np.linalg.lstsq(H_ls,   yk, rcond=None)[0].ravel()) / np.sqrt(Pi)
                    Hh = H_ls.conj().T
                    Wmmse = np.linalg.inv(Hh@H_ls + (No/Pi)*np.eye(N_t)) @ Hh
                    Xhat_MMSE[k,:] = (Wmmse @ yk).ravel() / np.sqrt(Pi)

                # ESN estimate
                X_in_d = np.zeros((N + Delay_Max + CP, N_t*2))
                X_in_d[:, 0] = np.concatenate([y_cp_d[:, 0].real, np.zeros(Delay_Max)])
                X_in_d[:, 1] = np.concatenate([y_cp_d[:, 0].imag, np.zeros(Delay_Max)])
                X_in_d[:, 2] = np.concatenate([y_cp_d[:, 1].real, np.zeros(Delay_Max)])
                X_in_d[:, 3] = np.concatenate([y_cp_d[:, 1].imag, np.zeros(Delay_Max)])

                x_hat_temp = esn.predict(X_in_d, nForgetPoints, continuation=False)
                x_hat_0 = x_hat_temp[Delay[0]-Delay_Min : Delay[0]-Delay_Min + N + 1, 0] + 1j*x_hat_temp[Delay[1]-Delay_Min : Delay[1]-Delay_Min + N + 1, 1]
                x_hat_1 = x_hat_temp[Delay[2]-Delay_Min : Delay[2]-Delay_Min + N + 1, 2] + 1j*x_hat_temp[Delay[3]-Delay_Min : Delay[3]-Delay_Min + N + 1, 3]

                x_hat_td = np.vstack([x_hat_0, x_hat_1]).T
                x_hat_td = x_hat_td[:N, :]
                Xhat_ESN = np.zeros_like(Xd, dtype=np.complex128)
                for tx in range(N_t):
                    Xhat_ESN[:, tx] = (1.0/N) * np.fft.fft(x_hat_td[:, tx]) / np.sqrt(Pi)

                Rx_ESN  = hard_demap_nearest(Xhat_ESN)
                Rx_LS   = hard_demap_nearest(Xhat_LS)
                Rx_MMSE = hard_demap_nearest(Xhat_MMSE)
                Rx_PERF = hard_demap_nearest(Xhat_PERF)

                Err_ESN  += np.sum(TxBits != Rx_ESN)
                Err_LS   += np.sum(TxBits != Rx_LS)
                Err_MMSE += np.sum(TxBits != Rx_MMSE)
                Err_PERF += np.sum(TxBits != Rx_PERF)
                TotalBitNum += m*N

                noise_var = No
                llr_esn  = sionna_app_llrs_awgn(Xhat_ESN, noise_var)
                llr_ls   = sionna_app_llrs_awgn(Xhat_LS,  noise_var)
                llr_mmse = sionna_app_llrs_awgn(Xhat_MMSE, noise_var)
                llr_perf = sionna_app_llrs_awgn(Xhat_PERF, noise_var)

                BER_ESN_dec  = run_ldpc_chain(enc, dec, llr_esn,  TxBits)
                BER_LS_dec   = run_ldpc_chain(enc, dec, llr_ls,   TxBits)
                BER_MMSE_dec = run_ldpc_chain(enc, dec, llr_mmse, TxBits)
                BER_PERF_dec = run_ldpc_chain(enc, dec, llr_perf, TxBits)

                TotalBERs_ESN_DEC.append(BER_ESN_dec)
                TotalBERs_LS_DEC.append(BER_LS_dec)
                TotalBERs_MMSE_DEC.append(BER_MMSE_dec)
                TotalBERs_PERF_DEC.append(BER_PERF_dec)

                sym_idx += 1
                if sym_idx >= NumOfdmSymbols:
                    break

        BER_ESN[jj]  = Err_ESN / TotalBitNum
        BER_LS[jj]   = Err_LS  / TotalBitNum
        BER_MMSE[jj] = Err_MMSE/ TotalBitNum
        BER_PERF[jj] = Err_PERF/ TotalBitNum

        BER_ESN_DEC[jj]  = float(np.mean(TotalBERs_ESN_DEC)) if len(TotalBERs_ESN_DEC)>0 else 0.0
        BER_LS_DEC[jj]   = float(np.mean(TotalBERs_LS_DEC)) if len(TotalBERs_LS_DEC)>0 else 0.0
        BER_MMSE_DEC[jj] = float(np.mean(TotalBERs_MMSE_DEC)) if len(TotalBERs_MMSE_DEC)>0 else 0.0
        BER_PERF_DEC[jj] = float(np.mean(TotalBERs_PERF_DEC)) if len(TotalBERs_PERF_DEC)>0 else 0.0

        print(f'  uncoded BER | ESN {BER_ESN[jj]:.3e} | LS {BER_LS[jj]:.3e} | MMSE {BER_MMSE[jj]:.3e} | Oracle {BER_PERF[jj]:.3e}')
        print(f'  decoded  BER | ESN {BER_ESN_DEC[jj]:.3e} | LS {BER_LS_DEC[jj]:.3e} | MMSE {BER_MMSE_DEC[jj]:.3e} | Oracle {BER_PERF_DEC[jj]:.3e}')

    out = {
        'EBN0': EbNoDB.tolist(),
        'BER_uncoded': {'ESN': BER_ESN.tolist(), 'LS': BER_LS.tolist(), 'MMSE': BER_MMSE.tolist(), 'Oracle': BER_PERF.tolist()},
        'BER_decoded': {'ESN': BER_ESN_DEC.tolist(), 'LS': BER_LS_DEC.tolist(), 'MMSE': BER_MMSE_DEC.tolist(), 'Oracle': BER_PERF_DEC.tolist()}
    }
    with open('BER_results_sionna_blockfading.pkl', 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
    main()
