import numpy as np
import math

def unit_qam_constellation(Bi):
    """Generate normalized QAM constellation for given bits per symbol."""
    EvenSquareRoot = math.ceil(math.sqrt(2 ** Bi) / 2) * 2
    PamM = EvenSquareRoot
    PamConstellation = np.arange(-(PamM - 1), PamM, 2).astype(np.int32)
    PamConstellation = np.reshape(PamConstellation, (1, -1))
    SquareMatrix = np.matmul(np.ones((PamM, 1)), PamConstellation)
    C = SquareMatrix + 1j * (SquareMatrix.T)
    C_tmp = np.zeros(C.shape[0] * C.shape[1]).astype('complex128')
    for i in range(C.shape[1]):
        for j in range(C.shape[0]):
            C_tmp[i * C.shape[0] + j] = C[j][i]
    C = C_tmp
    return C / math.sqrt(np.mean(abs(C) ** 2))

def compute_channel_corr_matrix(IsiMagnitude):
    """Compute channel correlation matrix from power delay profile."""
    N = len(IsiMagnitude)
    r_f_bold = np.fft.fft(IsiMagnitude)
    r_f_bold_prime = r_f_bold[1:][::-1]
    r_f_bold_conj = np.conjugate(r_f_bold)
    r_f_bold_ext = np.append(r_f_bold_prime, r_f_bold_conj)
    R_f = np.zeros((N, N)).astype('complex128')
    for k in range(0, N):
        R_f[N - k - 1, :] = r_f_bold_ext[k: N + k]
    return R_f

def train_mimo_esn(esn, DelayFlag, Min_Delay, Max_Delay, CyclicPrefixLen, N, N_t, N_r, IsiDuration, y_CP, x_CP):
    """Train MIMO ESN with delay search and compute NMSE."""
    if DelayFlag:
        Delay_LUT = np.zeros(((Max_Delay + 1 - Min_Delay) ** 4, 4)).astype('int32')
        count = 0
        temp = np.zeros(Delay_LUT.shape[0])
        for ii in range(Min_Delay, Max_Delay + 1):
            for jj in range(Min_Delay, Max_Delay + 1):
                for kk in range(Min_Delay, Max_Delay + 1):
                    for ll in range(Min_Delay, Max_Delay + 1):
                        Delay_LUT[count, :] = np.array([ii, jj, kk, ll])
                        if (abs(ii - jj) > 2 or abs(kk - ll) > 2 or abs(ii - kk) > 2 or
                            abs(ii - ll) > 2 or abs(jj - kk) > 2 or abs(jj - ll) > 2):
                            temp[count] = 1
                        count += 1
        Delay_LUT = np.delete(Delay_LUT, np.where(temp > 0)[0], axis=0)
    else:
        Delay_LUT = np.zeros((Max_Delay - Min_Delay + 1, 4)).astype('int32')
        for jjjj in range(Min_Delay, Max_Delay + 1):
            Delay_LUT[jjjj - Min_Delay, :] = jjjj * np.ones(4)

    Delay_Max = np.amax(Delay_LUT, axis=1)
    Delay_Min = np.amin(Delay_LUT, axis=1)
    NMSE_ESN_Training = np.zeros(Delay_LUT.shape[0])
    for jjj in range(Delay_LUT.shape[0]):
        Curr_Delay = Delay_LUT[jjj, :]
        ESN_input = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))
        ESN_output = np.zeros((N + Delay_Max[jjj] + CyclicPrefixLen, N_t * 2))
        ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[jjj]))
        ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[jjj]))
        ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[jjj]))
        ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[jjj]))
        ESN_output[Curr_Delay[0]: Curr_Delay[0] + N + CyclicPrefixLen, 0] = x_CP[:, 0].real
        ESN_output[Curr_Delay[1]: Curr_Delay[1] + N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
        ESN_output[Curr_Delay[2]: Curr_Delay[2] + N + CyclicPrefixLen, 2] = x_CP[:, 1].real
        ESN_output[Curr_Delay[3]: Curr_Delay[3] + N + CyclicPrefixLen, 3] = x_CP[:, 1].imag
        nForgetPoints = Delay_Min[jjj] + CyclicPrefixLen
        esn.fit(ESN_input, ESN_output, nForgetPoints)
        x_hat_ESN_temp = esn.predict(ESN_input, continuation=False)[nForgetPoints:, :]
        x_hat_ESN_0 = x_hat_ESN_temp[Curr_Delay[0] - Delay_Min[jjj]: Curr_Delay[0] - Delay_Min[jjj] + N + 1, 0] + 1j * x_hat_ESN_temp[Curr_Delay[1] - Delay_Min[jjj]: Curr_Delay[1] - Delay_Min[jjj] + N + 1, 1]
        x_hat_ESN_1 = x_hat_ESN_temp[Curr_Delay[2] - Delay_Min[jjj]: Curr_Delay[2] - Delay_Min[jjj] + N + 1, 2] + 1j * x_hat_ESN_temp[Curr_Delay[3] - Delay_Min[jjj]: Curr_Delay[3] - Delay_Min[jjj] + N + 1, 3]
        x_hat_ESN_0 = x_hat_ESN_0.reshape(-1, 1)
        x_hat_ESN_1 = x_hat_ESN_1.reshape(-1, 1)
        x_hat_ESN = np.hstack((x_hat_ESN_0, x_hat_ESN_1))
        x = x_CP[IsiDuration - 1:, :]
        NMSE_ESN_Training[jjj] = (
            np.linalg.norm(x_hat_ESN[:, 0] - x[:, 0]) ** 2 / np.linalg.norm(x[:, 0]) ** 2 +
            np.linalg.norm(x_hat_ESN[:, 1] - x[:, 1]) ** 2 / np.linalg.norm(x[:, 1]) ** 2
        )
    Delay_Idx = np.argmin(NMSE_ESN_Training)
    NMSE_ESN = np.min(NMSE_ESN_Training)
    Delay = Delay_LUT[Delay_Idx, :]
    ESN_input = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))
    ESN_output = np.zeros((N + Delay_Max[Delay_Idx] + CyclicPrefixLen, N_t * 2))
    ESN_input[:, 0] = np.append(y_CP[:, 0].real, np.zeros(Delay_Max[Delay_Idx]))
    ESN_input[:, 1] = np.append(y_CP[:, 0].imag, np.zeros(Delay_Max[Delay_Idx]))
    ESN_input[:, 2] = np.append(y_CP[:, 1].real, np.zeros(Delay_Max[Delay_Idx]))
    ESN_input[:, 3] = np.append(y_CP[:, 1].imag, np.zeros(Delay_Max[Delay_Idx]))
    ESN_output[Delay[0]: Delay[0] + N + CyclicPrefixLen, 0] = x_CP[:, 0].real
    ESN_output[Delay[1]: Delay[1] + N + CyclicPrefixLen, 1] = x_CP[:, 0].imag
    ESN_output[Delay[2]: Delay[2] + N + CyclicPrefixLen, 2] = x_CP[:, 1].real
    ESN_output[Delay[3]: Delay[3] + N + CyclicPrefixLen, 3] = x_CP[:, 1].imag
    nForgetPoints = Delay_Min[Delay_Idx] + CyclicPrefixLen
    esn.fit(ESN_input, ESN_output, nForgetPoints)
    Delay_Minn = Delay_Min[Delay_Idx]
    Delay_Maxx = Delay_Max[Delay_Idx]
    return [ESN_input, ESN_output, esn, Delay, Delay_Idx, Delay_Minn, Delay_Maxx, nForgetPoints, NMSE_ESN]