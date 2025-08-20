import numpy as np
import math
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.channel.tr38901 import CDL, PanelArray
from sionna.channel import cir_to_time_channel
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from HelpFunc import HelpFunc
from pyESN import ESN
# Assume HelpFunc and ESN classes are imported from provided files
# HelpFunc.py and pyESN.py are as provided by you

# Sionna Channel Setup
N_r, N_t = 2, 2
carrier_frequency = 3.5e9
bs_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization='dual',
    polarization_type='VH',
    antenna_pattern='38.901',
    carrier_frequency=carrier_frequency,
)
ut_array = PanelArray(
    num_rows_per_panel=1,
    num_cols_per_panel=1,
    polarization='dual',
    polarization_type='VH',
    antenna_pattern='omni',
    carrier_frequency=carrier_frequency,
)
delay_spread = 300e-9
cdl = CDL(
    model="C",
    delay_spread=delay_spread,
    carrier_frequency=carrier_frequency,
    ut_array=ut_array,
    bs_array=bs_array,
    direction="downlink",
    min_speed=0.,
    max_speed=0.,
    dtype=tf.complex64
)

# LDPC Parameters
code_rate = 1/2
n_ldpc = 256  # Matches N * m = 64 * 4
k_ldpc = int(n_ldpc * code_rate)  # 128
num_codewords = 1
if (64 * 4) != n_ldpc:
    raise ValueError("Number of bits per symbol per antenna must equal n_ldpc")
ldpc_encoder = LDPC5GEncoder(k=k_ldpc, n=n_ldpc, dtype=tf.float32)
ldpc_decoder = LDPC5GDecoder(encoder=ldpc_encoder, num_iter=20)

# Physical Parameters
W = 2 * 1.024e6
f_D = 100
No = 0.00001
IsiDuration = 8
cFlag = False
EbNoDB = np.arange(25, 31, 5).astype(np.int32)

# MIMO Parameters
N = 64
Subcarrier_Spacing = W / N
m = 4
m_pilot = 4
NumOfdmSymbols = 1000
Ptotal = 10 ** (EbNoDB / 10) * No * N * code_rate

# Secondary Parameters
T_OFDM = N / W
T_OFDM_Total = (N + IsiDuration - 1) / W
T_s = 1 / W
tau_c = 0.5 / f_D
L = math.floor(tau_c / T_OFDM_Total)
Pi = Ptotal / N
NumBitsPerSymbol = m * N
Const = HelpFunc.UnitQamConstellation(m)
ConstPilot = HelpFunc.UnitQamConstellation(m_pilot)
PowersOfTwo = np.power(2, np.arange(m)).reshape((1, -1))
CyclicPrefixLen = IsiDuration - 1

# ESN Parameters
var_x = np.power(10, (EbNoDB / 10)) * No * N * code_rate
nInputUnits = N_t * 2
nOutputUnits = N_t * 2
nInternalUnits = 5000
inputScaler = 0.1  # Adjusted
inputOffset = 0.0
feedbackScaler = 0.0
teacherScaling = 0.1 * np.ones(N_t * 2)  # Adjusted
spectralRadius = 0.9
teacherShift = np.zeros(N_t * 2)
feedbackScaling = feedbackScaler * np.ones(N_t * 2)
Min_Delay = 0
Max_Delay = math.ceil(IsiDuration / 2) + 2
DelayFlag = 0

# Initialize Arrays
ESN_train_input = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_train_teacher = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_test_input = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
ESN_test_output = [[None] * len(EbNoDB) for _ in range(NumOfdmSymbols)]
BER_ESN = np.zeros(len(EbNoDB))
BER_Perfect = np.zeros(len(EbNoDB))
BER_LS = np.zeros(len(EbNoDB))
BER_MMSE = np.zeros(len(EbNoDB))
NMSE_ESN_Testing = np.zeros(len(EbNoDB))
NMSE_ESN_Training = np.zeros(len(EbNoDB))
c = [[None] * N_t for _ in range(N_r)]
Ci = [[None] * N_t for _ in range(N_r)]
Ci_LS = [[None] * N_t for _ in range(N_r)]
Ci_MMSE = [[None] * N_t for _ in range(N_r)]
Ci_LS_Pilots = [[None] * N_t for _ in range(N_r)]
MMSEScaler = No / Pi
R_h = np.zeros((IsiDuration, IsiDuration))  # Placeholder, computed later

# Data Flow Check
print("=== Parameter Initialization ===")
print(f"EbNoDB: {EbNoDB}")
print(f"Ptotal shape: {Ptotal.shape}, values: {Ptotal}")
print(f"Subcarrier Spacing: {Subcarrier_Spacing:.2e} Hz")
print(f"Coherence Length (L): {L} symbols")
print(f"Data Constellation shape: {Const.shape}, first few points: {Const[:3]}")
print(f"BER Arrays shape: {BER_ESN.shape}")