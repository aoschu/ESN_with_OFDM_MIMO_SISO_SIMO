import numpy as np
import matplotlib.pyplot as plt
from pyESN import ESN
from sklearn.metrics import accuracy_score

# ----------------------------------------
# Generate Noisy 16-QAM Symbols
def generate_qam_data(n_symbols=10000, modulation_order=16, snr_db=15):
    m = int(np.sqrt(modulation_order))
    real = 2 * np.random.randint(0, m, size=n_symbols) - (m - 1)
    imag = 2 * np.random.randint(0, m, size=n_symbols) - (m - 1)
    labels = (real + (m - 1)) // 2 * m + (imag + (m - 1)) // 2
    symbols = real + 1j * imag
    symbols /= np.sqrt((2 / 3) * (modulation_order - 1))  # Normalize power
    noise_std = 1 / np.sqrt(2 * 10**(snr_db / 10))
    noise = noise_std * (np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols))
    noisy = symbols + noise
    return noisy, labels

# Define 16-QAM Constellation Points
def qam_constellation(mod_order=16):
    m = int(np.sqrt(mod_order))
    re = np.arange(-(m - 1), m + 1, 2)
    im = np.arange(-(m - 1), m + 1, 2)
    const = np.array([r + 1j * i for r in re for i in im])
    return const / np.sqrt((2 / 3) * (mod_order - 1))

# Map to Nearest QAM Constellation Point
def nearest_constellation_symbol(y_pred_complex, constellation):
    distances = np.abs(y_pred_complex[:, None] - constellation[None, :])
    return np.argmin(distances, axis=1)

# ----------------------------------------
# Main ESN QAM Classifier using pyESN
def run_qam_esn():
    # Generate QAM dataset
    symbols, labels = generate_qam_data(n_symbols=10000, modulation_order=16, snr_db=15)
    constellation = qam_constellation()

    # Dummy input (pyESN needs some input signal)
    X_input = np.ones((len(symbols), 1))
    y_target = np.stack([symbols.real, symbols.imag], axis=1)

    # Train-test split
    split_idx = int(0.8 * len(symbols))
    X_train, X_test = X_input[:split_idx], X_input[split_idx:]
    y_train, y_test = y_target[:split_idx], y_target[split_idx:]
    labels_true = labels[split_idx:]

    # 
    
    esn_real = ESN(n_inputs=1, n_outputs=1, n_reservoir=5000, spectral_radius=1.5, random_state=42)
    esn_imag = ESN(n_inputs=1, n_outputs=1, n_reservoir=5000, spectral_radius=1.5, random_state=42)

    esn_real.fit(X_train, y_train[:, 0])
    esn_imag.fit(X_train, y_train[:, 1])

    # Predict only for test samples
    y_pred_real = esn_real.predict(X_test).reshape(-1)
    y_pred_imag = esn_imag.predict(X_test).reshape(-1)

    assert len(y_pred_real) == len(y_pred_imag) == len(labels_true), \
        f"Inconsistent prediction length: real={len(y_pred_real)}, imag={len(y_pred_imag)}, labels={len(labels_true)}"

    # Form complex-valued predictions
    y_pred_complex = y_pred_real + 1j * y_pred_imag

    # Nearest QAM symbol classification
    predicted_labels = nearest_constellation_symbol(y_pred_complex, constellation)
    predicted_labels = np.asarray(predicted_labels).astype(int).ravel()
    labels_true = np.asarray(labels_true).astype(int).ravel()

    # Accuracy
    acc = accuracy_score(labels_true, predicted_labels)
    print(f"\n Symbol detection accuracy (16-QAM): {acc:.4f}")


    # ----------------------------------------
    # Optional: Plot Constellation
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[:, 0], y_test[:, 1], label="True", alpha=0.4, s=10)
    plt.scatter(y_pred_real, y_pred_imag, label="Predicted", alpha=0.4, s=10)
    plt.title("16-QAM Prediction via pyESN")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_qam_esn()
