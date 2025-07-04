import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the vectorized bitwise Hamming distance between two arrays.

    Parameters
    ----------
    a : np.ndarray
        First array of bits (0/1 or ±1), shape (..., n_bits).
    b : np.ndarray
        Second array, same shape as a.

    Returns
    -------
    np.ndarray
        Hamming distance for each pair along the last axis.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    # Convert ±1 to 0/1 if needed
    if np.any((a == -1) | (b == -1)):
        a = (a > 0).astype(int)
        b = (b > 0).astype(int)
    return np.sum(a != b, axis=-1)

def uniqueness(challenges: np.ndarray, responses_matrix: np.ndarray) -> float:
    """
    Compute mean inter-chip Hamming distance (uniqueness) in percent.

    Parameters
    ----------
    challenges : np.ndarray
        Array of challenges, shape (n_challenges, n_bits).
    responses_matrix : np.ndarray
        Matrix of responses from different PUFs, shape (n_pufs, n_challenges).

    Returns
    -------
    float
        Mean inter-chip Hamming distance as a percentage (0-100).
    """
    n_pufs = responses_matrix.shape[0]
    hd_sum = 0
    count = 0
    for i in range(n_pufs):
        for j in range(i + 1, n_pufs):
            hd = hamming(responses_matrix[i], responses_matrix[j])
            hd_sum += np.mean(hd)
            count += 1
    mean_hd = hd_sum / count if count > 0 else 0.0
    n_bits = responses_matrix.shape[1]
    return 100.0 * mean_hd / n_bits

def bit_error_rate(golden: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute the bit error rate (BER) as a percentage.

    Parameters
    ----------
    golden : np.ndarray
        Reference (golden) response array.
    noisy : np.ndarray
        Noisy or test response array, same shape as golden.

    Returns
    -------
    float
        Bit error rate as a percentage (0-100).
    """
    golden = np.asarray(golden)
    noisy = np.asarray(noisy)
    total = golden.size
    errors = np.sum(golden != noisy)
    return 100.0 * errors / total


def simulate_ecc(noisy: np.ndarray, golden: np.ndarray, t: int) -> float:
    """
    Return the fraction of response vectors whose Hamming distance exceeds `t`,
    i.e. ECC failure rate for a (n, k, t) BCH-style code.
    Assumes inputs are shape (N, R) where R is response length.

    Parameters
    ----------
    noisy : np.ndarray
        Noisy response vectors, shape (N, R).
    golden : np.ndarray
        Golden/reference response vectors, shape (N, R).
    t : int
        Error-correcting capability (max correctable errors).

    Returns
    -------
    float
        Fraction of vectors with HD > t (ECC failure rate).
    """
    hd = hamming(noisy, golden)
    failures = np.sum(hd > t)
    return failures / len(hd)

def _set_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply common plot styling."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_uniqueness(x: np.ndarray, y: np.ndarray) -> Figure:
    """
    Plot uniqueness metric vs. some variable.
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)
    _set_style(ax, "PUF Uniqueness", "Variable", "Uniqueness (%)")
    return fig


def plot_reliability_vs_temperature(temperatures: np.ndarray, reliability: np.ndarray) -> Figure:
    """
    Plot reliability vs. temperature.
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(temperatures, reliability)
    _set_style(ax, "PUF Reliability vs. Temperature", "Temperature (°C)", "Reliability (%)")
    return fig


def plot_attack_accuracy(x: np.ndarray, accuracy: np.ndarray) -> Figure:
    """
    Plot attack accuracy vs. some variable (e.g., #CRPs).
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(x, accuracy)
    _set_style(ax, "Attack Accuracy", "Variable", "Accuracy (%)")
    return fig


def plot_ecc_comparison(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, label1: str = "ECC1", label2: str = "ECC2") -> Figure:
    """
    Plot comparison of two ECC schemes.
    Returns the matplotlib Figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=label1)
    ax.plot(x, y2, label=label2)
    ax.legend()
    _set_style(ax, "ECC Comparison", "Variable", "Failure Rate (%)")
    return fig

if __name__ == "__main__":
    # Self-tests
    # Hamming: identical arrays
    a = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    b = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    assert np.all(hamming(a, b) == 0)
    # Hamming: ±1 encoding
    a_pm = np.array([[1, -1, 1, -1]])
    b_pm = np.array([[1, -1, -1, 1]])
    assert np.all(hamming(a_pm, b_pm) == 2)
    # BER: identical
    golden = np.random.randint(0, 2, size=100)
    assert bit_error_rate(golden, golden) == 0.0
    # Uniqueness: two random PUFs, 50 ± 1 %
    rng = np.random.default_rng(123)
    n_chal = 1000
    n_bits = 64
    chal = rng.integers(0, 2, size=(n_chal, n_bits))
    resp1 = rng.integers(0, 2, size=(n_chal,))
    resp2 = rng.integers(0, 2, size=(n_chal,))
    uniq = uniqueness(chal, np.stack([resp1, resp2]))
    print(f"Uniqueness (random PUFs): {uniq:.2f}%")
    assert 49.0 <= uniq <= 51.0, f"Uniqueness out of expected range: {uniq}"

    # ECC demo: 10 noisy clones, 3% flipped bits, t=4, 128-bit responses
    rng = np.random.default_rng(42)
    N = 10
    R = 128
    golden = rng.integers(0, 2, size=(N, R))
    noisy = golden.copy()
    # Flip 3% of bits in each row
    for i in range(N):
        flip_idx = rng.choice(R, size=int(0.03 * R), replace=False)
        noisy[i, flip_idx] ^= 1
    t = 4
    fail_rate = simulate_ecc(noisy, golden, t)
    print(f"ECC failure rate (t={t}, 3% flips, 128 bits, N=10): {fail_rate:.2f}")
    # Hand-check math on tiny example
    golden_tiny = np.array([[0, 1, 1, 0], [1, 0, 1, 1]])
    noisy_tiny = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])  # HDs: 1, 1
    assert np.isclose(simulate_ecc(noisy_tiny, golden_tiny, t=0), 1.0)  # All HD>0
    assert np.isclose(simulate_ecc(noisy_tiny, golden_tiny, t=1), 0.0)  # All HD<=1

    # Smoke test for plotting utilities
    import matplotlib
    matplotlib.use('Agg')  # For headless environments
    dummy_x = np.linspace(0, 10, 20)
    dummy_y = np.random.rand(20) * 100
    fig1 = plot_uniqueness(dummy_x, dummy_y)
    fig2 = plot_reliability_vs_temperature(dummy_x, 100 - dummy_y)
    fig3 = plot_attack_accuracy(dummy_x, dummy_y)
    fig4 = plot_ecc_comparison(dummy_x, dummy_y, 100 - dummy_y, label1="BCH", label2="RS")
    # Save to buffer to ensure no exceptions
    import io
    for i, fig in enumerate([fig1, fig2, fig3, fig4], 1):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.close()
    print("Plotting smoke test passed.")
