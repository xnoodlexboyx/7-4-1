import os
import numpy as np
from typing import Generator, Tuple
import argparse
import json
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "ppet-thesis")
from puf_models import ArbiterPUF
from stressors import apply_temperature
from attacks import MLAttacker
from analysis import (
    bit_error_rate, uniqueness, plot_reliability_vs_temperature,
    plot_attack_accuracy, plot_ecc_comparison
)

# Top-level constants
N_STAGES = 64
N_CHAL = 10000
TEMPS = [-20, 0, 25, 50, 75, 100]
ECC_T = 4
DATA_DIR = "data"
FIG_DIR = "figures"
CHALLENGES_PATH = os.path.join(DATA_DIR, "challenges.npy")
GOLDEN_PATH = os.path.join(DATA_DIR, "responses_golden.npy")
RESULTS_PATH = os.path.join(DATA_DIR, "results.json")


def load_or_generate_data(force_regen: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load challenges and golden responses from .npy files, or generate and save them if missing or forced.
    Challenges: random ±1, shape (N_CHAL, N_STAGES)
    Golden responses: from a fresh ArbiterPUF
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not force_regen and os.path.exists(CHALLENGES_PATH) and os.path.exists(GOLDEN_PATH):
        print(f"Loading existing data from {CHALLENGES_PATH} and {GOLDEN_PATH}.")
        challenges = np.load(CHALLENGES_PATH)
        golden = np.load(GOLDEN_PATH)
    else:
        print("Generating new challenges and golden responses...")
        rng = np.random.default_rng(42)
        challenges = rng.integers(0, 2, size=(N_CHAL, N_STAGES))
        puf = ArbiterPUF(N_STAGES, seed=123)
        golden = puf.eval(challenges)
        np.save(CHALLENGES_PATH, challenges)
        np.save(GOLDEN_PATH, golden)
        print(f"Saved new data to {CHALLENGES_PATH} and {GOLDEN_PATH}.")
    return challenges, golden


def run_temperature_sweep(challenges: np.ndarray, golden: np.ndarray) -> Generator[
    Tuple[int, np.ndarray, float, float], None, None]:
    """
    For each temperature, yield (T, noisy_responses, attack_accuracy, ecc_fail_rate).
    """
    puf = ArbiterPUF(N_STAGES, seed=123)
    for T in TEMPS:
        stressed = apply_temperature(puf, T_current=T)
        noisy = stressed.eval(challenges)
        # Train attacker on noisy data
        attacker = MLAttacker(N_STAGES)
        attacker.train(challenges, noisy)
        acc = attacker.accuracy(challenges, noisy)
        # ECC fail rate
        # For demo, treat golden as reference, noisy as received
        # Assume responses are ±1, convert to 0/1 for ECC
        golden_bin = (golden > 0).astype(int)
        noisy_bin = (noisy > 0).astype(int)
        ecc_fail = 0.0
        if golden_bin.shape == noisy_bin.shape:
            from analysis import simulate_ecc
            BLOCK_SIZE = 128  # or 64 if you want to use N_STAGES
            num_blocks = len(noisy_bin) // BLOCK_SIZE
            noisy_blocks = noisy_bin[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
            golden_blocks = golden_bin[:num_blocks * BLOCK_SIZE].reshape(num_blocks, BLOCK_SIZE)
            ecc_fail = simulate_ecc(noisy_blocks, golden_blocks, ECC_T)
        yield T, noisy, acc, ecc_fail


def main():
    import os
    import numpy as np
    import pandas as pd
    import argparse
    import json
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "ppet-thesis")
    from puf_models import ArbiterPUF
    from stressors import apply_temperature
    from attacks import MLAttacker
    from analysis import (
        bit_error_rate, uniqueness, plot_reliability_vs_temperature,
        plot_attack_accuracy, plot_ecc_comparison
    )
    parser = argparse.ArgumentParser(description="PUF main experiment runner")
    parser.add_argument('--regenerate', action='store_true', help='Force regeneration of challenges and golden responses')
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    challenges, golden = load_or_generate_data(force_regen=args.regenerate)
    temps = []
    accs = []
    eccs = []
    bers = []
    results = []
    for T, noisy, train_acc, ecc_fail in run_temperature_sweep(challenges, golden):
        # Train new attacker on noisy, evaluate on golden
        attacker = MLAttacker(N_STAGES)
        attacker.train(challenges, noisy)
        test_acc = attacker.accuracy(challenges, golden)
        ber = bit_error_rate(golden, noisy)
        print(f"Temp {T}°C: Train acc={train_acc:.3f}, Test acc={test_acc:.3f}, BER={ber:.3f}%, ECC fail={ecc_fail:.3f}")
        temps.append(T)
        accs.append(train_acc * 100)
        eccs.append(ecc_fail * 100)
        bers.append(ber)
        results.append({
            "temperature": T,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "bit_error_rate": float(ber),
            "ecc_failure_rate": float(ecc_fail),
        })
    # ECC Performance vs. Temperature summary plot
    plt.figure(figsize=(10, 6))
    plt.plot(temps, [b * 100 for b in bers], marker='o', linestyle='-', label='Raw (Noisy) Response')
    plt.plot(temps, eccs, marker='x', linestyle='--', label='Post-ECC Corrected Response')
    plt.title('ECC Performance vs. Temperature', fontsize=16)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Failure Rate (%)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.ylim(-1, max([b * 100 for b in bers] + eccs) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ecc_performance_vs_temperature.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # ML Attack Accuracy vs. Temperature summary plot
    plt.figure(figsize=(10, 6))
    plt.plot(temps, accs, marker='o', linestyle='-', color='r')
    plt.title('ML Attack Accuracy vs. Temperature', fontsize=16)
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Model Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.ylim(80, 101)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ml_attack_accuracy_vs_temperature.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Optionally, keep reliability and attack accuracy summary plots for other uses
    fig3 = plot_reliability_vs_temperature(np.array(temps), 100 - np.array(eccs))
    fname3 = os.path.join(FIG_DIR, "reliability_vs_temperature.png")
    fig3.savefig(fname3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    fig4 = plot_attack_accuracy(np.array(temps), np.array(accs))
    fname4 = os.path.join(FIG_DIR, "attack_accuracy_vs_temperature.png")
    fig4.savefig(fname4, dpi=300, bbox_inches='tight')
    plt.close(fig4)
    # Dump results to JSON
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics for {len(results)} temperatures to {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
