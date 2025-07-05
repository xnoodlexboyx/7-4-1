import numpy as np
import argparse
import os
import sys
sys.path.insert(0, "ppet-thesis")
from ppet.puf_models import ArbiterPUF

def main():
    parser = argparse.ArgumentParser(description="Generate PUF challenge/response data.")
    parser.add_argument('--n_chal', type=int, default=10000, help='Number of challenges (default: 10000)')
    parser.add_argument('--n_stages', type=int, default=64, help='Challenge length / number of PUF stages (default: 64)')
    args = parser.parse_args()

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    challenges_path = os.path.join(data_dir, 'challenges.npy')
    responses_path = os.path.join(data_dir, 'responses_golden.npy')

    rng = np.random.default_rng(42)
    challenges = rng.integers(0, 2, size=(args.n_chal, args.n_stages))
    puf = ArbiterPUF(args.n_stages, seed=123)
    responses = puf.eval(challenges)

    np.save(challenges_path, challenges)
    np.save(responses_path, responses)

    print(f"Saved {args.n_chal} challenges to {challenges_path}")
    print(f"Saved {args.n_chal} golden responses to {responses_path}")
    sys.exit(0)

if __name__ == "__main__":
    main() 