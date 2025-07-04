import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional
import json

class BasePUF(ABC):
    """
    Abstract base class for Physical Unclonable Functions (PUFs).
    Defines the interface for evaluating a PUF on a set of challenges.
    """
    @abstractmethod
    def eval(self, challenges: np.ndarray) -> np.ndarray:
        """
        Evaluate the PUF on the given challenges.

        Parameters
        ----------
        challenges : np.ndarray
            Array of challenges, shape (n_challenges, n_stages).

        Returns
        -------
        np.ndarray
            Array of responses (+1 or -1), shape (n_challenges,).
        """
        pass

class ArbiterPUF(BasePUF):
    """
    Implements the Arbiter PUF as described by Suh & Devadas (2007).

    References
    ----------
    Suh, G. E., & Devadas, S. (2007). Physical unclonable functions for device authentication and secret key generation. In DAC '07.
    """
    def __init__(self, n_stages: int, seed: Optional[int] = None, t_nominal: float = 25.0) -> None:
        """
        Initialize an Arbiter PUF instance.

        Parameters
        ----------
        n_stages : int
            Number of PUF stages (challenge length).
        seed : int or None, optional
            Random seed for reproducibility.
        t_nominal : float, optional
            Nominal temperature (unused, for future extensions).
        """
        self.n_stages = n_stages
        self.t_nominal = t_nominal
        rng = np.random.default_rng(seed)
        self.delay_params = rng.standard_normal(n_stages)

    def _transform_challenge(self, ch: np.ndarray) -> np.ndarray:
        """
        Transform a challenge into the parity feature vector Φ(C).

        Parameters
        ----------
        ch : np.ndarray
            Challenge vector, shape (n_stages,).

        Returns
        -------
        np.ndarray
            Parity feature vector, shape (n_stages,).
        """
        # Φ_i(C) = Π_{j=i}^n (1 - 2c_j)
        # c_j in {0, 1}
        phi = np.empty(self.n_stages)
        prod = 1.0
        for i in range(self.n_stages):
            prod = 1.0
            for j in range(i, self.n_stages):
                prod *= 1 - 2 * ch[j]
            phi[i] = prod
        return phi

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        """
        Evaluate the Arbiter PUF on a batch of challenges.

        Parameters
        ----------
        challenges : np.ndarray
            Array of challenges, shape (n_challenges, n_stages), values in {0, 1}.

        Returns
        -------
        np.ndarray
            Array of responses (+1 or -1), shape (n_challenges,).
        """
        challenges = np.asarray(challenges)
        if challenges.ndim == 1:
            challenges = challenges.reshape(1, -1)
        n_challenges, n_stages = challenges.shape
        assert n_stages == self.n_stages, "Challenge length mismatch."
        responses = np.empty(n_challenges, dtype=int)
        for i, ch in enumerate(challenges):
            phi = self._transform_challenge(ch)
            delay = np.dot(self.delay_params, phi)
            responses[i] = 1 if delay > 0 else -1  # Never 0
        return responses

    def to_json(self) -> str:
        """
        Serialize the PUF parameters to a JSON string.
        """
        return json.dumps({
            "n_stages": self.n_stages,
            "t_nominal": self.t_nominal,
            "delay_params": self.delay_params.tolist(),
        })

    @classmethod
    def from_json(cls, data: str) -> "ArbiterPUF":
        """
        Deserialize an ArbiterPUF from a JSON string.
        """
        obj = json.loads(data)
        inst = cls(obj["n_stages"], t_nominal=obj["t_nominal"])
        inst.delay_params = np.array(obj["delay_params"])
        return inst

if __name__ == "__main__":
    # Demo: Instantiate a 64-stage Arbiter PUF, evaluate 5 random challenges, print responses
    n_stages = 64
    puf = ArbiterPUF(n_stages=n_stages, seed=42)
    rng = np.random.default_rng(123)
    challenges = rng.integers(0, 2, size=(5, n_stages))
    responses = puf.eval(challenges)
    print("Random challenges:\n", challenges)
    print("PUF responses:", responses)
