import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional, Any
from puf_models import ArbiterPUF

class MLAttacker:
    """
    Machine Learning attacker for Arbiter PUFs using logistic regression.

    Parameters
    ----------
    n_stages : int
        Number of PUF stages (challenge length).
    C : float, optional
        Inverse of regularization strength for LogisticRegression.
    penalty : str, optional
        Regularization type ('l2', etc.).
    kwargs : dict
        Additional arguments for LogisticRegression.

    Methods
    -------
    train(challenges, responses):
        Fit the ML model to CRPs.
    predict(challenges):
        Predict responses (+1/-1) for given challenges.
    accuracy(X, y):
        Compute accuracy on given CRPs.
    feature_map(challenges):
        Static method. Maps challenges to parity features Φ(C).

    Examples
    --------
    >>> from puf_models import ArbiterPUF
    >>> rng = np.random.default_rng(42)
    >>> n_stages = 64
    >>> puf = ArbiterPUF(n_stages, seed=1)
    >>> challenges = rng.integers(0, 2, size=(1000, n_stages))
    >>> responses = puf.eval(challenges)
    >>> attacker = MLAttacker(n_stages)
    >>> attacker.train(challenges, responses)
    >>> acc = attacker.accuracy(challenges, responses)
    >>> print(f"Training accuracy: {acc:.3f}")
    Training accuracy: 1.000
    """
    def __init__(self, n_stages: int, C: float = 1.0, penalty: str = 'l2', **kwargs: Any) -> None:
        self.n_stages = n_stages
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=5000,
            C=C,
            penalty=penalty,
            **kwargs
        )
        self._is_fitted = False

    @staticmethod
    def feature_map(challenges: np.ndarray) -> np.ndarray:
        """
        Map challenges to parity feature vectors Φ(C) using ArbiterPUF logic.
        """
        challenges = np.asarray(challenges)
        if challenges.ndim == 1:
            challenges = challenges.reshape(1, -1)
        n_challenges, n_stages = challenges.shape
        phi = np.empty((n_challenges, n_stages))
        for i, ch in enumerate(challenges):
            phi[i] = MLAttacker._parity_transform(ch)
        return phi

    @staticmethod
    def _parity_transform(ch: np.ndarray) -> np.ndarray:
        n_stages = ch.shape[0]
        phi = np.empty(n_stages)
        for i in range(n_stages):
            prod = 1.0
            for j in range(i, n_stages):
                prod *= 1 - 2 * ch[j]
            phi[i] = prod
        return phi

    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        """
        Fit the ML model to CRPs.
        """
        X = self.feature_map(challenges)
        y = np.asarray(responses)
        # Convert responses to {0,1} for logistic regression, then back to ±1
        y_bin = (y > 0).astype(int)
        self.model.fit(X, y_bin)
        self._is_fitted = True

    def predict(self, challenges: np.ndarray) -> np.ndarray:
        """
        Predict responses (+1/-1) for given challenges.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained yet.")
        X = self.feature_map(challenges)
        y_pred_bin = self.model.predict(X)
        return np.where(y_pred_bin == 1, 1, -1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on given CRPs.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return np.mean(y_pred == y_true)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Explicit run to show reproducibility
    from puf_models import ArbiterPUF
    rng = np.random.default_rng(42)
    n_stages = 64
    puf = ArbiterPUF(n_stages, seed=1)
    challenges = rng.integers(0, 2, size=(1000, n_stages))
    responses = puf.eval(challenges)
    attacker = MLAttacker(n_stages)
    attacker.train(challenges, responses)
    acc = attacker.accuracy(challenges, responses)
    print(f"Training accuracy: {acc:.3f}")
    assert acc >= 0.99, "Training accuracy should be >= 99%"
