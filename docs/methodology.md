# Methodology

## 1. System Overview Diagram

The overall system architecture for our PUF-based security evaluation is illustrated in the following diagram:

**[See: `figures/system_overview.png`]**

This diagram depicts the flow from challenge generation, through the Arbiter PUF, to the application of environmental stressors (such as temperature), and finally to the evaluation and attack modules. The system is modular, allowing for the simulation of various stressors and the application of machine learning attacks and error-correcting code (ECC) post-processing.

---

## 2. Detailed Mathematics of the Arbiter PUF

The Arbiter PUF is a delay-based strong PUF architecture, first introduced by Suh & Devadas (2007). It consists of a series of multiplexers (stages), each controlled by a challenge bit. The challenge determines the path of a signal through the PUF, and the final output is determined by which path is faster, as measured by an arbiter at the end.

### Mathematical Model

Let $C \in \{0,1\}^n$ be the challenge vector of length $n$ (where $n$ is the number of stages). The Arbiter PUF can be modeled as a linear function in a transformed feature space:

\[
\Delta = \mathbf{W} \cdot \Phi(C)
\]

where:
- $\Delta$ is the total delay difference between the two paths.
- $\mathbf{W} \in \mathbb{R}^n$ is the vector of stage delay parameters (unique to each PUF instance).
- $\Phi(C) \in \{-1, +1\}^n$ is the parity feature vector derived from the challenge.

The response $r$ is then:

\[
r = \mathrm{sign}(\Delta) = \begin{cases} +1 & \text{if } \Delta > 0 \\ -1 & \text{otherwise} \end{cases}
\]

### Parity Feature Vector

The parity feature vector $\Phi(C)$ is defined as:

\[
\Phi_i(C) = \prod_{j=i}^n (1 - 2c_j)
\]

where $c_j$ is the $j$-th bit of the challenge.

### Code Snippet: ArbiterPUF Implementation

```python
class ArbiterPUF(BasePUF):
    def __init__(self, n_stages: int, seed: Optional[int] = None, t_nominal: float = 25.0) -> None:
        self.n_stages = n_stages
        self.t_nominal = t_nominal
        rng = np.random.default_rng(seed)
        self.delay_params = rng.standard_normal(n_stages)

    def _transform_challenge(self, ch: np.ndarray) -> np.ndarray:
        phi = np.empty(self.n_stages)
        for i in range(self.n_stages):
            prod = 1.0
            for j in range(i, self.n_stages):
                prod *= 1 - 2 * ch[j]
            phi[i] = prod
        return phi

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        responses = np.empty(len(challenges), dtype=int)
        for i, ch in enumerate(challenges):
            phi = self._transform_challenge(ch)
            delay = np.dot(self.delay_params, phi)
            responses[i] = 1 if delay > 0 else -1
        return responses
```

---

## 3. Temperature Stressor Model Derivation

Physical Unclonable Functions are sensitive to environmental conditions, especially temperature. To simulate the effect of temperature on the Arbiter PUF, we use a linear perturbation model inspired by Hiller et al. (CHES 2012):

\[
\mathbf{W}_\text{stressed} = \mathbf{W}_\text{nominal} \cdot \left[1 + k_T (T_\text{current} - T_\text{nominal})\right] + \mathcal{N}(0, \sigma_\text{noise}^2)
\]

where:
- $k_T$ is the temperature sensitivity coefficient (empirically set, e.g., $0.0005$).
- $T_\text{current}$ is the current temperature.
- $T_\text{nominal}$ is the reference temperature (typically $25^\circ$C).
- $\mathcal{N}(0, \sigma_\text{noise}^2)$ is Gaussian noise modeling random environmental effects.

This model allows us to generate a new PUF instance whose delay parameters reflect the impact of temperature.

### Code Snippet: Temperature Stressor

```python
def apply_temperature(puf: 'BasePUF', T_current: float, k_T: float = 0.0005,
                      sigma_noise: float = 0.01, T_nominal: float | None = None) -> 'BasePUF':
    """
    Return *a NEW ArbiterPUF instance* whose delay_params have been linearly
    perturbed to model temperature effects:
        W_stressed = W_nominal * (1 + k_T*(T_current - T_nominal)) + N(0, sigma_noise^2)
    Keeps the same n_stages & seed for reproducibility.
    """
    if not isinstance(puf, ArbiterPUF):
        raise TypeError("apply_temperature currently only supports ArbiterPUF.")
    if T_nominal is None:
        T_nominal = getattr(puf, 't_nominal', 25.0)
    W_nominal = puf.delay_params
    rng = np.random.default_rng()
    noise = rng.normal(0, sigma_noise, size=W_nominal.shape)
    W_stressed = W_nominal * (1 + k_T * (T_current - T_nominal)) + noise
    stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=T_nominal)
    stressed.delay_params = W_stressed
    return stressed
```

---

## 4. Logistic Regression Attack Description

Machine learning attacks, particularly logistic regression, are highly effective against linear PUF models like the Arbiter PUF. The attacker collects a set of challenge-response pairs (CRPs) and learns a model that predicts the PUF's response to new challenges.

### Attack Workflow
1. **Feature Extraction:** The attacker computes the parity feature vector $\Phi(C)$ for each challenge, mirroring the PUF's internal logic.
2. **Model Training:** Using the observed CRPs, the attacker fits a logistic regression model:
   \[
   P(r = +1 | C) = \sigma(\mathbf{w}^T \Phi(C))
   \]
   where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function.
3. **Prediction:** The trained model predicts the response for new challenges by thresholding the output probability at 0.5.

### Code Snippet: MLAttacker

```python
class MLAttacker:
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
        # ...parity transform logic...

    def train(self, challenges: np.ndarray, responses: np.ndarray) -> None:
        X = self.feature_map(challenges)
        y = (np.asarray(responses) > 0).astype(int)
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, challenges: np.ndarray) -> np.ndarray:
        X = self.feature_map(challenges)
        y_pred_bin = self.model.predict(X)
        return np.where(y_pred_bin == 1, 1, -1)
```

This approach is effective because the Arbiter PUF is (ideally) a linear threshold function in the parity feature space, making it vulnerable to linear classifiers.

---

## 5. ECC-Simulation Rationale

In real-world deployments, PUF responses are subject to noise and environmental variation, leading to bit errors. To mitigate this, error-correcting codes (ECC) are used to recover the original response, provided the number of errors does not exceed the code's correction capability $t$.

### ECC Failure Rate Simulation

We simulate the ECC failure rate by grouping response bits into blocks (e.g., 128 bits), computing the Hamming distance between the noisy and golden (reference) responses, and counting the fraction of blocks where the number of errors exceeds $t$:

\[
\text{Failure Rate} = \frac{\#\{\text{blocks with HD} > t\}}{\text{total blocks}}
\]

This metric quantifies the likelihood that a block cannot be corrected by the ECC, which is critical for system reliability.

### Code Snippet: ECC Simulation

```python
def simulate_ecc(noisy: np.ndarray, golden: np.ndarray, t: int) -> float:
    """
    Return the fraction of response vectors whose Hamming distance exceeds `t`,
    i.e. ECC failure rate for a (n, k, t) BCH-style code.
    Assumes inputs are shape (N, R) where R is response length.
    """
    hd = hamming(noisy, golden)
    failures = np.sum(hd > t)
    return failures / len(hd)
```

---

## References

- Suh, G. E., & Devadas, S. (2007). Physical unclonable functions for device authentication and secret key generation. *Proceedings of the 44th annual Design Automation Conference* (DAC).
- Hiller, M., et al. (2012). Breaking through fixed PUF block limitations with reliability-based modeling. *CHES 2012*.
- Rührmair, U., et al. (2010). Modeling attacks on physical unclonable functions. *ACM Conference on Computer and Communications Security* (CCS). 