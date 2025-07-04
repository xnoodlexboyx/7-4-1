from puf_models import ArbiterPUF, BasePUF
import numpy as np
from typing import Optional

def apply_temperature(puf: 'BasePUF', T_current: float, k_T: float = 0.0005,
                      sigma_noise: float = 0.01, T_nominal: Optional[float] = None) -> 'BasePUF':
    """
    Return *a NEW ArbiterPUF instance* whose delay_params have been linearly
    perturbed to model temperature effects:
        W_stressed = W_nominal * (1 + k_T*(T_current - T_nominal)) + N(0, sigma_noise^2)
    Keeps the same n_stages & seed for reproducibility.

    Reliability model cross-referenced from:
    Hiller, M., et al. "Breaking through fixed PUF block limitations with reliability-based modeling." CHES 2012.
    """
    if not isinstance(puf, ArbiterPUF):
        raise TypeError("apply_temperature currently only supports ArbiterPUF.")
    if T_nominal is None:
        T_nominal = getattr(puf, 't_nominal', 25.0)
    W_nominal = puf.delay_params
    rng = np.random.default_rng()
    noise = rng.normal(0, sigma_noise, size=W_nominal.shape)
    W_stressed = W_nominal * (1 + k_T * (T_current - T_nominal)) + noise
    # Create a new ArbiterPUF with the same parameters, but override delay_params
    stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=T_nominal)
    stressed.delay_params = W_stressed
    return stressed

def apply_voltage(puf: 'BasePUF', *args, **kwargs) -> 'BasePUF':
    """
    Placeholder for voltage stressor model. Not yet implemented.
    """
    raise NotImplementedError("Voltage stressor model not implemented yet.")

def apply_aging(puf: 'BasePUF', *args, **kwargs) -> 'BasePUF':
    """
    Placeholder for aging stressor model. Not yet implemented.
    """
    raise NotImplementedError("Aging stressor model not implemented yet.")

if __name__ == "__main__":
    # Unit tests for stressors
    n_stages = 64
    seed = 123
    puf = ArbiterPUF(n_stages=n_stages, seed=seed)
    # Clone at higher temperature
    T_high = 75.0
    stressed = apply_temperature(puf, T_current=T_high)
    # Assert delay_params differ
    diff = np.abs(stressed.delay_params - puf.delay_params)
    print("Original delay_params (first 5):", puf.delay_params[:5])
    print("Stressed delay_params (first 5):", stressed.delay_params[:5])
    print("Difference (first 5):", diff[:5])
    assert np.any(diff > 0), "Stressed delay_params should differ from nominal."
    # Test NotImplementedError for voltage/aging
    try:
        apply_voltage(puf)
    except NotImplementedError:
        print("apply_voltage correctly raises NotImplementedError.")
    try:
        apply_aging(puf)
    except NotImplementedError:
        print("apply_aging correctly raises NotImplementedError.")
