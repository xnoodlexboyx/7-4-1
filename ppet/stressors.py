from .puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF, BasePUF
import numpy as np
from typing import Optional
import copy

def apply_temperature(puf: 'BasePUF', T_current: float, k_T: float = 0.0005,
                      sigma_noise: float = 0.01, T_nominal: Optional[float] = None,
                      military_spec: bool = False) -> 'BasePUF':
    """
    Apply temperature stress to any PUF type. Returns a new PUF instance with
    temperature-perturbed parameters.
    
    Temperature effects:
    - ArbiterPUF: Linear perturbation of delay parameters
    - SRAMPUF: Threshold voltage shift and noise sensitivity changes
    - RingOscillatorPUF: Frequency drift and stability degradation
    - ButterflyPUF: Metastability time and crosstalk changes
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to stress
    T_current : float
        Current temperature in Celsius (-55°C to +125°C for military spec)
    k_T : float, optional
        Temperature coefficient (enhanced for military spec)
    sigma_noise : float, optional
        Gaussian noise standard deviation
    T_nominal : float, optional
        Nominal temperature (defaults to puf.t_nominal)
    military_spec : bool, optional
        Use military-grade temperature modeling (MIL-STD-810)
        
    Returns
    -------
    BasePUF
        New stressed PUF instance
    """
    if T_nominal is None:
        T_nominal = getattr(puf, 't_nominal', 25.0)
    
    temp_delta = T_current - T_nominal
    rng = np.random.default_rng(42)
    
    # Military-spec temperature modeling (MIL-STD-810)
    if military_spec:
        # Enhanced temperature coefficients for extreme conditions
        if T_current < -40:  # Arctic conditions
            k_T = k_T * 1.8  # 80% increase in sensitivity
            sigma_noise = sigma_noise * 2.0  # Double noise floor
        elif T_current > 85:  # Desert/engine compartment conditions
            k_T = k_T * 1.5  # 50% increase in sensitivity
            sigma_noise = sigma_noise * 1.5  # 50% increase in noise
        
        # Non-linear temperature effects for extreme ranges
        if abs(temp_delta) > 60:  # Beyond normal operating range
            temp_delta = temp_delta * (1 + 0.001 * abs(temp_delta))  # Quadratic effect
    
    if isinstance(puf, ArbiterPUF):
        # Original arbiter PUF temperature model
        W_nominal = puf.delay_params
        noise = rng.normal(0, sigma_noise, size=W_nominal.shape)
        W_stressed = W_nominal * (1 + k_T * temp_delta) + noise
        stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=T_current)
        stressed.delay_params = W_stressed
        return stressed
        
    elif isinstance(puf, SRAMPUF):
        # SRAM PUF temperature effects: threshold voltage shift
        stressed = SRAMPUF(puf.n_cells, seed=None, t_nominal=T_current,
                          radiation_hardening=puf.radiation_hardening,
                          low_power_mode=puf.low_power_mode)
        
        # Temperature affects threshold voltages and noise sensitivity
        temp_factor = 1 + k_T * temp_delta
        noise = rng.normal(0, sigma_noise * 10, size=puf.vth_variations.shape)  # mV scale
        stressed.vth_variations = puf.vth_variations * temp_factor + noise
        
        # Noise sensitivity increases with temperature
        noise_factor = 1 + 0.02 * abs(temp_delta)  # 2% per degree
        stressed.noise_sensitivity = puf.noise_sensitivity * noise_factor
        return stressed
        
    elif isinstance(puf, RingOscillatorPUF):
        # Ring Oscillator temperature effects: frequency drift
        stressed = RingOscillatorPUF(puf.n_rings, seed=None, t_nominal=T_current,
                                   emi_resistance=puf.emi_resistance,
                                   freq_stability=puf.freq_stability)
        
        # Temperature coefficient for frequency (typically -50 to -100 ppm/°C)
        freq_temp_coeff = -75e-6  # -75 ppm/°C
        temp_factor = 1 + freq_temp_coeff * temp_delta
        noise = rng.normal(0, sigma_noise * 0.5, size=puf.base_frequencies.shape)  # MHz scale
        
        stressed.base_frequencies = puf.base_frequencies * temp_factor + noise
        stressed.process_variations = puf.process_variations.copy()
        
        # Frequency stability degrades with extreme temperatures
        stability_degradation = 1 - 0.001 * abs(temp_delta)  # 0.1% per degree
        stressed.freq_stability = max(0.1, puf.freq_stability * stability_degradation)
        
        # EMI susceptibility may change with temperature
        stressed.emi_susceptibility = puf.emi_susceptibility * (1 + 0.001 * abs(temp_delta))
        return stressed
        
    elif isinstance(puf, ButterflyPUF):
        # Butterfly PUF temperature effects: metastability and crosstalk changes
        stressed = ButterflyPUF(puf.n_butterflies, seed=None, t_nominal=T_current,
                              metastability_time=puf.metastability_time,
                              crosstalk_resistance=puf.crosstalk_resistance)
        
        # Temperature affects latch imbalances and settling times
        temp_factor = 1 + k_T * temp_delta
        noise = rng.normal(0, sigma_noise * 0.05, size=puf.latch_imbalances.shape)
        stressed.latch_imbalances = puf.latch_imbalances * temp_factor + noise
        
        # Settling times change with temperature (faster at higher temp)
        settling_temp_factor = 1 - 0.005 * temp_delta  # 0.5% per degree
        stressed.settling_times = puf.settling_times * max(0.1, settling_temp_factor)
        
        # Crosstalk increases with temperature due to increased leakage
        crosstalk_temp_factor = 1 + 0.002 * abs(temp_delta)  # 0.2% per degree
        stressed.crosstalk_factors = puf.crosstalk_factors * crosstalk_temp_factor
        return stressed
    
    else:
        raise TypeError(f"apply_temperature does not support PUF type: {type(puf)}")

def apply_voltage(puf: 'BasePUF', V_current: float, V_nominal: float = 3.3,
                  military_spec: bool = False) -> 'BasePUF':
    """
    Apply voltage stress to any PUF type. Models power supply variations
    and their effects on PUF characteristics.
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to stress
    V_current : float
        Current supply voltage in Volts
    V_nominal : float, optional
        Nominal supply voltage (default 3.3V)
    military_spec : bool, optional
        Use military-grade voltage modeling
        
    Returns
    -------
    BasePUF
        New voltage-stressed PUF instance
    """
    voltage_delta = V_current - V_nominal
    rng = np.random.default_rng(42)
    
    # Military spec: wider voltage tolerance (±10% vs ±5% commercial)
    if military_spec:
        voltage_tolerance = 0.1  # ±10%
        if abs(voltage_delta) > V_nominal * voltage_tolerance:
            # Beyond military spec - severe degradation
            voltage_delta = voltage_delta * 1.5
    
    if isinstance(puf, ArbiterPUF):
        # Voltage affects delay parameters through threshold voltage changes
        k_V = 0.15  # 15% change per volt
        voltage_factor = 1 + k_V * voltage_delta / V_nominal
        noise = rng.normal(0, 0.005, size=puf.delay_params.shape)
        
        stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0))
        stressed.delay_params = puf.delay_params * voltage_factor + noise
        return stressed
        
    elif isinstance(puf, SRAMPUF):
        # SRAM is highly voltage-sensitive
        k_V = 0.25  # 25% change per volt
        voltage_factor = 1 + k_V * voltage_delta / V_nominal
        noise = rng.normal(0, 15, size=puf.vth_variations.shape)  # mV scale
        
        stressed = SRAMPUF(puf.n_cells, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                          radiation_hardening=puf.radiation_hardening,
                          low_power_mode=puf.low_power_mode)
        stressed.vth_variations = puf.vth_variations * voltage_factor + noise
        stressed.noise_sensitivity = puf.noise_sensitivity * (1 + 0.1 * abs(voltage_delta))
        return stressed
        
    elif isinstance(puf, RingOscillatorPUF):
        # Ring oscillator frequency is voltage-dependent
        k_V = 0.2  # 20% frequency change per volt
        voltage_factor = 1 + k_V * voltage_delta / V_nominal
        noise = rng.normal(0, 0.1, size=puf.base_frequencies.shape)  # MHz scale
        
        stressed = RingOscillatorPUF(puf.n_rings, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                                   emi_resistance=puf.emi_resistance,
                                   freq_stability=puf.freq_stability)
        stressed.base_frequencies = puf.base_frequencies * voltage_factor + noise
        stressed.process_variations = puf.process_variations.copy()
        return stressed
        
    elif isinstance(puf, ButterflyPUF):
        # Butterfly PUF settling times are voltage-dependent
        k_V = 0.18  # 18% change per volt
        voltage_factor = 1 + k_V * voltage_delta / V_nominal
        noise = rng.normal(0, 0.02, size=puf.latch_imbalances.shape)
        
        stressed = ButterflyPUF(puf.n_butterflies, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                              metastability_time=puf.metastability_time,
                              crosstalk_resistance=puf.crosstalk_resistance)
        stressed.latch_imbalances = puf.latch_imbalances * voltage_factor + noise
        # Faster settling at higher voltages
        stressed.settling_times = puf.settling_times / voltage_factor
        return stressed
    
    else:
        raise TypeError(f"apply_voltage does not support PUF type: {type(puf)}")

def apply_aging(puf: 'BasePUF', age_hours: float, temperature_history: Optional[float] = None,
                voltage_stress_hours: float = 0.0, military_spec: bool = False) -> 'BasePUF':
    """
    Apply aging stress to any PUF type. Models degradation over time
    due to various wear-out mechanisms.
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to age
    age_hours : float
        Operating time in hours
    temperature_history : float, optional
        Average operating temperature (accelerates aging)
    voltage_stress_hours : float, optional
        Hours of operation at elevated voltage
    military_spec : bool, optional
        Use military-grade aging models (extended lifetime)
        
    Returns
    -------
    BasePUF
        New aged PUF instance
    """
    if temperature_history is None:
        temperature_history = getattr(puf, 't_nominal', 25.0)
    
    # Arrhenius equation for temperature-accelerated aging
    # Ea = 0.7 eV (typical for silicon devices)
    k_B = 8.617e-5  # Boltzmann constant (eV/K)
    Ea = 0.7  # Activation energy (eV)
    T_ref = 298.15  # Reference temperature (25°C in Kelvin)
    T_op = temperature_history + 273.15  # Operating temperature in Kelvin
    
    # Acceleration factor
    AF = np.exp((Ea / k_B) * (1/T_ref - 1/T_op))
    effective_age = age_hours * AF
    
    # Military spec devices have better aging characteristics
    if military_spec:
        effective_age = effective_age * 0.7  # 30% better aging resistance
    
    # Voltage stress acceleration (10x life reduction per 0.1V overstress)
    if voltage_stress_hours > 0:
        voltage_AF = 1 + voltage_stress_hours / 1000  # Simplified model
        effective_age = effective_age * voltage_AF
    
    rng = np.random.default_rng(42)
    
    if isinstance(puf, ArbiterPUF):
        # Aging affects delay parameters through threshold voltage shift
        # and mobility degradation
        aging_factor = 1 + 0.001 * np.sqrt(effective_age / 8760)  # Per year
        drift_noise = rng.normal(0, 0.002, size=puf.delay_params.shape)
        
        stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0))
        stressed.delay_params = puf.delay_params * aging_factor + drift_noise
        return stressed
        
    elif isinstance(puf, SRAMPUF):
        # SRAM aging: threshold voltage shift and increased variability
        aging_factor = 1 + 0.002 * np.sqrt(effective_age / 8760)  # Per year
        drift_noise = rng.normal(0, 5, size=puf.vth_variations.shape)  # mV scale
        
        stressed = SRAMPUF(puf.n_cells, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                          radiation_hardening=puf.radiation_hardening,
                          low_power_mode=puf.low_power_mode)
        stressed.vth_variations = puf.vth_variations * aging_factor + drift_noise
        # Aging increases noise sensitivity
        stressed.noise_sensitivity = puf.noise_sensitivity * (1 + 0.0005 * effective_age / 8760)
        return stressed
        
    elif isinstance(puf, RingOscillatorPUF):
        # Ring oscillator aging: frequency drift and stability degradation
        aging_factor = 1 - 0.0008 * np.sqrt(effective_age / 8760)  # Frequency decreases
        drift_noise = rng.normal(0, 0.05, size=puf.base_frequencies.shape)  # MHz scale
        
        stressed = RingOscillatorPUF(puf.n_rings, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                                   emi_resistance=puf.emi_resistance,
                                   freq_stability=puf.freq_stability)
        stressed.base_frequencies = puf.base_frequencies * aging_factor + drift_noise
        stressed.process_variations = puf.process_variations.copy()
        # Stability degrades with age
        stability_degradation = 1 - 0.0002 * effective_age / 8760
        stressed.freq_stability = max(0.1, puf.freq_stability * stability_degradation)
        return stressed
        
    elif isinstance(puf, ButterflyPUF):
        # Butterfly aging: latch imbalance drift and settling time changes
        aging_factor = 1 + 0.0015 * np.sqrt(effective_age / 8760)  # Per year
        drift_noise = rng.normal(0, 0.01, size=puf.latch_imbalances.shape)
        
        stressed = ButterflyPUF(puf.n_butterflies, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                              metastability_time=puf.metastability_time,
                              crosstalk_resistance=puf.crosstalk_resistance)
        stressed.latch_imbalances = puf.latch_imbalances * aging_factor + drift_noise
        # Settling times increase with age
        stressed.settling_times = puf.settling_times * (1 + 0.0003 * effective_age / 8760)
        return stressed
    
    else:
        raise TypeError(f"apply_aging does not support PUF type: {type(puf)}")

def apply_radiation(puf: 'BasePUF', dose_krad: float, dose_rate_rad_s: float = 0.01,
                    particle_type: str = 'gamma', military_spec: bool = False) -> 'BasePUF':
    """
    Apply radiation stress to any PUF type. Models effects of ionizing radiation
    on PUF characteristics including total ionizing dose (TID) and single-event effects.
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to irradiate
    dose_krad : float
        Total ionizing dose in krad (Si)
    dose_rate_rad_s : float, optional
        Dose rate in rad/s (affects annealing)
    particle_type : str, optional
        Radiation type ('gamma', 'neutron', 'proton', 'heavy_ion')
    military_spec : bool, optional
        Use military/space-grade radiation hardening models
        
    Returns
    -------
    BasePUF
        New radiation-stressed PUF instance
    """
    dose_rad = dose_krad * 1000  # Convert to rad
    rng = np.random.default_rng(42)
    
    # Radiation hardening factor for military/space applications
    hardening_factor = 1.0
    if military_spec:
        hardening_factor = 0.3  # 70% reduction in radiation effects
    
    # Different particles have different damage effectiveness
    damage_factor = {
        'gamma': 1.0,
        'neutron': 2.5,  # Higher displacement damage
        'proton': 1.8,
        'heavy_ion': 3.5  # Highest LET
    }.get(particle_type, 1.0)
    
    effective_dose = dose_rad * damage_factor * hardening_factor
    
    # Dose rate effects (enhanced low dose rate sensitivity)
    if dose_rate_rad_s < 0.001:  # Very low dose rate
        effective_dose *= 1.5  # 50% increase due to annealing suppression
    
    if isinstance(puf, ArbiterPUF):
        # Radiation affects delay parameters through threshold voltage shifts
        # and mobility degradation
        rad_factor = 1 + 0.0001 * effective_dose  # 0.01% per rad
        # Increase noise magnitude so that radiation can flip response bits.
        # Scale linearly with effective dose to ensure neutron damage exceeds
        # gamma damage in a deterministic manner.
        rad_noise = rng.normal(0, 0.0005 * effective_dose,
                              size=puf.delay_params.shape)
        
        stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0))
        stressed.delay_params = puf.delay_params * rad_factor + rad_noise
        return stressed
        
    elif isinstance(puf, SRAMPUF):
        # SRAM is highly radiation-sensitive
        rad_factor = 1 + 0.0002 * effective_dose  # 0.02% per rad
        rad_noise = rng.normal(0, 2 * np.sqrt(effective_dose), size=puf.vth_variations.shape)
        
        stressed = SRAMPUF(puf.n_cells, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                          radiation_hardening=max(0.1, puf.radiation_hardening - 0.001 * effective_dose),
                          low_power_mode=puf.low_power_mode)
        stressed.vth_variations = puf.vth_variations * rad_factor + rad_noise
        # Radiation increases noise sensitivity
        stressed.noise_sensitivity = puf.noise_sensitivity * (1 + 0.00005 * effective_dose)
        return stressed
        
    elif isinstance(puf, RingOscillatorPUF):
        # Ring oscillator frequency shifts due to radiation
        rad_factor = 1 - 0.00008 * effective_dose  # Frequency decreases
        # Increase noise magnitude to better reflect radiation-induced jitter.
        rad_noise = rng.normal(0, 0.0005 * effective_dose,
                              size=puf.base_frequencies.shape)
        
        stressed = RingOscillatorPUF(puf.n_rings, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                                   emi_resistance=puf.emi_resistance,
                                   freq_stability=puf.freq_stability)
        stressed.base_frequencies = puf.base_frequencies * rad_factor + rad_noise
        stressed.process_variations = puf.process_variations.copy()
        # Radiation degrades frequency stability
        stability_degradation = 1 - 0.00002 * effective_dose
        stressed.freq_stability = max(0.1, puf.freq_stability * stability_degradation)
        return stressed
        
    elif isinstance(puf, ButterflyPUF):
        # Butterfly PUF latch imbalances affected by radiation
        rad_factor = 1 + 0.00012 * effective_dose  # 0.012% per rad
        # Increase noise magnitude for observable BER effects
        rad_noise = rng.normal(0, 0.0005 * effective_dose,
                              size=puf.latch_imbalances.shape)
        
        stressed = ButterflyPUF(puf.n_butterflies, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                              metastability_time=puf.metastability_time,
                              crosstalk_resistance=puf.crosstalk_resistance)
        stressed.latch_imbalances = puf.latch_imbalances * rad_factor + rad_noise
        # Radiation increases settling times
        stressed.settling_times = puf.settling_times * (1 + 0.00001 * effective_dose)
        return stressed
    
    else:
        raise TypeError(f"apply_radiation does not support PUF type: {type(puf)}")

def apply_emi(puf: 'BasePUF', frequency_mhz: float, field_strength_v_m: float,
              pulse_duration_us: float = 1.0, military_spec: bool = False) -> 'BasePUF':
    """
    Apply electromagnetic interference (EMI) stress to any PUF type.
    Models effects of electromagnetic fields on PUF operation.
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to stress with EMI
    frequency_mhz : float
        EMI frequency in MHz
    field_strength_v_m : float
        Electric field strength in V/m
    pulse_duration_us : float, optional
        Pulse duration in microseconds
    military_spec : bool, optional
        Use military EMI standards (MIL-STD-461)
        
    Returns
    -------
    BasePUF
        New EMI-stressed PUF instance
    """
    rng = np.random.default_rng(42)
    
    # Military spec has higher EMI resistance thresholds
    if military_spec:
        # MIL-STD-461 thresholds are typically 10x higher
        effective_field = field_strength_v_m / 10
    else:
        effective_field = field_strength_v_m
    
    # Frequency-dependent coupling efficiency
    if frequency_mhz < 1:  # Low frequency
        coupling_factor = 0.1
    elif frequency_mhz < 100:  # Medium frequency
        coupling_factor = 0.5
    elif frequency_mhz < 1000:  # High frequency
        coupling_factor = 1.0
    else:  # Very high frequency
        coupling_factor = 0.8  # Skin effect reduces coupling
    
    # Pulse duration effects (shorter pulses are more disruptive)
    if pulse_duration_us < 0.1:
        pulse_factor = 2.0  # Fast transients
    elif pulse_duration_us < 1.0:
        pulse_factor = 1.5
    else:
        pulse_factor = 1.0
    
    emi_severity = effective_field * coupling_factor * pulse_factor
    
    if isinstance(puf, ArbiterPUF):
        # EMI affects delay parameters through power supply noise
        emi_factor = 1 + 0.001 * emi_severity / 1000  # Normalized to 1000 V/m
        emi_noise = rng.normal(0, 0.002 * emi_severity / 1000, size=puf.delay_params.shape)
        
        stressed = ArbiterPUF(puf.n_stages, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0))
        stressed.delay_params = puf.delay_params * emi_factor + emi_noise
        return stressed
        
    elif isinstance(puf, SRAMPUF):
        # SRAM is moderately EMI-sensitive
        emi_factor = 1 + 0.0008 * emi_severity / 1000
        emi_noise = rng.normal(0, 3 * emi_severity / 1000, size=puf.vth_variations.shape)
        
        stressed = SRAMPUF(puf.n_cells, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                          radiation_hardening=puf.radiation_hardening,
                          low_power_mode=puf.low_power_mode)
        stressed.vth_variations = puf.vth_variations * emi_factor + emi_noise
        stressed.noise_sensitivity = puf.noise_sensitivity * (1 + 0.0002 * emi_severity / 1000)
        return stressed
        
    elif isinstance(puf, RingOscillatorPUF):
        # Ring oscillators are highly EMI-sensitive
        emi_resistance = getattr(puf, 'emi_resistance', 0.5)
        emi_factor = 1 + 0.002 * emi_severity / 1000 * (1 - emi_resistance)
        emi_noise = rng.normal(0, 0.1 * emi_severity / 1000, size=puf.base_frequencies.shape)
        
        stressed = RingOscillatorPUF(puf.n_rings, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                                   emi_resistance=emi_resistance,
                                   freq_stability=puf.freq_stability)
        stressed.base_frequencies = puf.base_frequencies * emi_factor + emi_noise
        stressed.process_variations = puf.process_variations.copy()
        # EMI degrades frequency stability
        stability_degradation = 1 - 0.0005 * emi_severity / 1000
        stressed.freq_stability = max(0.1, puf.freq_stability * stability_degradation)
        return stressed
        
    elif isinstance(puf, ButterflyPUF):
        # Butterfly PUF settling affected by EMI
        emi_factor = 1 + 0.0015 * emi_severity / 1000
        emi_noise = rng.normal(0, 0.005 * emi_severity / 1000, size=puf.latch_imbalances.shape)
        
        stressed = ButterflyPUF(puf.n_butterflies, seed=None, t_nominal=getattr(puf, 't_nominal', 25.0),
                              metastability_time=puf.metastability_time,
                              crosstalk_resistance=puf.crosstalk_resistance)
        stressed.latch_imbalances = puf.latch_imbalances * emi_factor + emi_noise
        # EMI increases settling times
        stressed.settling_times = puf.settling_times * (1 + 0.0003 * emi_severity / 1000)
        return stressed
    
    else:
        raise TypeError(f"apply_emi does not support PUF type: {type(puf)}")

def apply_multi_stress(puf: 'BasePUF', temperature: Optional[float] = None,
                      voltage: Optional[float] = None, age_hours: Optional[float] = None,
                      radiation_krad: Optional[float] = None, emi_field: Optional[float] = None,
                      military_spec: bool = False) -> 'BasePUF':
    """
    Apply multiple environmental stressors simultaneously to model
    realistic operational conditions.
    
    Parameters
    ----------
    puf : BasePUF
        PUF instance to stress
    temperature : float, optional
        Temperature in Celsius
    voltage : float, optional
        Supply voltage in Volts
    age_hours : float, optional
        Operating age in hours
    radiation_krad : float, optional
        Radiation dose in krad
    emi_field : float, optional
        EMI field strength in V/m at 100 MHz
    military_spec : bool, optional
        Use military-grade modeling
        
    Returns
    -------
    BasePUF
        New multi-stressed PUF instance
    """
    stressed_puf = puf
    
    # Apply stressors in order of typical operational sequence
    if age_hours is not None:
        stressed_puf = apply_aging(stressed_puf, age_hours, 
                                 temperature_history=temperature,
                                 military_spec=military_spec)
    
    if temperature is not None:
        stressed_puf = apply_temperature(stressed_puf, temperature, 
                                       military_spec=military_spec)
    
    if voltage is not None:
        stressed_puf = apply_voltage(stressed_puf, voltage, 
                                   military_spec=military_spec)
    
    if radiation_krad is not None:
        stressed_puf = apply_radiation(stressed_puf, radiation_krad, 
                                     military_spec=military_spec)
    
    if emi_field is not None:
        stressed_puf = apply_emi(stressed_puf, frequency_mhz=100, 
                               field_strength_v_m=emi_field,
                               military_spec=military_spec)
    
    return stressed_puf

if __name__ == "__main__":
    # Unit tests for stressors - test all PUF types
    print("=== Advanced Stressor Testing ===")
    n_stages = 64
    seed = 123
    T_high = 75.0
    
    # Test Arbiter PUF
    print("\n--- Arbiter PUF Temperature Stress ---")
    arbiter_puf = ArbiterPUF(n_stages=n_stages, seed=seed)
    stressed_arbiter = apply_temperature(arbiter_puf, T_current=T_high)
    diff = np.abs(stressed_arbiter.delay_params - arbiter_puf.delay_params)
    print(f"Original delay_params (first 5): {arbiter_puf.delay_params[:5]}")
    print(f"Stressed delay_params (first 5): {stressed_arbiter.delay_params[:5]}")
    print(f"Difference (first 5): {diff[:5]}")
    assert np.any(diff > 0), "Stressed delay_params should differ from nominal."
    
    # Test SRAM PUF
    print("\n--- SRAM PUF Temperature Stress ---")
    sram_puf = SRAMPUF(n_cells=n_stages, seed=seed, radiation_hardening=0.95)
    stressed_sram = apply_temperature(sram_puf, T_current=T_high)
    vth_diff = np.abs(stressed_sram.vth_variations - sram_puf.vth_variations)
    noise_diff = np.abs(stressed_sram.noise_sensitivity - sram_puf.noise_sensitivity)
    print(f"Original VTH variations (first 5): {sram_puf.vth_variations[:5]}")
    print(f"Stressed VTH variations (first 5): {stressed_sram.vth_variations[:5]}")
    print(f"VTH difference (first 5): {vth_diff[:5]}")
    print(f"Noise sensitivity change: {np.mean(noise_diff):.4f}")
    assert np.any(vth_diff > 0), "SRAM VTH variations should change with temperature."
    assert np.any(noise_diff > 0), "SRAM noise sensitivity should change with temperature."
    
    # Test Ring Oscillator PUF
    print("\n--- Ring Oscillator PUF Temperature Stress ---")
    ro_puf = RingOscillatorPUF(n_rings=n_stages, seed=seed, emi_resistance=0.9)
    stressed_ro = apply_temperature(ro_puf, T_current=T_high)
    freq_diff = np.abs(stressed_ro.base_frequencies - ro_puf.base_frequencies)
    stability_change = abs(stressed_ro.freq_stability - ro_puf.freq_stability)
    print(f"Original frequencies (first 5): {ro_puf.base_frequencies[:5]}")
    print(f"Stressed frequencies (first 5): {stressed_ro.base_frequencies[:5]}")
    print(f"Frequency difference (first 5): {freq_diff[:5]}")
    print(f"Stability change: {stability_change:.4f}")
    assert np.any(freq_diff > 0), "RO frequencies should change with temperature."
    assert stability_change > 0, "RO stability should degrade with temperature."
    
    # Test Butterfly PUF
    print("\n--- Butterfly PUF Temperature Stress ---")
    butterfly_puf = ButterflyPUF(n_butterflies=n_stages, seed=seed, crosstalk_resistance=0.85)
    stressed_butterfly = apply_temperature(butterfly_puf, T_current=T_high)
    imbalance_diff = np.abs(stressed_butterfly.latch_imbalances - butterfly_puf.latch_imbalances)
    settling_diff = np.abs(stressed_butterfly.settling_times - butterfly_puf.settling_times)
    crosstalk_diff = np.abs(stressed_butterfly.crosstalk_factors - butterfly_puf.crosstalk_factors)
    print(f"Original latch imbalances (first 5): {butterfly_puf.latch_imbalances[:5]}")
    print(f"Stressed latch imbalances (first 5): {stressed_butterfly.latch_imbalances[:5]}")
    print(f"Imbalance difference (first 5): {imbalance_diff[:5]}")
    print(f"Settling time change: {np.mean(settling_diff):.4f}")
    print(f"Crosstalk change: {np.mean(crosstalk_diff):.6f}")
    assert np.any(imbalance_diff > 0), "Butterfly latch imbalances should change with temperature."
    assert np.any(settling_diff > 0), "Butterfly settling times should change with temperature."
    assert np.any(crosstalk_diff > 0), "Butterfly crosstalk should change with temperature."
    
    # Test functional behavior changes
    print("\n--- Functional Behavior Test ---")
    rng = np.random.default_rng(456)
    test_challenges = rng.integers(0, 2, size=(10, n_stages))
    
    # Compare responses before and after temperature stress
    for puf_name, original, stressed in [
        ("Arbiter", arbiter_puf, stressed_arbiter),
        ("SRAM", sram_puf, stressed_sram), 
        ("RingOsc", ro_puf, stressed_ro),
        ("Butterfly", butterfly_puf, stressed_butterfly)
    ]:
        orig_responses = original.eval(test_challenges)
        stress_responses = stressed.eval(test_challenges)
        bit_errors = np.sum(orig_responses != stress_responses)
        ber = 100.0 * bit_errors / len(orig_responses)
        print(f"{puf_name} PUF BER at {T_high}°C: {ber:.1f}% ({bit_errors}/{len(orig_responses)} bits)")
    
    # Test new advanced stressors
    print("\n--- Advanced Stressor Testing ---")
    
    # Test voltage stress
    print("\n-- Voltage Stress Test --")
    voltage_stressed = apply_voltage(arbiter_puf, V_current=3.0, military_spec=True)
    orig_responses = arbiter_puf.eval(test_challenges)
    volt_responses = voltage_stressed.eval(test_challenges)
    volt_ber = 100.0 * np.sum(orig_responses != volt_responses) / len(orig_responses)
    print(f"Voltage stress (3.0V) BER: {volt_ber:.1f}%")
    
    # Test aging stress
    print("\n-- Aging Stress Test --")
    aged_puf = apply_aging(arbiter_puf, age_hours=8760, temperature_history=85.0, military_spec=True)
    aged_responses = aged_puf.eval(test_challenges)
    aging_ber = 100.0 * np.sum(orig_responses != aged_responses) / len(orig_responses)
    print(f"Aging stress (1 year at 85°C) BER: {aging_ber:.1f}%")
    
    # Test radiation stress
    print("\n-- Radiation Stress Test --")
    rad_puf = apply_radiation(arbiter_puf, dose_krad=100, particle_type='gamma', military_spec=True)
    rad_responses = rad_puf.eval(test_challenges)
    rad_ber = 100.0 * np.sum(orig_responses != rad_responses) / len(orig_responses)
    print(f"Radiation stress (100 krad gamma) BER: {rad_ber:.1f}%")
    
    # Test EMI stress
    print("\n-- EMI Stress Test --")
    emi_puf = apply_emi(arbiter_puf, frequency_mhz=100, field_strength_v_m=200, military_spec=True)
    emi_responses = emi_puf.eval(test_challenges)
    emi_ber = 100.0 * np.sum(orig_responses != emi_responses) / len(orig_responses)
    print(f"EMI stress (200 V/m at 100 MHz) BER: {emi_ber:.1f}%")
    
    # Test multi-stress scenario
    print("\n-- Multi-Stress Test --")
    multi_stressed = apply_multi_stress(arbiter_puf, temperature=85.0, voltage=3.0, 
                                      age_hours=4380, radiation_krad=50, emi_field=100,
                                      military_spec=True)
    multi_responses = multi_stressed.eval(test_challenges)
    multi_ber = 100.0 * np.sum(orig_responses != multi_responses) / len(orig_responses)
    print(f"Multi-stress scenario BER: {multi_ber:.1f}%")
    
    print("\n=== All advanced stressor tests passed! ===")
