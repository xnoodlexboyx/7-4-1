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

class SRAMPUF(BasePUF):
    """
    Implements SRAM PUF based on startup values of uninitialized SRAM cells.
    Models process variations in SRAM cell threshold voltages.
    
    Defense-specific parameters:
    - Radiation hardening coefficient
    - Low-power operation mode
    - Temperature coefficient for military applications
    
    References
    ----------
    Holcomb, D. E., et al. (2009). Power-up SRAM state as an identifying fingerprint and source of true random numbers. IEEE Trans. Computers.
    """
    
    def __init__(self, n_cells: int, seed: Optional[int] = None, t_nominal: float = 25.0,
                 radiation_hardening: float = 0.95, low_power_mode: bool = False) -> None:
        """
        Initialize SRAM PUF instance.
        
        Parameters
        ----------
        n_cells : int
            Number of SRAM cells (equivalent to challenge length).
        seed : int or None, optional
            Random seed for reproducibility.
        t_nominal : float, optional
            Nominal temperature in Celsius.
        radiation_hardening : float, optional
            Radiation hardening coefficient (0.0-1.0), affects cell stability.
        low_power_mode : bool, optional
            Low power operation mode for battery-powered military devices.
        """
        self.n_cells = n_cells
        self.t_nominal = t_nominal
        self.radiation_hardening = radiation_hardening
        self.low_power_mode = low_power_mode
        
        rng = np.random.default_rng(seed)
        # Model threshold voltage variations (mV) - defense-grade cells have tighter control
        self.vth_variations = rng.normal(0, 50 * (2 - radiation_hardening), n_cells)
        # Power supply noise sensitivity
        self.noise_sensitivity = rng.uniform(0.8, 1.2, n_cells)
        
    def eval(self, challenges: np.ndarray) -> np.ndarray:
        """
        Evaluate SRAM PUF. Challenge bits select which cells to read.
        
        Parameters
        ----------
        challenges : np.ndarray
            Array of challenges, shape (n_challenges, n_cells), values in {0, 1}.
            
        Returns
        -------
        np.ndarray
            Array of responses (+1 or -1), shape (n_challenges,).
        """
        challenges = np.asarray(challenges)
        if challenges.ndim == 1:
            challenges = challenges.reshape(1, -1)
        n_challenges, n_cells = challenges.shape
        assert n_cells == self.n_cells, "Challenge length mismatch."
        
        responses = np.empty(n_challenges, dtype=int)
        for i, ch in enumerate(challenges):
            # SRAM cell startup behavior based on threshold voltage variations
            # Apply challenge-dependent addressing
            cell_values = np.zeros(self.n_cells)
            for j in range(self.n_cells):
                # Model cell startup value based on threshold voltage and addressing
                vth_effective = self.vth_variations[j] * (1 + 0.1 * ch[j])
                if self.low_power_mode:
                    vth_effective *= 1.1  # Reduced drive strength in low power mode
                cell_values[j] = vth_effective
            
            # XOR reduction for response generation (common in SRAM PUFs)
            response_bits = (cell_values > 0).astype(int)
            response = np.sum(response_bits * ch) % 2
            responses[i] = 1 if response == 1 else -1
            
        return responses
    
    def to_json(self) -> str:
        """Serialize SRAM PUF parameters to JSON."""
        return json.dumps({
            "n_cells": self.n_cells,
            "t_nominal": self.t_nominal,
            "radiation_hardening": self.radiation_hardening,
            "low_power_mode": self.low_power_mode,
            "vth_variations": self.vth_variations.tolist(),
            "noise_sensitivity": self.noise_sensitivity.tolist(),
        })
    
    @classmethod
    def from_json(cls, data: str) -> "SRAMPUF":
        """Deserialize SRAM PUF from JSON."""
        obj = json.loads(data)
        inst = cls(obj["n_cells"], t_nominal=obj["t_nominal"],
                  radiation_hardening=obj["radiation_hardening"],
                  low_power_mode=obj["low_power_mode"])
        inst.vth_variations = np.array(obj["vth_variations"])
        inst.noise_sensitivity = np.array(obj["noise_sensitivity"])
        return inst


class RingOscillatorPUF(BasePUF):
    """
    Implements Ring Oscillator PUF based on frequency variations of ring oscillators.
    Models process variations in transistor characteristics affecting oscillation frequency.
    
    Defense-specific parameters:
    - EMI resistance for battlefield environments
    - Frequency stability under temperature variations
    - Power analysis resistance
    
    References
    ----------
    Suh, G. E., & Devadas, S. (2007). Physical unclonable functions for device authentication and secret key generation. In DAC '07.
    """
    
    def __init__(self, n_rings: int, seed: Optional[int] = None, t_nominal: float = 25.0,
                 emi_resistance: float = 0.9, freq_stability: float = 0.8) -> None:
        """
        Initialize Ring Oscillator PUF instance.
        
        Parameters
        ----------
        n_rings : int
            Number of ring oscillators.
        seed : int or None, optional
            Random seed for reproducibility.
        t_nominal : float, optional
            Nominal temperature in Celsius.
        emi_resistance : float, optional
            EMI resistance coefficient (0.0-1.0) for battlefield environments.
        freq_stability : float, optional
            Frequency stability coefficient (0.0-1.0) under temperature variations.
        """
        self.n_rings = n_rings
        self.t_nominal = t_nominal
        self.emi_resistance = emi_resistance
        self.freq_stability = freq_stability
        
        rng = np.random.default_rng(seed)
        # Base frequencies (MHz) with process variations
        self.base_frequencies = rng.normal(100, 5, n_rings)  # 100MHz ± 5MHz
        # Process variation factors
        self.process_variations = rng.normal(1.0, 0.02, n_rings)
        # EMI susceptibility factors
        self.emi_susceptibility = rng.uniform(0.95, 1.05, n_rings) * (2 - emi_resistance)
        
    def eval(self, challenges: np.ndarray) -> np.ndarray:
        """
        Evaluate Ring Oscillator PUF. Challenge selects pairs of oscillators to compare.
        
        Parameters
        ----------
        challenges : np.ndarray
            Array of challenges, shape (n_challenges, n_rings), values in {0, 1}.
            
        Returns
        -------
        np.ndarray
            Array of responses (+1 or -1), shape (n_challenges,).
        """
        challenges = np.asarray(challenges)
        if challenges.ndim == 1:
            challenges = challenges.reshape(1, -1)
        n_challenges, n_rings = challenges.shape
        assert n_rings == self.n_rings, "Challenge length mismatch."
        
        responses = np.empty(n_challenges, dtype=int)
        for i, ch in enumerate(challenges):
            # Current operating frequencies with variations
            current_freqs = self.base_frequencies * self.process_variations * self.emi_susceptibility
            
            # Challenge-based ring selection and comparison
            selected_rings = np.where(ch == 1)[0]
            if len(selected_rings) < 2:
                # Default comparison if insufficient rings selected
                freq_diff = current_freqs[0] - current_freqs[1]
            else:
                # Compare first two selected rings
                freq_diff = current_freqs[selected_rings[0]] - current_freqs[selected_rings[1]]
            
            # Apply frequency stability factor
            freq_diff *= self.freq_stability
            
            responses[i] = 1 if freq_diff > 0 else -1
            
        return responses
    
    def to_json(self) -> str:
        """Serialize Ring Oscillator PUF parameters to JSON."""
        return json.dumps({
            "n_rings": self.n_rings,
            "t_nominal": self.t_nominal,
            "emi_resistance": self.emi_resistance,
            "freq_stability": self.freq_stability,
            "base_frequencies": self.base_frequencies.tolist(),
            "process_variations": self.process_variations.tolist(),
            "emi_susceptibility": self.emi_susceptibility.tolist(),
        })
    
    @classmethod
    def from_json(cls, data: str) -> "RingOscillatorPUF":
        """Deserialize Ring Oscillator PUF from JSON."""
        obj = json.loads(data)
        inst = cls(obj["n_rings"], t_nominal=obj["t_nominal"],
                  emi_resistance=obj["emi_resistance"],
                  freq_stability=obj["freq_stability"])
        inst.base_frequencies = np.array(obj["base_frequencies"])
        inst.process_variations = np.array(obj["process_variations"])
        inst.emi_susceptibility = np.array(obj["emi_susceptibility"])
        return inst


class ButterflyPUF(BasePUF):
    """
    Implements Butterfly PUF based on cross-coupled latch settling behavior.
    Models metastability resolution in cross-coupled latches.
    
    Defense-specific parameters:
    - Metastability resolution time for secure operations
    - Cross-talk resistance for dense military electronics
    - Supply voltage sensitivity for power-constrained environments
    
    References
    ----------
    Kumar, S. S., et al. (2008). The butterfly PUF protecting IP on every FPGA. In HOST '08.
    """
    
    def __init__(self, n_butterflies: int, seed: Optional[int] = None, t_nominal: float = 25.0,
                 metastability_time: float = 1.0, crosstalk_resistance: float = 0.85) -> None:
        """
        Initialize Butterfly PUF instance.
        
        Parameters
        ----------
        n_butterflies : int
            Number of butterfly (cross-coupled latch) structures.
        seed : int or None, optional
            Random seed for reproducibility.
        t_nominal : float, optional
            Nominal temperature in Celsius.
        metastability_time : float, optional
            Metastability resolution time coefficient (higher = more stable).
        crosstalk_resistance : float, optional
            Cross-talk resistance coefficient (0.0-1.0) for dense electronics.
        """
        self.n_butterflies = n_butterflies
        self.t_nominal = t_nominal
        self.metastability_time = metastability_time
        self.crosstalk_resistance = crosstalk_resistance
        
        rng = np.random.default_rng(seed)
        # Latch imbalance factors (process variations)
        self.latch_imbalances = rng.normal(0, 0.1, n_butterflies)
        # Settling time variations
        self.settling_times = rng.exponential(metastability_time, n_butterflies)
        # Cross-talk coupling factors
        self.crosstalk_factors = rng.uniform(0.01, 0.05, n_butterflies) * (2 - crosstalk_resistance)
        
    def eval(self, challenges: np.ndarray) -> np.ndarray:
        """
        Evaluate Butterfly PUF. Challenge determines latch initialization pattern.
        
        Parameters
        ----------
        challenges : np.ndarray
            Array of challenges, shape (n_challenges, n_butterflies), values in {0, 1}.
            
        Returns
        -------
        np.ndarray
            Array of responses (+1 or -1), shape (n_challenges,).
        """
        challenges = np.asarray(challenges)
        if challenges.ndim == 1:
            challenges = challenges.reshape(1, -1)
        n_challenges, n_butterflies = challenges.shape
        assert n_butterflies == self.n_butterflies, "Challenge length mismatch."
        
        responses = np.empty(n_challenges, dtype=int)
        for i, ch in enumerate(challenges):
            # Model latch settling behavior
            latch_states = np.zeros(self.n_butterflies)
            
            for j in range(self.n_butterflies):
                # Challenge affects initial latch state
                init_bias = 0.1 * (2 * ch[j] - 1)  # Convert 0/1 to -1/+1 and scale
                
                # Latch imbalance with process variations
                total_imbalance = self.latch_imbalances[j] + init_bias
                
                # Cross-talk from neighboring butterflies
                if j > 0:
                    total_imbalance += self.crosstalk_factors[j] * latch_states[j-1]
                
                # Settling behavior based on imbalance and settling time
                settling_factor = 1.0 / (1.0 + self.settling_times[j])
                final_state = np.tanh(total_imbalance / settling_factor)
                
                latch_states[j] = final_state
            
            # Generate response from latch states
            response_value = np.sum(latch_states * ch) / self.n_butterflies
            responses[i] = 1 if response_value > 0 else -1
            
        return responses
    
    def to_json(self) -> str:
        """Serialize Butterfly PUF parameters to JSON."""
        return json.dumps({
            "n_butterflies": self.n_butterflies,
            "t_nominal": self.t_nominal,
            "metastability_time": self.metastability_time,
            "crosstalk_resistance": self.crosstalk_resistance,
            "latch_imbalances": self.latch_imbalances.tolist(),
            "settling_times": self.settling_times.tolist(),
            "crosstalk_factors": self.crosstalk_factors.tolist(),
        })
    
    @classmethod
    def from_json(cls, data: str) -> "ButterflyPUF":
        """Deserialize Butterfly PUF from JSON."""
        obj = json.loads(data)
        inst = cls(obj["n_butterflies"], t_nominal=obj["t_nominal"],
                  metastability_time=obj["metastability_time"],
                  crosstalk_resistance=obj["crosstalk_resistance"])
        inst.latch_imbalances = np.array(obj["latch_imbalances"])
        inst.settling_times = np.array(obj["settling_times"])
        inst.crosstalk_factors = np.array(obj["crosstalk_factors"])
        return inst


if __name__ == "__main__":
    # Demo: Test all PUF types
    print("=== PUF Models Demo ===")
    n_stages = 64
    n_challenges = 5
    rng = np.random.default_rng(123)
    challenges = rng.integers(0, 2, size=(n_challenges, n_stages))
    
    # Arbiter PUF
    print("\n--- Arbiter PUF ---")
    arbiter_puf = ArbiterPUF(n_stages=n_stages, seed=42)
    arbiter_responses = arbiter_puf.eval(challenges)
    print(f"Responses: {arbiter_responses}")
    
    # SRAM PUF
    print("\n--- SRAM PUF (Defense-Grade) ---")
    sram_puf = SRAMPUF(n_cells=n_stages, seed=42, radiation_hardening=0.95, low_power_mode=True)
    sram_responses = sram_puf.eval(challenges)
    print(f"Responses: {sram_responses}")
    
    # Ring Oscillator PUF
    print("\n--- Ring Oscillator PUF (EMI-Resistant) ---")
    ro_puf = RingOscillatorPUF(n_rings=n_stages, seed=42, emi_resistance=0.9, freq_stability=0.8)
    ro_responses = ro_puf.eval(challenges)
    print(f"Responses: {ro_responses}")
    
    # Butterfly PUF
    print("\n--- Butterfly PUF (High-Density) ---")
    butterfly_puf = ButterflyPUF(n_butterflies=n_stages, seed=42, 
                                metastability_time=1.0, crosstalk_resistance=0.85)
    butterfly_responses = butterfly_puf.eval(challenges)
    print(f"Responses: {butterfly_responses}")
    
    # Test serialization
    print("\n--- Serialization Test ---")
    sram_json = sram_puf.to_json()
    sram_restored = SRAMPUF.from_json(sram_json)
    sram_restored_responses = sram_restored.eval(challenges)
    print(f"Serialization test passed: {np.array_equal(sram_responses, sram_restored_responses)}")
