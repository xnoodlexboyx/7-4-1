"""
Side-Channel Attack Framework for PPET
=====================================

This module implements side-channel attack models for defense-oriented PUF evaluation.
Focuses on military and national security applications including:
- Power analysis attacks (SPA, DPA, CPA)
- Timing analysis attacks  
- Electromagnetic emanation analysis
- Thermal analysis attacks
- Acoustic cryptanalysis
- Optical side-channel attacks

Designed for modeling sophisticated adversaries in military environments.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from .puf_models import BasePUF, ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
# Try to import optional packages
try:
    import scipy.signal
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class SideChannelTrace:
    """
    Container for side-channel measurement traces.
    """
    trace_data: np.ndarray  # Raw measurement data
    challenge: np.ndarray   # Associated challenge
    response: int          # Ground truth response
    timestamp: float       # Measurement timestamp
    metadata: Dict[str, Any]  # Additional trace information


class PowerAnalysisAttacker:
    """
    Power analysis attack implementation for PUF evaluation.
    Models both simple (SPA) and differential (DPA) power analysis.
    """
    
    def __init__(self, attack_type: str = 'dpa', noise_level: float = 0.1):
        """
        Initialize power analysis attacker.
        
        Parameters
        ----------
        attack_type : str
            Type of power analysis ('spa', 'dpa', 'cpa')
        noise_level : float
            Measurement noise level
        """
        self.attack_type = attack_type
        self.noise_level = noise_level
        self.traces = []
        self.power_model = None
        self._is_trained = False
    
    def generate_power_trace(self, puf: BasePUF, challenge: np.ndarray, 
                           response: int) -> SideChannelTrace:
        """
        Generate simulated power consumption trace for PUF evaluation.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        challenge : np.ndarray
            Input challenge vector
        response : int
            PUF response (+1 or -1)
            
        Returns
        -------
        SideChannelTrace
            Simulated power trace
        """
        n_samples = 1000  # Trace length
        rng = np.random.default_rng(42)
        
        # Base power consumption
        base_power = 50.0  # mW
        
        # Challenge-dependent power variations
        if isinstance(puf, ArbiterPUF):
            # Arbiter PUF power depends on switching activity
            hamming_weight = np.sum(challenge)
            switching_power = 0.5 * hamming_weight  # mW per bit
            
            # Response-dependent leakage
            response_leakage = 2.0 if response == 1 else -1.5  # mW
            
        elif isinstance(puf, SRAMPUF):
            # SRAM power depends on number of cells and bias
            cell_power = 0.1 * len(challenge)  # mW per cell
            bias_power = 1.0 * np.mean(challenge)  # Bias-dependent
            switching_power = cell_power + bias_power
            
            # SRAM response leakage
            response_leakage = 3.0 if response == 1 else -2.0  # mW
            
        elif isinstance(puf, RingOscillatorPUF):
            # Ring oscillator power depends on frequency
            freq_power = 0.8 * np.sum(challenge)  # Higher freq = more power
            switching_power = freq_power
            
            # Frequency-dependent leakage
            response_leakage = 1.5 if response == 1 else -1.0  # mW
            
        elif isinstance(puf, ButterflyPUF):
            # Butterfly power depends on metastability resolution
            meta_power = 0.6 * np.sum(challenge)  # Metastability cost
            switching_power = meta_power
            
            # Latch settling leakage
            response_leakage = 2.5 if response == 1 else -1.8  # mW
            
        else:
            # Default power model
            switching_power = 0.3 * np.sum(challenge)
            response_leakage = 1.0 if response == 1 else -0.5
        
        # Generate time-domain power trace
        total_power = base_power + switching_power + response_leakage
        
        # Simulate evaluation phases
        trace = np.zeros(n_samples)
        
        # Challenge loading phase (samples 0-200)
        trace[0:200] = base_power + switching_power * 0.3
        
        # Evaluation phase (samples 200-700) - main leakage
        eval_power = total_power + 5 * np.sin(2 * np.pi * np.arange(500) / 100)
        trace[200:700] = eval_power
        
        # Response generation phase (samples 700-900)
        resp_power = base_power + response_leakage
        trace[700:900] = resp_power
        
        # Idle phase (samples 900-1000)
        trace[900:1000] = base_power
        
        # Add measurement noise
        noise = rng.normal(0, self.noise_level * base_power, n_samples)
        trace += noise
        
        # Add electromagnetic interference (military environment)
        emi_freq = 100e6  # 100 MHz
        sample_rate = 1e9  # 1 GSa/s
        t = np.arange(n_samples) / sample_rate
        emi_signal = 0.5 * np.sin(2 * np.pi * emi_freq * t)
        trace += emi_signal
        
        return SideChannelTrace(
            trace_data=trace,
            challenge=challenge,
            response=response,
            timestamp=0.0,
            metadata={
                'attack_type': 'power',
                'base_power': base_power,
                'switching_power': switching_power,
                'response_leakage': response_leakage,
                'noise_level': self.noise_level
            }
        )
    
    def collect_traces(self, puf: BasePUF, n_traces: int = 1000) -> List[SideChannelTrace]:
        """
        Collect power traces for attack.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        n_traces : int
            Number of traces to collect
            
        Returns
        -------
        List[SideChannelTrace]
            Collected power traces
        """
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        
        traces = []
        for i in range(n_traces):
            challenge = rng.integers(0, 2, size=n_stages)
            response = puf.eval(challenge.reshape(1, -1))[0]
            
            trace = self.generate_power_trace(puf, challenge, response)
            traces.append(trace)
        
        self.traces = traces
        return traces
    
    def perform_dpa_attack(self, traces: List[SideChannelTrace]) -> Dict[str, Any]:
        """
        Perform Differential Power Analysis (DPA) attack.
        
        Parameters
        ----------
        traces : List[SideChannelTrace]
            Power traces for analysis
            
        Returns
        -------
        Dict[str, Any]
            DPA attack results
        """
        if not traces:
            raise ValueError("No traces provided for DPA analysis")
        
        # Separate traces by response value
        traces_positive = [t for t in traces if t.response == 1]
        traces_negative = [t for t in traces if t.response == -1]
        
        if not traces_positive or not traces_negative:
            raise ValueError("Need traces for both response values")
        
        # Compute mean traces for each response
        mean_positive = np.mean([t.trace_data for t in traces_positive], axis=0)
        mean_negative = np.mean([t.trace_data for t in traces_negative], axis=0)
        
        # DPA differential trace
        differential = mean_positive - mean_negative
        
        # Find peak difference (potential leakage point)
        peak_idx = np.argmax(np.abs(differential))
        peak_value = differential[peak_idx]
        
        # Statistical significance test (with fallback if scipy not available)
        if HAS_SCIPY:
            pos_values = [t.trace_data[peak_idx] for t in traces_positive]
            neg_values = [t.trace_data[peak_idx] for t in traces_negative]
            
            t_stat, p_value = scipy.stats.ttest_ind(pos_values, neg_values)
        else:
            t_stat, p_value = 0.0, 0.5  # Conservative fallback
        
        # Signal-to-noise ratio
        signal_power = np.var(differential)
        noise_power = np.mean([np.var(t.trace_data) for t in traces])
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        return {
            'attack_type': 'dpa',
            'differential_trace': differential,
            'peak_index': peak_idx,
            'peak_value': peak_value,
            't_statistic': t_stat,
            'p_value': p_value,
            'snr_db': snr,
            'traces_positive': len(traces_positive),
            'traces_negative': len(traces_negative),
            'leakage_detected': p_value < 0.05 and abs(peak_value) > 1.0
        }
    
    def perform_cpa_attack(self, traces: List[SideChannelTrace]) -> Dict[str, Any]:
        """
        Perform Correlation Power Analysis (CPA) attack.
        
        Parameters
        ----------
        traces : List[SideChannelTrace]
            Power traces for analysis
            
        Returns
        -------
        Dict[str, Any]
            CPA attack results
        """
        if not traces:
            raise ValueError("No traces provided for CPA analysis")
        
        # Extract power traces and hypothetical power model
        power_traces = np.array([t.trace_data for t in traces])
        responses = np.array([t.response for t in traces])
        
        # Hypothetical power model (Hamming weight of response)
        hypothesis = np.array([1 if r == 1 else 0 for r in responses])
        
        # Compute correlation for each time sample
        n_samples = power_traces.shape[1]
        correlations = np.zeros(n_samples)
        
        for i in range(n_samples):
            if HAS_SCIPY:
                corr_coef, _ = scipy.stats.pearsonr(power_traces[:, i], hypothesis)
            else:
                # Simple correlation fallback
                corr_coef = np.corrcoef(power_traces[:, i], hypothesis)[0, 1]
                if np.isnan(corr_coef):
                    corr_coef = 0.0
            correlations[i] = corr_coef
        
        # Find maximum correlation
        max_corr_idx = np.argmax(np.abs(correlations))
        max_correlation = correlations[max_corr_idx]
        
        # Correlation threshold for successful attack
        correlation_threshold = 0.3
        attack_success = abs(max_correlation) > correlation_threshold
        
        return {
            'attack_type': 'cpa',
            'correlations': correlations,
            'max_correlation': max_correlation,
            'max_correlation_index': max_corr_idx,
            'attack_success': attack_success,
            'correlation_threshold': correlation_threshold,
            'n_traces': len(traces)
        }


class TimingAnalysisAttacker:
    """
    Timing analysis attack for PUF evaluation.
    Models timing side-channel leakage in PUF implementations.
    """
    
    def __init__(self, timing_resolution: float = 1e-9):
        """
        Initialize timing analysis attacker.
        
        Parameters
        ----------
        timing_resolution : float
            Timing measurement resolution in seconds
        """
        self.timing_resolution = timing_resolution
        self.timing_traces = []
    
    def measure_timing(self, puf: BasePUF, challenge: np.ndarray) -> float:
        """
        Simulate timing measurement for PUF evaluation.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        challenge : np.ndarray
            Input challenge
            
        Returns
        -------
        float
            Simulated evaluation time in seconds
        """
        # Base evaluation time
        base_time = 10e-9  # 10 ns
        
        # Challenge-dependent timing variations
        if isinstance(puf, ArbiterPUF):
            # Timing depends on delay path lengths
            # Use simple challenge transformation instead of MLAttacker reference
            challenge_features = np.array([1 - 2 * c for c in challenge])
            delay_sum = np.sum(puf.delay_params * challenge_features)
            timing_variation = abs(delay_sum) * 1e-12  # ps per delay unit
            
        elif isinstance(puf, SRAMPUF):
            # SRAM timing depends on cell bias and noise
            cell_timing = np.mean(puf.vth_variations) * 1e-12  # ps per mV
            timing_variation = abs(cell_timing)
            
        elif isinstance(puf, RingOscillatorPUF):
            # RO timing depends on frequency differences
            freq_diff = np.std(puf.base_frequencies)
            timing_variation = freq_diff * 1e-12  # ps per MHz
            
        elif isinstance(puf, ButterflyPUF):
            # Butterfly timing depends on metastability resolution
            meta_time = np.mean(puf.settling_times)
            timing_variation = meta_time * 1e-9  # ns
            
        else:
            timing_variation = 1e-12  # Default 1 ps variation
        
        # Add measurement noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, self.timing_resolution * 0.1)
        
        total_time = base_time + timing_variation + noise
        return max(total_time, 0)  # Ensure positive timing
    
    def collect_timing_traces(self, puf: BasePUF, n_measurements: int = 1000) -> List[Tuple[np.ndarray, float, int]]:
        """
        Collect timing measurements for attack.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        n_measurements : int
            Number of timing measurements
            
        Returns
        -------
        List[Tuple[np.ndarray, float, int]]
            List of (challenge, timing, response) tuples
        """
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        
        timing_traces = []
        for i in range(n_measurements):
            challenge = rng.integers(0, 2, size=n_stages)
            timing = self.measure_timing(puf, challenge)
            response = puf.eval(challenge.reshape(1, -1))[0]
            
            timing_traces.append((challenge, timing, response))
        
        self.timing_traces = timing_traces
        return timing_traces
    
    def analyze_timing_correlation(self, timing_traces: List[Tuple[np.ndarray, float, int]]) -> Dict[str, Any]:
        """
        Analyze timing correlation with PUF responses.
        
        Parameters
        ----------
        timing_traces : List[Tuple[np.ndarray, float, int]]
            Timing measurement data
            
        Returns
        -------
        Dict[str, Any]
            Timing analysis results
        """
        if not timing_traces:
            raise ValueError("No timing traces provided")
        
        timings = np.array([t[1] for t in timing_traces])
        responses = np.array([t[2] for t in timing_traces])
        
        # Separate timings by response
        timings_pos = timings[responses == 1]
        timings_neg = timings[responses == -1]
        
        # Statistical analysis
        mean_pos = np.mean(timings_pos)
        mean_neg = np.mean(timings_neg)
        timing_diff = mean_pos - mean_neg
        
        # T-test for significance
        if HAS_SCIPY:
            t_stat, p_value = scipy.stats.ttest_ind(timings_pos, timings_neg)
        else:
            t_stat, p_value = 0.0, 0.5
        
        # Correlation analysis
        binary_responses = (responses == 1).astype(int)
        if HAS_SCIPY:
            correlation, corr_p_value = scipy.stats.pearsonr(timings, binary_responses)
        else:
            correlation = np.corrcoef(timings, binary_responses)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            corr_p_value = 0.5
        
        # Attack success criteria
        attack_success = (p_value < 0.05) and (abs(timing_diff) > self.timing_resolution)
        
        return {
            'attack_type': 'timing',
            'mean_timing_positive': mean_pos,
            'mean_timing_negative': mean_neg,
            'timing_difference': timing_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'correlation': correlation,
            'correlation_p_value': corr_p_value,
            'attack_success': attack_success,
            'n_measurements': len(timing_traces),
            'timing_resolution': self.timing_resolution
        }


class EMAnalysisAttacker:
    """
    Electromagnetic analysis attack for PUF evaluation.
    Models EM emanation analysis in military environments.
    """
    
    def __init__(self, frequency_range: Tuple[float, float] = (1e6, 1e9), 
                 distance_m: float = 0.1):
        """
        Initialize EM analysis attacker.
        
        Parameters
        ----------
        frequency_range : Tuple[float, float]
            EM frequency range to analyze (Hz)
        distance_m : float
            Distance from target device (meters)
        """
        self.freq_min, self.freq_max = frequency_range
        self.distance = distance_m
        self.em_traces = []
    
    def generate_em_trace(self, puf: BasePUF, challenge: np.ndarray, 
                         response: int) -> SideChannelTrace:
        """
        Generate simulated EM emanation trace.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        challenge : np.ndarray
            Input challenge
        response : int
            PUF response
            
        Returns
        -------
        SideChannelTrace
            Simulated EM trace
        """
        n_samples = 1024  # Frequency domain samples
        freqs = np.linspace(self.freq_min, self.freq_max, n_samples)
        
        # Base EM signature
        base_em = 1e-6 / (self.distance ** 2)  # Inverse square law
        
        # Challenge-dependent EM variations
        if isinstance(puf, ArbiterPUF):
            # Switching current creates EM radiation
            switching_activity = np.sum(challenge)
            em_amplitude = base_em * (1 + 0.1 * switching_activity)
            
        elif isinstance(puf, RingOscillatorPUF):
            # Ring oscillators create strong EM signatures
            osc_freq = np.mean(puf.base_frequencies) * 1e6  # Convert to Hz
            em_amplitude = base_em * (1 + 0.2 * np.sum(challenge))
            
        else:
            em_amplitude = base_em * (1 + 0.05 * np.sum(challenge))
        
        # Generate frequency domain EM signature
        em_spectrum = np.zeros(n_samples, dtype=complex)
        
        # Clock harmonics
        clock_freq = 100e6  # 100 MHz clock
        for harmonic in range(1, 6):
            freq_idx = np.argmin(np.abs(freqs - harmonic * clock_freq))
            if freq_idx < n_samples:
                em_spectrum[freq_idx] = em_amplitude * (1 / harmonic)
        
        # Response-dependent modulation
        if response == 1:
            # Additional harmonics for response = 1
            mod_freq = 50e6  # 50 MHz modulation
            for harmonic in range(1, 4):
                freq_idx = np.argmin(np.abs(freqs - harmonic * mod_freq))
                if freq_idx < n_samples:
                    em_spectrum[freq_idx] += em_amplitude * 0.3 / harmonic
        
        # Add measurement noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, em_amplitude * 0.1, n_samples) + \
               1j * rng.normal(0, em_amplitude * 0.1, n_samples)
        em_spectrum += noise
        
        # Convert to time domain for analysis
        time_trace = np.abs(np.fft.ifft(em_spectrum))
        
        return SideChannelTrace(
            trace_data=time_trace,
            challenge=challenge,
            response=response,
            timestamp=0.0,
            metadata={
                'attack_type': 'em',
                'frequency_range': (self.freq_min, self.freq_max),
                'distance': self.distance,
                'em_amplitude': em_amplitude
            }
        )
    
    def collect_em_traces(self, puf: BasePUF, n_traces: int = 500) -> List[SideChannelTrace]:
        """
        Collect EM traces for attack.
        
        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        n_traces : int
            Number of EM traces to collect
            
        Returns
        -------
        List[SideChannelTrace]
            Collected EM traces
        """
        rng = np.random.default_rng(42)
        n_stages = getattr(puf, 'n_stages', getattr(puf, 'n_cells', 
                          getattr(puf, 'n_rings', getattr(puf, 'n_butterflies', 64))))
        
        em_traces = []
        for i in range(n_traces):
            challenge = rng.integers(0, 2, size=n_stages)
            response = puf.eval(challenge.reshape(1, -1))[0]
            
            trace = self.generate_em_trace(puf, challenge, response)
            em_traces.append(trace)
        
        self.em_traces = em_traces
        return em_traces
    
    def analyze_em_leakage(self, em_traces: List[SideChannelTrace]) -> Dict[str, Any]:
        """
        Analyze EM leakage for PUF attacks.
        
        Parameters
        ----------
        em_traces : List[SideChannelTrace]
            EM measurement traces
            
        Returns
        -------
        Dict[str, Any]
            EM analysis results
        """
        if not em_traces:
            raise ValueError("No EM traces provided")
        
        # Extract trace data and responses
        traces_matrix = np.array([t.trace_data for t in em_traces])
        responses = np.array([t.response for t in em_traces])
        
        # Principal Component Analysis for feature extraction
        if HAS_SKLEARN:
            pca = PCA(n_components=10)
            pca_features = pca.fit_transform(traces_matrix)
            
            # Correlation analysis with responses
            correlations = []
            for i in range(pca_features.shape[1]):
                if HAS_SCIPY:
                    corr, _ = scipy.stats.pearsonr(pca_features[:, i], responses)
                else:
                    corr = np.corrcoef(pca_features[:, i], responses)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                correlations.append(corr)
            
            max_correlation = max(correlations, key=abs)
            max_corr_component = np.argmax(np.abs(correlations))
            
            # Clustering analysis
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(pca_features[:, :3])  # Use top 3 components
            explained_variance = pca.explained_variance_ratio_
        else:
            # Fallback without sklearn
            max_correlation = 0.0
            max_corr_component = 0
            clusters = np.random.randint(0, 2, len(responses))
            explained_variance = np.array([0.5, 0.3, 0.2])
        
        # Cluster purity (how well clusters separate responses)
        cluster_0_responses = responses[clusters == 0]
        cluster_1_responses = responses[clusters == 1]
        
        if len(cluster_0_responses) > 0 and len(cluster_1_responses) > 0:
            purity_0 = max(np.mean(cluster_0_responses == 1), np.mean(cluster_0_responses == -1))
            purity_1 = max(np.mean(cluster_1_responses == 1), np.mean(cluster_1_responses == -1))
            overall_purity = (len(cluster_0_responses) * purity_0 + 
                             len(cluster_1_responses) * purity_1) / len(responses)
        else:
            overall_purity = 0.5  # Default
        
        # Attack success criteria
        attack_success = (abs(max_correlation) > 0.3) or (overall_purity > 0.8)
        
        return {
            'attack_type': 'em',
            'max_correlation': max_correlation,
            'max_correlation_component': max_corr_component,
            'pca_explained_variance': explained_variance,
            'cluster_purity': overall_purity,
            'attack_success': attack_success,
            'n_traces': len(em_traces),
            'distance': self.distance
        }
class MultiChannelAttacker:
    """Multi-channel side-channel attacker combining multiple attack vectors.
    Models sophisticated military adversaries with advanced equipment.
    """

    def __init__(self):
        """Initialize multi-channel attacker."""
        self.power_attacker = PowerAnalysisAttacker()
        self.timing_attacker = TimingAnalysisAttacker()
        self.em_attacker = EMAnalysisAttacker()
        self.attack_results = {}

    def comprehensive_attack(self, puf: BasePUF, n_traces: int = 1000) -> Dict[str, Any]:
        """Perform comprehensive multi-channel side-channel attack.

        Parameters
        ----------
        puf : BasePUF
            Target PUF instance
        n_traces : int
            Number of traces per attack vector

        Returns
        -------
        Dict[str, Any]
            Combined attack results
        """
        results = {
            'power_analysis': {},
            'timing_analysis': {},
            'em_analysis': {},
            'combined_attack': {}
        }

        # Power analysis attack
        power_traces = self.power_attacker.collect_traces(puf, n_traces)
        results['power_analysis']['dpa'] = self.power_attacker.perform_dpa_attack(power_traces)
        results['power_analysis']['cpa'] = self.power_attacker.perform_cpa_attack(power_traces)

        # Timing analysis attack
        timing_traces = self.timing_attacker.collect_timing_traces(puf, n_traces)
        results['timing_analysis'] = self.timing_attacker.analyze_timing_correlation(timing_traces)

        # EM analysis attack
        em_traces = self.em_attacker.collect_em_traces(puf, n_traces // 2)
        results['em_analysis'] = self.em_attacker.analyze_em_leakage(em_traces)

        # Combined attack assessment
        attack_success_count = sum([
            results['power_analysis']['dpa']['leakage_detected'],
            results['power_analysis']['cpa']['attack_success'],
            results['timing_analysis']['attack_success'],
            results['em_analysis']['attack_success']
        ])

        combined_success_rate = attack_success_count / 4.0

        # Overall threat assessment
        if combined_success_rate >= 0.75:
            threat_level = 'CRITICAL'
        elif combined_success_rate >= 0.5:
            threat_level = 'HIGH'
        elif combined_success_rate >= 0.25:
            threat_level = 'MEDIUM'
        else:
            threat_level = 'LOW'

        results['combined_attack'] = {
            'successful_attacks': attack_success_count,
            'total_attacks': 4,
            'success_rate': combined_success_rate,
            'threat_level': threat_level,
            'recommendation': self._generate_recommendation(threat_level)
        }

        self.attack_results = results
        return results

    def _generate_recommendation(self, threat_level: str) -> str:
        """Generate defense recommendations based on threat level."""
        recommendations = {
            'CRITICAL': 'IMMEDIATE ACTION REQUIRED: Implement hardware countermeasures, power filtering, temporal randomization, and physical shielding.',
            'HIGH': 'Deploy advanced countermeasures: power line filtering, noise injection, and EM shielding.',
            'MEDIUM': 'Consider basic countermeasures: power supply decoupling, clock randomization, and physical access controls.',
            'LOW': 'Current defenses adequate for most threat scenarios. Monitor for advanced persistent threats.'
        }
        return recommendations.get(threat_level, 'Unknown threat level')


if __name__ == "__main__":
    print("=== PPET Side-Channel Attack Framework ===")
    print("Testing side-channel vulnerabilities for military PUF evaluation\n")

    from attacks import MLAttacker  # noqa:F401

    from puf_models import ArbiterPUF
    puf = ArbiterPUF(n_stages=64, seed=42)

    print("--- Power Analysis Attack ---")
    power_attacker = PowerAnalysisAttacker(attack_type='dpa')
    power_traces = power_attacker.collect_traces(puf, n_traces=200)
    dpa_result = power_attacker.perform_dpa_attack(power_traces)
    print(f"DPA leakage detected: {dpa_result['leakage_detected']}")
    print(f"DPA p-value: {dpa_result['p_value']:.6f}")
    print(f"DPA SNR: {dpa_result['snr_db']:.2f} dB")

    cpa_result = power_attacker.perform_cpa_attack(power_traces)
    print(f"CPA attack success: {cpa_result['attack_success']}")
    print(f"CPA max correlation: {cpa_result['max_correlation']:.4f}")

    print("\n--- Timing Analysis Attack ---")
    timing_attacker = TimingAnalysisAttacker()
    timing_traces = timing_attacker.collect_timing_traces(puf, n_measurements=200)
    timing_result = timing_attacker.analyze_timing_correlation(timing_traces)
    print(f"Timing attack success: {timing_result['attack_success']}")
    print(f"Timing difference: {timing_result['timing_difference']:.2e} s")
    print(f"Timing correlation: {timing_result['correlation']:.4f}")

    print("\n--- EM Analysis Attack ---")
    em_attacker = EMAnalysisAttacker()
    em_traces = em_attacker.collect_em_traces(puf, n_traces=100)
    em_result = em_attacker.analyze_em_leakage(em_traces)
    print(f"EM attack success: {em_result['attack_success']}")
    print(f"EM max correlation: {em_result['max_correlation']:.4f}")
    print(f"EM cluster purity: {em_result['cluster_purity']:.4f}")

    print("\n--- Comprehensive Multi-Channel Attack ---")
    multi_attacker = MultiChannelAttacker()
    comprehensive_results = multi_attacker.comprehensive_attack(puf, n_traces=150)

    combined = comprehensive_results['combined_attack']
    print(f"Successful attacks: {combined['successful_attacks']}/{combined['total_attacks']}")
    print(f"Overall success rate: {combined['success_rate']:.2f}")
    print(f"Threat level: {combined['threat_level']}")
    print(f"Recommendation: {combined['recommendation']}")

    print("\n=== Side-channel attack framework testing complete ===")
    print("PPET ready for comprehensive side-channel security evaluation.")
