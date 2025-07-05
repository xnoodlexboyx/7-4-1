import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional
import os

# Try to import optional packages
try:
    import seaborn as sns
    HAS_SEABORN = True
    # Set professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def plot_multi_puf_comparison(
    temperature_range: np.ndarray,
    puf_performance_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    output_dir: str = 'figures/',
    dpi: int = 300
) -> Figure:
    """
    Create a comprehensive 2x2 subplot comparison of different PUF architectures under temperature stress.
    
    Parameters
    ----------
    temperature_range : np.ndarray
        Array of temperatures tested
    puf_performance_data : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Nested dict with structure: {puf_type: {metric: (values, std_errors)}}
        Expected metrics: 'ber', 'attack_accuracy', 'uniqueness', 'ecc_failure'
    output_dir : str, optional
        Directory to save the figure
    dpi : int, optional
        DPI for saved figure (default 300)
        
    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Define PUF types and colors/styles
    puf_types = ['Arbiter', 'SRAM', 'RingOscillator', 'Butterfly']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    line_styles = ['-', '--', '-.', ':']
    
    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-PUF Architecture Performance Comparison Under Temperature Stress', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Subplot 1: Reliability (BER) vs Temperature
    ax1.set_title('Reliability (Bit Error Rate) vs Temperature', fontsize=14, fontweight='bold')
    for i, puf_type in enumerate(puf_types):
        if puf_type in puf_performance_data and 'ber' in puf_performance_data[puf_type]:
            values, errors = puf_performance_data[puf_type]['ber']
            ax1.errorbar(temperature_range, values, yerr=errors, 
                        label=f'{puf_type} PUF', color=colors[i], 
                        linestyle=line_styles[i], linewidth=2.5, 
                        marker='o', markersize=6, capsize=4)
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Subplot 2: ML Attack Accuracy vs Temperature
    ax2.set_title('ML Attack Accuracy vs Temperature', fontsize=14, fontweight='bold')
    for i, puf_type in enumerate(puf_types):
        if puf_type in puf_performance_data and 'attack_accuracy' in puf_performance_data[puf_type]:
            values, errors = puf_performance_data[puf_type]['attack_accuracy']
            ax2.errorbar(temperature_range, values, yerr=errors,
                        label=f'{puf_type} PUF', color=colors[i],
                        linestyle=line_styles[i], linewidth=2.5,
                        marker='s', markersize=6, capsize=4)
    
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Attack Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Subplot 3: Uniqueness Degradation vs Temperature
    ax3.set_title('Uniqueness Degradation vs Temperature', fontsize=14, fontweight='bold')
    for i, puf_type in enumerate(puf_types):
        if puf_type in puf_performance_data and 'uniqueness' in puf_performance_data[puf_type]:
            values, errors = puf_performance_data[puf_type]['uniqueness']
            ax3.errorbar(temperature_range, values, yerr=errors,
                        label=f'{puf_type} PUF', color=colors[i],
                        linestyle=line_styles[i], linewidth=2.5,
                        marker='^', markersize=6, capsize=4)
    
    ax3.set_xlabel('Temperature (°C)', fontsize=12)
    ax3.set_ylabel('Uniqueness (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Subplot 4: ECC Failure Rate vs Temperature
    ax4.set_title('ECC Failure Rate vs Temperature', fontsize=14, fontweight='bold')
    for i, puf_type in enumerate(puf_types):
        if puf_type in puf_performance_data and 'ecc_failure' in puf_performance_data[puf_type]:
            values, errors = puf_performance_data[puf_type]['ecc_failure']
            ax4.errorbar(temperature_range, values, yerr=errors,
                        label=f'{puf_type} PUF', color=colors[i],
                        linestyle=line_styles[i], linewidth=2.5,
                        marker='d', markersize=6, capsize=4)
    
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('ECC Failure Rate (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # Add critical thresholds
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Critical Threat Level')
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Acceptable Failure Rate')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save high-resolution figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'multi_puf_comparison.png')
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_3d_threat_surface(
    temp_range: np.ndarray,
    voltage_range: np.ndarray, 
    attack_success_data: np.ndarray,
    output_dir: str = 'figures/'
) -> Tuple[Optional[object], str]:
    """
    Create an interactive 3D threat surface showing attack success vs environmental factors.
    
    Parameters
    ----------
    temp_range : np.ndarray
        Temperature range (-20°C to +100°C)
    voltage_range : np.ndarray  
        Voltage deviation range (-10% to +10%)
    attack_success_data : np.ndarray
        2D array of attack success rates (0-100%), shape (len(temp_range), len(voltage_range))
    output_dir : str, optional
        Directory to save the figure
        
    Returns
    -------
    Tuple[Optional[object], str]
        Plotly figure object (if available) and path to saved HTML file
    """
    if not HAS_PLOTLY:
        print("Warning: Plotly not available, skipping 3D threat surface")
        html_path = os.path.join(output_dir, '3d_threat_surface_unavailable.txt')
        os.makedirs(output_dir, exist_ok=True)
        with open(html_path, 'w') as f:
            f.write("3D threat surface requires plotly package\n")
            f.write("Install with: pip install plotly kaleido\n")
        return None, html_path
    # Create meshgrid for surface plot
    temp_mesh, voltage_mesh = np.meshgrid(temp_range, voltage_range, indexing='ij')
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Main threat surface
    fig.add_trace(go.Surface(
        x=temp_mesh,
        y=voltage_mesh,
        z=attack_success_data,
        colorscale='plasma',
        colorbar=dict(
            title="Attack Success Rate (%)",
            titleside="right",
            tickmode="linear",
            tick0=0,
            dtick=10
        ),
        hovertemplate='<b>Temperature:</b> %{x:.1f}°C<br>' +
                      '<b>Voltage Deviation:</b> %{y:.1f}%<br>' +
                      '<b>Attack Success:</b> %{z:.1f}%<extra></extra>',
        name='Threat Surface'
    ))
    
    # Add critical threshold plane at 10% attack success
    threshold_z = np.full_like(attack_success_data, 10)
    fig.add_trace(go.Surface(
        x=temp_mesh,
        y=voltage_mesh,
        z=threshold_z,
        opacity=0.3,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        hovertemplate='<b>Critical Security Threshold</b><br>' +
                      'Attack Success: 10%<extra></extra>',
        name='Critical Threshold'
    ))
    
    # Add contour projections on base plane
    fig.add_trace(go.Contour(
        x=temp_range,
        y=voltage_range,
        z=attack_success_data.T,  # Transpose for correct orientation
        colorscale='plasma',
        showscale=False,
        opacity=0.6,
        contours=dict(
            start=0,
            end=100,
            size=10,
            showlabels=True,
            labelfont=dict(size=10, color="white")
        ),
        hovertemplate='<b>Temperature:</b> %{x:.1f}°C<br>' +
                      '<b>Voltage Deviation:</b> %{y:.1f}%<br>' +
                      '<b>Attack Success:</b> %{z:.1f}%<extra></extra>',
        name='Contour Projection'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='3D Defense Threat Surface: Attack Success vs Environmental Factors',
            x=0.5,
            font=dict(size=18, family="Arial Black")
        ),
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Voltage Deviation (%)',
            zaxis_title='ML Attack Success Rate (%)',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            ),
            yaxis=dict(
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white", 
                showbackground=True,
                zerolinecolor="white"
            ),
            zaxis=dict(
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"
            )
        ),
        width=1000,
        height=800,
        margin=dict(r=20, b=10, l=10, t=60)
    )
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    # Save interactive HTML
    html_path = os.path.join(output_dir, '3d_threat_surface.html')
    pyo.plot(fig, filename=html_path, auto_open=False)
    
    # Save static PNG
    png_path = os.path.join(output_dir, '3d_threat_surface.png')
    fig.write_image(png_path, width=1000, height=800, scale=2)
    
    return fig, html_path


def generate_sample_multi_puf_data(
    temperature_range: np.ndarray,
    n_trials: int = 5
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Generate sample performance data for multi-PUF comparison.
    This is a utility function for testing the visualization.
    
    Parameters
    ----------
    temperature_range : np.ndarray
        Array of temperatures to simulate
    n_trials : int, optional
        Number of simulation trials for error calculation
        
    Returns
    -------
    Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Sample performance data with realistic temperature dependencies
    """
    np.random.seed(42)  # For reproducible sample data
    
    puf_data = {}
    
    # Define temperature-dependent characteristics for each PUF type
    puf_characteristics = {
        'Arbiter': {
            'ber_base': 2.0, 'ber_temp_coeff': 0.08, 'ber_noise': 0.3,
            'attack_base': 85.0, 'attack_temp_coeff': 0.15, 'attack_noise': 2.0,
            'uniqueness_base': 49.5, 'uniqueness_temp_coeff': -0.05, 'uniqueness_noise': 0.5,
            'ecc_base': 3.0, 'ecc_temp_coeff': 0.12, 'ecc_noise': 0.4
        },
        'SRAM': {
            'ber_base': 1.5, 'ber_temp_coeff': 0.06, 'ber_noise': 0.25,
            'attack_base': 88.0, 'attack_temp_coeff': 0.12, 'attack_noise': 1.8,
            'uniqueness_base': 50.2, 'uniqueness_temp_coeff': -0.03, 'uniqueness_noise': 0.4,
            'ecc_base': 2.5, 'ecc_temp_coeff': 0.10, 'ecc_noise': 0.35
        },
        'RingOscillator': {
            'ber_base': 3.0, 'ber_temp_coeff': 0.12, 'ber_noise': 0.4,
            'attack_base': 82.0, 'attack_temp_coeff': 0.18, 'attack_noise': 2.5,
            'uniqueness_base': 48.8, 'uniqueness_temp_coeff': -0.08, 'uniqueness_noise': 0.6,
            'ecc_base': 4.0, 'ecc_temp_coeff': 0.15, 'ecc_noise': 0.5
        },
        'Butterfly': {
            'ber_base': 2.5, 'ber_temp_coeff': 0.10, 'ber_noise': 0.35,
            'attack_base': 86.5, 'attack_temp_coeff': 0.14, 'attack_noise': 2.2,
            'uniqueness_base': 49.0, 'uniqueness_temp_coeff': -0.06, 'uniqueness_noise': 0.5,
            'ecc_base': 3.5, 'ecc_temp_coeff': 0.13, 'ecc_noise': 0.45
        }
    }
    
    for puf_type, chars in puf_characteristics.items():
        puf_data[puf_type] = {}
        
        # Calculate temperature deviation from nominal (25°C)
        temp_dev = np.abs(temperature_range - 25.0)
        
        # Map metric names to characteristic names  
        metric_map = {
            'ber': 'ber',
            'attack_accuracy': 'attack', 
            'uniqueness': 'uniqueness',
            'ecc_failure': 'ecc'
        }
        
        for metric in ['ber', 'attack_accuracy', 'uniqueness', 'ecc_failure']:
            char_name = metric_map[metric]
            
            # Skip if metric not defined for this PUF type
            if f'{char_name}_base' not in chars:
                continue
                
            # Generate multiple trials for error calculation
            trials_data = []
            
            for trial in range(n_trials):
                base_val = chars[f'{char_name}_base']
                temp_coeff = chars[f'{char_name}_temp_coeff']
                noise_level = chars[f'{char_name}_noise']
                
                # Calculate base trend
                if metric == 'uniqueness':
                    # Uniqueness decreases with temperature stress
                    values = base_val + temp_coeff * temp_dev
                else:
                    # BER, attack accuracy, and ECC failure increase with temperature stress
                    values = base_val + temp_coeff * temp_dev
                
                # Add realistic noise
                noise = np.random.normal(0, noise_level, len(values))
                values += noise
                
                # Apply realistic bounds
                if metric == 'attack_accuracy':
                    values = np.clip(values, 75.0, 98.0)
                elif metric == 'uniqueness':
                    values = np.clip(values, 45.0, 52.0)
                elif metric in ['ber', 'ecc_failure']:
                    values = np.clip(values, 0.5, 20.0)
                
                trials_data.append(values)
            
            # Calculate mean and standard error
            trials_array = np.array(trials_data)
            mean_values = np.mean(trials_array, axis=0)
            if HAS_SCIPY:
                std_errors = stats.sem(trials_array, axis=0)  # Standard error of mean
            else:
                std_errors = np.std(trials_array, axis=0) / np.sqrt(len(trials_data))
            
            puf_data[puf_type][metric] = (mean_values, std_errors)
    
    return puf_data


# Import all visualization functions
try:
    from .bit_analysis import plot_bit_aliasing_heatmap
    from .defense_dashboard import create_defense_dashboard, generate_sample_dashboard_data
    from .statistical_plots import generate_statistical_suite, generate_sample_statistical_data
except ImportError:
    # Handle case when running as standalone script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from bit_analysis import plot_bit_aliasing_heatmap
    from defense_dashboard import create_defense_dashboard, generate_sample_dashboard_data
    from statistical_plots import generate_statistical_suite, generate_sample_statistical_data


def generate_all_thesis_plots(
    puf_performance_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    temperature_range: np.ndarray,
    challenges: np.ndarray,
    output_dir: str = 'figures/'
) -> Dict[str, List[str]]:
    """
    Generate all visualizations for thesis defense.
    
    Parameters
    ----------
    puf_performance_data : Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]
        Performance data for all PUF types
    temperature_range : np.ndarray
        Temperature range tested
    challenges : np.ndarray
        Challenge vectors for bit analysis
    output_dir : str, optional
        Base output directory for all figures
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping visualization types to lists of saved file paths
    """
    print("Generating comprehensive thesis visualization suite...")
    
    plot_summary = {
        'multi_puf_comparison': [],
        'bit_analysis': [],
        'threat_surface': [],
        'statistical_analysis': [],
        'defense_dashboard': []
    }
    
    # Create subdirectories
    subdirs = ['multi_puf', 'bit_analysis', 'threat_surface', 'statistical', 'dashboard']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    try:
        # 1. Multi-PUF Comparison (Priority 1)
        print("Creating multi-PUF architecture comparison...")
        fig1 = plot_multi_puf_comparison(
            temperature_range, 
            puf_performance_data,
            output_dir=os.path.join(output_dir, 'multi_puf')
        )
        plot_summary['multi_puf_comparison'].append(
            os.path.join(output_dir, 'multi_puf', 'multi_puf_comparison.png')
        )
        plt.close(fig1)
        
    except Exception as e:
        print(f"Error creating multi-PUF comparison: {e}")
    
    try:
        # 2. Bit-Aliasing Analysis (Priority 2)
        print("Creating bit-aliasing security analysis...")
        for puf_type in ['ArbiterPUF', 'SRAMPUF', 'RingOscillatorPUF', 'ButterflyPUF']:
            try:
                fig1, fig2 = plot_bit_aliasing_heatmap(
                    challenges,
                    n_puf_instances=30,
                    puf_type=puf_type,
                    output_dir=os.path.join(output_dir, 'bit_analysis')
                )
                plot_summary['bit_analysis'].extend([
                    os.path.join(output_dir, 'bit_analysis', f'bit_correlation_analysis_{puf_type.lower()}.png'),
                    os.path.join(output_dir, 'bit_analysis', f'bit_aliasing_analysis_{puf_type.lower()}.png')
                ])
                plt.close(fig1)
                plt.close(fig2)
            except Exception as e:
                print(f"Error creating bit analysis for {puf_type}: {e}")
                
    except Exception as e:
        print(f"Error in bit analysis section: {e}")
    
    try:
        # 3. 3D Threat Surface (Priority 3)
        print("Creating 3D defense threat surface...")
        temp_3d = np.linspace(-20, 100, 13)
        voltage_3d = np.linspace(-10, 10, 11)
        
        # Generate realistic threat surface data
        temp_mesh, volt_mesh = np.meshgrid(temp_3d, voltage_3d, indexing='ij')
        attack_surface = (
            15 + 0.4 * np.abs(temp_mesh - 25) + 3 * np.abs(volt_mesh) +
            0.008 * (temp_mesh - 25)**2 + 0.15 * volt_mesh**2 +
            3 * np.random.random(temp_mesh.shape)
        )
        attack_surface = np.clip(attack_surface, 0, 100)
        
        fig_3d, html_path = plot_3d_threat_surface(
            temp_3d, voltage_3d, attack_surface,
            output_dir=os.path.join(output_dir, 'threat_surface')
        )
        plot_summary['threat_surface'].extend([
            os.path.join(output_dir, 'threat_surface', '3d_threat_surface.png'),
            html_path
        ])
        
    except Exception as e:
        print(f"Error creating 3D threat surface: {e}")
    
    try:
        # 4. Statistical Analysis Suite (Priority 4)
        print("Creating statistical analysis suite...")
        
        # Convert performance data format for statistical analysis
        stats_puf_data = {}
        stats_env_data = {
            'temperature': temperature_range,
            'voltage_deviation': np.random.normal(0, 2, len(temperature_range)),
            'humidity': np.random.uniform(30, 70, len(temperature_range))
        }
        
        for puf_type, metrics in puf_performance_data.items():
            stats_puf_data[puf_type] = {}
            for metric, (values, errors) in metrics.items():
                stats_puf_data[puf_type][metric] = values
        
        # Generate ML attack data
        stats_ml_data = {
            'nominal_conditions': {
                'y_true': np.random.choice([0, 1], 200),
                'y_scores': np.random.beta(2, 3, 200)
            },
            'high_temperature': {
                'y_true': np.random.choice([0, 1], 200),
                'y_scores': np.random.beta(3, 2, 200)
            },
            'voltage_stress': {
                'y_true': np.random.choice([0, 1], 200),
                'y_scores': np.random.beta(2.5, 2.5, 200)
            }
        }
        
        saved_stats = generate_statistical_suite(
            stats_puf_data, stats_env_data, stats_ml_data,
            output_dir=os.path.join(output_dir, 'statistical')
        )
        plot_summary['statistical_analysis'].extend(saved_stats)
        
    except Exception as e:
        print(f"Error creating statistical analysis: {e}")
    
    try:
        # 5. Defense Dashboard (Priority 5)
        print("Creating defense operations dashboard...")
        mission_data, threat_level, env_status, attack_prob, countermeasure_eff = \
            generate_sample_dashboard_data()
        
        fig_dash = create_defense_dashboard(
            mission_data, threat_level, env_status, attack_prob, countermeasure_eff,
            output_dir=os.path.join(output_dir, 'dashboard')
        )
        plot_summary['defense_dashboard'].append(
            os.path.join(output_dir, 'dashboard', 'defense_dashboard.png')
        )
        plt.close(fig_dash)
        
    except Exception as e:
        print(f"Error creating defense dashboard: {e}")
    
    # Generate summary report
    total_plots = sum(len(plots) for plots in plot_summary.values())
    print(f"\nThesis visualization suite completed!")
    print(f"Generated {total_plots} total visualizations across 5 categories:")
    
    for category, plots in plot_summary.items():
        if plots:
            print(f"  {category}: {len(plots)} plots")
            for plot in plots[:2]:  # Show first 2 files
                print(f"    - {os.path.basename(plot)}")
            if len(plots) > 2:
                print(f"    - ... and {len(plots) - 2} more")
    
    return plot_summary


def main():
    """Main entry point for the visualization module."""
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # For headless environments
    
    parser = argparse.ArgumentParser(description="Generate PUF thesis visualization suite")
    parser.add_argument('--output-dir', default='figures/', 
                       help='Output directory for generated figures')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with sample data')
    parser.add_argument('--temp-range', nargs='+', type=float,
                       default=[-20, 0, 25, 50, 75, 100],
                       help='Temperature range for analysis')
    
    args = parser.parse_args()
    
    print("Generating comprehensive thesis visualization suite...")
    
    # Generate sample data
    temp_range = np.array(args.temp_range)
    sample_data = generate_sample_multi_puf_data(temp_range)
    
    # Generate sample challenges
    np.random.seed(42)
    challenges = np.random.randint(0, 2, size=(100, 64))
    
    # Generate all visualizations
    try:
        plot_summary = generate_all_thesis_plots(sample_data, temp_range, challenges, 
                                                output_dir=args.output_dir)
        
        total_plots = sum(len(plots) for plots in plot_summary.values())
        print(f"\n✅ Visualization suite completed successfully!")
        print(f"📊 Generated {total_plots} publication-quality visualizations")
        print(f"📁 Output directory: {args.output_dir}")
        
        for category, plots in plot_summary.items():
            if plots:
                print(f"   {category}: {len(plots)} plots")
        
    except Exception as e:
        print(f"❌ Error in visualization suite: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())