import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

# Try to import optional packages
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Set professional plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def plot_bit_aliasing_heatmap(
    challenges: np.ndarray,
    n_puf_instances: int = 50,
    puf_type: str = 'ArbiterPUF',
    output_dir: str = 'figures/',
    dpi: int = 300
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create sophisticated bit-aliasing analysis visualization with heatmaps and dendrograms.
    
    Parameters
    ----------
    challenges : np.ndarray
        Challenge vectors, shape (n_challenges, n_bits)
    n_puf_instances : int, optional
        Number of different PUF instances to generate (default 50)
    puf_type : str, optional
        Type of PUF to analyze (default 'ArbiterPUF')
    output_dir : str, optional
        Directory to save figures
    dpi : int, optional
        DPI for saved figures
        
    Returns
    -------
    Tuple[plt.Figure, plt.Figure]
        Correlation heatmap figure and entropy analysis figure
    """
    if not HAS_SCIPY:
        print("Warning: SciPy not available, creating simplified bit analysis")
        # Create simplified version without clustering
        return _create_simplified_bit_analysis(challenges, n_puf_instances, puf_type, output_dir, dpi)
    try:
        from .puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
    except ImportError:
        # Handle case when running as standalone script
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
    
    n_challenges, n_bits = challenges.shape
    
    # Generate multiple PUF instances
    print(f"Generating {n_puf_instances} {puf_type} instances for bit analysis...")
    
    # Choose PUF class based on type
    puf_classes = {
        'ArbiterPUF': ArbiterPUF,
        'SRAMPUF': SRAMPUF, 
        'RingOscillatorPUF': RingOscillatorPUF,
        'ButterflyPUF': ButterflyPUF
    }
    
    if puf_type not in puf_classes:
        raise ValueError(f"Unknown PUF type: {puf_type}")
    
    PUFClass = puf_classes[puf_type]
    
    # Collect responses from all PUF instances
    all_responses = []
    for i in range(n_puf_instances):
        puf = PUFClass(n_bits, seed=42 + i)
        responses = puf.eval(challenges)
        # Convert ±1 to 0/1
        binary_responses = (responses > 0).astype(int)
        all_responses.append(binary_responses)
    
    responses_matrix = np.array(all_responses)  # Shape: (n_instances, n_challenges)
    
    # Calculate bit position statistics
    bit_position_stats = []
    bit_entropies = []
    
    for bit_pos in range(n_challenges):
        # For each challenge (bit position), get responses across all PUF instances
        bit_responses = responses_matrix[:, bit_pos]  # Shape: (n_instances,)
        
        # Calculate aliasing percentage (how often it's '1')
        aliasing_percent = np.mean(bit_responses) * 100
        bit_position_stats.append(aliasing_percent)
        
        # Calculate entropy for this bit position
        p1 = np.mean(bit_responses)
        p0 = 1 - p1
        if p0 > 0 and p1 > 0:
            entropy = -p0 * np.log2(p0) - p1 * np.log2(p1)
        else:
            entropy = 0  # Completely biased
        bit_entropies.append(entropy)
    
    bit_position_stats = np.array(bit_position_stats)
    bit_entropies = np.array(bit_entropies)
    
    # Create correlation matrix between PUF instances
    print(f"Responses matrix shape: {responses_matrix.shape}")
    print(f"Sample responses: {responses_matrix[:3, :5]}")
    
    correlation_matrix = np.corrcoef(responses_matrix)
    
    # Add validation
    if np.isnan(correlation_matrix).any():
        print("Warning: NaN values in correlation matrix")
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Correlation matrix min/max: {np.min(correlation_matrix):.3f}/{np.max(correlation_matrix):.3f}")
    print(f"NaN values: {np.isnan(correlation_matrix).sum()}")
    
    # Perform hierarchical clustering for dendrograms  
    if HAS_SCIPY:
        linkage_matrix = linkage(pdist(responses_matrix, metric='hamming'), method='ward')
    else:
        linkage_matrix = None
    
    # Create first figure: Correlation heatmap with dendrograms
    fig1 = plt.figure(figsize=(16, 12))
    
    # Create grid for subplots
    gs = fig1.add_gridspec(3, 3, 
                          height_ratios=[1, 0.1, 4], 
                          width_ratios=[4, 0.1, 1],
                          hspace=0.05, wspace=0.05)
    
    # Main correlation heatmap
    ax_main = fig1.add_subplot(gs[2, 0])
    
    # Create custom colormap for correlation
    if HAS_SEABORN:
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
    else:
        cmap = 'RdBu_r'
    
    im = ax_main.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax_main.set_title(f'PUF Instance Correlation Matrix - {puf_type}', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.set_xlabel('PUF Instance ID', fontsize=12)
    ax_main.set_ylabel('PUF Instance ID', fontsize=12)
    
    # Add correlation values as text annotations (for smaller matrices)
    if n_puf_instances <= 20:
        for i in range(n_puf_instances):
            for j in range(n_puf_instances):
                text = ax_main.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
    
    # Top dendrogram (only if scipy available)
    if HAS_SCIPY and linkage_matrix is not None:
        ax_top = fig1.add_subplot(gs[0, 0], sharex=ax_main)
        dend_top = dendrogram(linkage_matrix, orientation='top', ax=ax_top, 
                             color_threshold=0.7*max(linkage_matrix[:,2]))
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.spines['left'].set_visible(False)
        
        # Right dendrogram
        ax_right = fig1.add_subplot(gs[2, 2], sharey=ax_main)
        dend_right = dendrogram(linkage_matrix, orientation='right', ax=ax_right,
                               color_threshold=0.7*max(linkage_matrix[:,2]))
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
    else:
        # Create empty subplots when scipy not available
        ax_top = fig1.add_subplot(gs[0, 0])
        ax_top.text(0.5, 0.5, 'Dendrogram requires scipy', ha='center', va='center', 
                   transform=ax_top.transAxes)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        
        ax_right = fig1.add_subplot(gs[2, 2])
        ax_right.text(0.5, 0.5, 'Dendrogram requires scipy', ha='center', va='center',
                     transform=ax_right.transAxes, rotation=90)
        ax_right.set_xticks([])
        ax_right.set_yticks([])
    
    # Colorbar
    ax_cbar = fig1.add_subplot(gs[2, 1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=12)
    
    # Second figure: Bit position analysis
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f'Bit-Aliasing Security Analysis - {puf_type}', 
                  fontsize=18, fontweight='bold')
    
    # Subplot 1: Bit aliasing heatmap
    challenge_indices = np.arange(min(n_challenges, 64))  # Limit to first 64 for readability
    aliasing_matrix = bit_position_stats[:len(challenge_indices)].reshape(1, -1)
    
    im1 = ax1.imshow(aliasing_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
    ax1.set_title('Bit Aliasing Percentage by Challenge Position', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Challenge Bit Position', fontsize=12)
    ax1.set_ylabel('Aliasing Analysis', fontsize=12)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Aliasing %'])
    
    # Add percentage annotations
    for i, val in enumerate(aliasing_matrix[0]):
        if i < len(challenge_indices):
            ax1.text(i, 0, f'{val:.1f}%', ha='center', va='center', 
                    color='white' if 20 < val < 80 else 'black', fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Aliasing Percentage (%)')
    
    # Subplot 2: Entropy per bit position
    ax2.bar(challenge_indices, bit_entropies[:len(challenge_indices)], 
           color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.set_title('Entropy per Bit Position', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Challenge Bit Position', fontsize=12)
    ax2.set_ylabel('Entropy (bits)', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Entropy')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Good Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Statistical distribution of aliasing
    ax3.hist(bit_position_stats, bins=20, color='lightcoral', alpha=0.7, 
            edgecolor='darkred', density=True)
    ax3.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Ideal (50%)')
    ax3.axvline(x=np.mean(bit_position_stats), color='blue', linestyle='-', 
               linewidth=2, label=f'Mean ({np.mean(bit_position_stats):.1f}%)')
    ax3.set_title('Distribution of Bit Aliasing Percentages', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Aliasing Percentage (%)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Quality metrics summary
    ax4.axis('off')
    
    # Calculate quality metrics
    ideal_aliasing = 50.0
    aliasing_deviation = np.mean(np.abs(bit_position_stats - ideal_aliasing))
    entropy_quality = np.mean(bit_entropies)
    worst_aliasing = np.max(np.abs(bit_position_stats - ideal_aliasing))
    
    # Determine security level
    if aliasing_deviation < 5 and entropy_quality > 0.9:
        security_level = "EXCELLENT"
        security_color = "green"
    elif aliasing_deviation < 10 and entropy_quality > 0.8:
        security_level = "GOOD"
        security_color = "orange"
    else:
        security_level = "POOR"
        security_color = "red"
    
    # Statistical tests
    _, p_value = stats.kstest(bit_position_stats, 
                             lambda x: stats.norm.cdf(x, 50, 10))
    
    metrics_text = f"""
    SECURITY ANALYSIS SUMMARY
    
    PUF Type: {puf_type}
    Instances Analyzed: {n_puf_instances}
    Challenges Tested: {n_challenges}
    
    KEY METRICS:
    Mean Aliasing Deviation: {aliasing_deviation:.2f}%
    Average Entropy: {entropy_quality:.3f} bits
    Worst Case Deviation: {worst_aliasing:.2f}%
    
    STATISTICAL TESTS:
    Normality p-value: {p_value:.4f}
    {'PASS' if p_value > 0.05 else 'FAIL'} (α = 0.05)
    
    SECURITY LEVEL: {security_level}
    
    RECOMMENDATIONS:
    • Target aliasing: 45-55%
    • Target entropy: > 0.9 bits
    • Monitor outlier bit positions
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Add security level indicator
    ax4.text(0.7, 0.3, security_level, transform=ax4.transAxes, 
            fontsize=24, fontweight='bold', color=security_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                     edgecolor=security_color, linewidth=3))
    
    plt.tight_layout()
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    
    fig1_path = os.path.join(output_dir, f'bit_correlation_analysis_{puf_type.lower()}.png')
    fig2_path = os.path.join(output_dir, f'bit_aliasing_analysis_{puf_type.lower()}.png')
    
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig2.savefig(fig2_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    print(f"Bit analysis figures saved to {fig1_path} and {fig2_path}")
    
    return fig1, fig2


def _create_simplified_bit_analysis(challenges, n_puf_instances, puf_type, output_dir, dpi):
    """Create simplified bit analysis without scipy dependencies."""
    try:
        from .puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
    except ImportError:
        # Handle case when running as standalone script
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from puf_models import ArbiterPUF, SRAMPUF, RingOscillatorPUF, ButterflyPUF
    
    print(f"Creating simplified bit analysis for {puf_type}...")
    
    # Choose PUF class
    puf_classes = {
        'ArbiterPUF': ArbiterPUF,
        'SRAMPUF': SRAMPUF, 
        'RingOscillatorPUF': RingOscillatorPUF,
        'ButterflyPUF': ButterflyPUF
    }
    
    PUFClass = puf_classes.get(puf_type, ArbiterPUF)
    n_challenges, n_bits = challenges.shape
    
    # Generate responses
    all_responses = []
    for i in range(min(n_puf_instances, 10)):  # Limit for simplified version
        puf = PUFClass(n_bits, seed=42 + i)
        responses = puf.eval(challenges)
        binary_responses = (responses > 0).astype(int)
        all_responses.append(binary_responses)
    
    responses_matrix = np.array(all_responses)
    
    # Create simplified figures
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    correlation_matrix = np.corrcoef(responses_matrix)
    
    im = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title(f'PUF Instance Correlation Matrix - {puf_type} (Simplified)', fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Correlation Coefficient')
    
    fig2, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(f'Bit-Aliasing Analysis - {puf_type} (Simplified)', fontweight='bold')
    
    # Bit aliasing analysis
    bit_stats = []
    for bit_pos in range(min(n_challenges, 64)):
        bit_responses = responses_matrix[:, bit_pos]
        aliasing_percent = np.mean(bit_responses) * 100
        bit_stats.append(aliasing_percent)
    
    # Plot aliasing percentages
    ax2.bar(range(len(bit_stats)), bit_stats, alpha=0.7)
    ax2.set_title('Bit Aliasing Percentages')
    ax2.set_xlabel('Bit Position')
    ax2.set_ylabel('Aliasing (%)')
    ax2.axhline(y=50, color='red', linestyle='--', label='Ideal (50%)')
    ax2.legend()
    
    # Plot distribution
    ax3.hist(bit_stats, bins=15, alpha=0.7, density=True)
    ax3.axvline(x=50, color='red', linestyle='--', label='Ideal (50%)')
    ax3.set_title('Distribution of Aliasing Percentages')
    ax3.set_xlabel('Aliasing (%)')
    ax3.set_ylabel('Density')
    ax3.legend()
    
    # Simple entropy calculation
    entropies = []
    for bit_pos in range(len(bit_stats)):
        p = bit_stats[bit_pos] / 100
        if 0 < p < 1:
            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        else:
            entropy = 0
        entropies.append(entropy)
    
    ax4.bar(range(len(entropies)), entropies, alpha=0.7, color='steelblue')
    ax4.set_title('Entropy per Bit Position')
    ax4.set_xlabel('Bit Position')
    ax4.set_ylabel('Entropy (bits)')
    ax4.axhline(y=1.0, color='red', linestyle='--', label='Perfect Entropy')
    ax4.legend()
    
    # Summary metrics
    ax5.axis('off')
    mean_aliasing = np.mean(bit_stats)
    mean_entropy = np.mean(entropies)
    
    summary_text = f"""
    SIMPLIFIED ANALYSIS SUMMARY
    
    PUF Type: {puf_type}
    Instances: {len(all_responses)}
    Challenges: {len(bit_stats)}
    
    Mean Aliasing: {mean_aliasing:.1f}%
    Mean Entropy: {mean_entropy:.3f} bits
    
    Status: {'GOOD' if 45 <= mean_aliasing <= 55 else 'NEEDS REVIEW'}
    """
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figures
    os.makedirs(output_dir, exist_ok=True)
    fig1_path = os.path.join(output_dir, f'bit_correlation_simplified_{puf_type.lower()}.png')
    fig2_path = os.path.join(output_dir, f'bit_aliasing_simplified_{puf_type.lower()}.png')
    
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches='tight')
    fig2.savefig(fig2_path, dpi=dpi, bbox_inches='tight')
    
    return fig1, fig2


def main():
    """Main entry point for the bit analysis module."""
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # For headless environments
    
    parser = argparse.ArgumentParser(description="Generate PUF bit-aliasing security analysis")
    parser.add_argument('--puf-type', choices=['ArbiterPUF', 'SRAMPUF', 'RingOscillatorPUF', 'ButterflyPUF'],
                       default='ArbiterPUF', help='PUF type to analyze')
    parser.add_argument('--n-instances', type=int, default=30,
                       help='Number of PUF instances to generate')
    parser.add_argument('--n-challenges', type=int, default=100,
                       help='Number of challenges to use')
    parser.add_argument('--n-bits', type=int, default=64,
                       help='Number of bits per challenge')
    parser.add_argument('--output-dir', default='figures/bit_analysis/',
                       help='Output directory for generated figures')
    
    args = parser.parse_args()
    
    print(f"Generating bit-aliasing analysis for {args.puf_type}...")
    
    # Generate sample challenges
    np.random.seed(42)
    challenges = np.random.randint(0, 2, size=(args.n_challenges, args.n_bits))
    
    try:
        fig1, fig2 = plot_bit_aliasing_heatmap(
            challenges, 
            n_puf_instances=args.n_instances,
            puf_type=args.puf_type,
            output_dir=args.output_dir
        )
        
        print(f"✅ Bit aliasing analysis completed successfully for {args.puf_type}")
        print(f"📁 Figures saved to: {args.output_dir}")
        
        # Close figures to save memory
        plt.close(fig1)
        plt.close(fig2)
        
    except Exception as e:
        print(f"❌ Error in bit aliasing analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())