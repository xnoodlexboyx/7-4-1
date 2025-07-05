import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

# Military color scheme
MILITARY_COLORS = {
    'bg_dark': '#1a1a1a',
    'bg_medium': '#2d2d2d', 
    'bg_light': '#404040',
    'green_safe': '#00ff41',
    'amber_caution': '#ffaa00',
    'red_danger': '#ff0041',
    'blue_info': '#0099ff',
    'white_text': '#ffffff',
    'gray_text': '#cccccc'
}

def create_defense_dashboard(
    mission_data: Dict[str, np.ndarray],
    threat_level: float,
    environmental_status: Dict[str, float],
    attack_probability: np.ndarray,
    countermeasure_effectiveness: np.ndarray,
    output_dir: str = 'figures/',
    save_format: str = 'png',
    dpi: int = 300
) -> plt.Figure:
    """
    Create a multi-panel defense scenario dashboard with military styling.
    
    Parameters
    ----------
    mission_data : Dict[str, np.ndarray]
        Mission timeline data with keys: 'time_months', 'reliability', 'security_level'
    threat_level : float
        Current threat level (0-100)
    environmental_status : Dict[str, float]
        Current environmental readings: 'temperature', 'radiation', 'emi'
    attack_probability : np.ndarray
        Attack success probability over time
    countermeasure_effectiveness : np.ndarray
        Countermeasure effectiveness over time
    output_dir : str, optional
        Directory to save the dashboard
    save_format : str, optional
        File format for saving ('png', 'pdf', 'svg')
    dpi : int, optional
        DPI for raster formats
        
    Returns
    -------
    plt.Figure
        The dashboard figure object
    """
    
    # Set up the figure with dark military background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 14), facecolor=MILITARY_COLORS['bg_dark'])
    
    # Create grid layout for dashboard panels
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1.2], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.25, left=0.05, right=0.95, 
                         top=0.93, bottom=0.07)
    
    # Panel 1: Mission Timeline (top, spanning 4 columns)
    ax_timeline = fig.add_subplot(gs[0, :])
    _create_mission_timeline(ax_timeline, mission_data)
    
    # Panel 2: Threat Level Gauge (middle-left)
    ax_threat = fig.add_subplot(gs[1, 0])
    _create_threat_gauge(ax_threat, threat_level)
    
    # Panel 3: Environmental Indicators (middle-center, 2 columns)
    ax_env = fig.add_subplot(gs[1, 1:3])
    _create_environmental_indicators(ax_env, environmental_status)
    
    # Panel 4: System Status (middle-right)
    ax_status = fig.add_subplot(gs[1, 3])
    _create_system_status(ax_status, environmental_status)
    
    # Panel 5: Attack Probability & Countermeasures (bottom, spanning 4 columns)
    ax_counter = fig.add_subplot(gs[2, :])
    _create_countermeasure_panel(ax_counter, attack_probability, countermeasure_effectiveness)
    
    # Add main title and timestamp
    fig.suptitle('DEFENSE OPERATIONS DASHBOARD - PUF SECURITY MONITORING', 
                fontsize=24, fontweight='bold', color=MILITARY_COLORS['white_text'], y=0.98)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    fig.text(0.99, 0.01, f'Last Update: {timestamp}', fontsize=12, 
            color=MILITARY_COLORS['gray_text'], ha='right', va='bottom')
    
    # Add classification marking
    fig.text(0.01, 0.99, 'UNCLASSIFIED//FOR OFFICIAL USE ONLY', fontsize=10,
            color=MILITARY_COLORS['green_safe'], ha='left', va='top', fontweight='bold')
    
    # Save the dashboard
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'defense_dashboard.{save_format}')
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor=MILITARY_COLORS['bg_dark'])
    
    return fig


def _create_mission_timeline(ax: plt.Axes, mission_data: Dict[str, np.ndarray]) -> None:
    """Create mission timeline showing PUF reliability over 5-year deployment."""
    
    # Generate sample data if not provided
    if not mission_data:
        time_months = np.linspace(0, 60, 61)  # 5 years
        reliability = 98 - 0.3 * time_months + 2 * np.sin(time_months / 6) + \
                     np.random.normal(0, 1, len(time_months))
        security_level = 95 - 0.2 * time_months + 1.5 * np.sin(time_months / 8) + \
                        np.random.normal(0, 0.8, len(time_months))
        mission_data = {
            'time_months': time_months,
            'reliability': np.clip(reliability, 70, 100),
            'security_level': np.clip(security_level, 70, 100)
        }
    
    time_months = mission_data['time_months']
    reliability = mission_data['reliability']
    security_level = mission_data.get('security_level', reliability - 5)
    
    # Plot reliability and security trends
    ax.plot(time_months, reliability, linewidth=3, color=MILITARY_COLORS['green_safe'], 
           label='PUF Reliability', marker='o', markersize=4)
    ax.plot(time_months, security_level, linewidth=3, color=MILITARY_COLORS['blue_info'],
           label='Security Level', marker='s', markersize=4)
    
    # Add critical thresholds
    ax.axhline(y=90, color=MILITARY_COLORS['amber_caution'], linestyle='--', linewidth=2,
              alpha=0.8, label='Caution Threshold')
    ax.axhline(y=80, color=MILITARY_COLORS['red_danger'], linestyle='--', linewidth=2,
              alpha=0.8, label='Critical Threshold')
    
    # Highlight mission phases
    phase_colors = [MILITARY_COLORS['bg_light'], MILITARY_COLORS['bg_medium']]
    phase_labels = ['Deployment', 'Operations', 'Maintenance', 'Extended Ops', 'EOL']
    phase_boundaries = [0, 12, 24, 36, 48, 60]
    
    for i in range(len(phase_boundaries) - 1):
        start, end = phase_boundaries[i], phase_boundaries[i + 1]
        ax.axvspan(start, end, alpha=0.2, color=phase_colors[i % 2])
        
        # Add phase labels
        mid_point = (start + end) / 2
        ax.text(mid_point, 102, phase_labels[i], ha='center', va='bottom',
               color=MILITARY_COLORS['white_text'], fontweight='bold', fontsize=10)
    
    # Customize timeline
    ax.set_title('MISSION TIMELINE - 5 YEAR DEPLOYMENT CYCLE', 
                fontsize=16, fontweight='bold', color=MILITARY_COLORS['white_text'])
    ax.set_xlabel('Mission Time (Months)', fontsize=12, color=MILITARY_COLORS['white_text'])
    ax.set_ylabel('Performance (%)', fontsize=12, color=MILITARY_COLORS['white_text'])
    ax.set_xlim(0, 60)
    ax.set_ylim(70, 105)
    ax.grid(True, alpha=0.3, color=MILITARY_COLORS['gray_text'])
    ax.legend(loc='lower left', fontsize=11)
    
    # Add current mission status
    current_month = time_months[-1] * 0.3  # Assume 30% through mission
    current_reliability = np.interp(current_month, time_months, reliability)
    ax.axvline(current_month, color=MILITARY_COLORS['red_danger'], linewidth=3,
              label='Current Position')
    ax.text(current_month + 1, current_reliability + 2, 
           f'NOW\n{current_reliability:.1f}%', ha='left', va='bottom',
           color=MILITARY_COLORS['red_danger'], fontweight='bold', fontsize=12)


def _create_threat_gauge(ax: plt.Axes, threat_level: float) -> None:
    """Create circular threat level gauge."""
    
    # Clear axis
    ax.clear()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Determine threat color
    if threat_level < 30:
        color = MILITARY_COLORS['green_safe']
        status = 'LOW'
    elif threat_level < 70:
        color = MILITARY_COLORS['amber_caution']
        status = 'MODERATE'
    else:
        color = MILITARY_COLORS['red_danger']
        status = 'HIGH'
    
    # Draw gauge background
    circle_bg = Circle((0, 0), 1, facecolor=MILITARY_COLORS['bg_medium'], 
                      edgecolor=MILITARY_COLORS['white_text'], linewidth=3)
    ax.add_patch(circle_bg)
    
    # Draw threat level arc
    angle = (threat_level / 100) * 270 - 135  # Start from bottom-left
    wedge = Wedge((0, 0), 1, -135, angle, facecolor=color, alpha=0.8)
    ax.add_patch(wedge)
    
    # Add scale markings
    for i in range(0, 101, 25):
        angle_rad = np.radians(i * 2.7 - 135)
        x1, y1 = 0.85 * np.cos(angle_rad), 0.85 * np.sin(angle_rad)
        x2, y2 = 0.95 * np.cos(angle_rad), 0.95 * np.sin(angle_rad)
        ax.plot([x1, x2], [y1, y2], color=MILITARY_COLORS['white_text'], linewidth=2)
        
        # Add numbers
        x_text, y_text = 0.75 * np.cos(angle_rad), 0.75 * np.sin(angle_rad)
        ax.text(x_text, y_text, str(i), ha='center', va='center',
               color=MILITARY_COLORS['white_text'], fontsize=10, fontweight='bold')
    
    # Add center text
    ax.text(0, 0.2, f'{threat_level:.0f}%', ha='center', va='center',
           color=MILITARY_COLORS['white_text'], fontsize=20, fontweight='bold')
    ax.text(0, -0.1, status, ha='center', va='center',
           color=color, fontsize=14, fontweight='bold')
    ax.text(0, -0.3, 'THREAT LEVEL', ha='center', va='center',
           color=MILITARY_COLORS['gray_text'], fontsize=10)


def _create_environmental_indicators(ax: plt.Axes, env_status: Dict[str, float]) -> None:
    """Create environmental stress indicators."""
    
    factors = ['Temperature', 'Radiation', 'EMI']
    values = [env_status.get('temperature', 45), env_status.get('radiation', 25), 
             env_status.get('emi', 35)]
    limits = [60, 50, 40]  # Warning thresholds
    
    # Create horizontal bar chart
    y_pos = np.arange(len(factors))
    colors = []
    
    for val, limit in zip(values, limits):
        if val < limit * 0.7:
            colors.append(MILITARY_COLORS['green_safe'])
        elif val < limit:
            colors.append(MILITARY_COLORS['amber_caution'])
        else:
            colors.append(MILITARY_COLORS['red_danger'])
    
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.6)
    
    # Add warning threshold lines
    for i, limit in enumerate(limits):
        ax.axvline(limit, color=MILITARY_COLORS['red_danger'], linestyle='--', 
                  alpha=0.7, linewidth=2)
        ax.text(limit + 1, i, f'Limit: {limit}', va='center', 
               color=MILITARY_COLORS['red_danger'], fontsize=10)
    
    # Add value labels
    for i, (val, bar) in enumerate(zip(values, bars)):
        ax.text(val + 1, i, f'{val:.1f}', va='center', ha='left',
               color=MILITARY_COLORS['white_text'], fontweight='bold', fontsize=12)
    
    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, color=MILITARY_COLORS['white_text'], fontsize=12)
    ax.set_xlabel('Stress Level', color=MILITARY_COLORS['white_text'], fontsize=12)
    ax.set_title('ENVIRONMENTAL STRESS INDICATORS', 
                color=MILITARY_COLORS['white_text'], fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(limits) * 1.3)
    ax.grid(True, alpha=0.3, axis='x')


def _create_system_status(ax: plt.Axes, env_status: Dict[str, float]) -> None:
    """Create system status indicator panel."""
    
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # System status based on environmental factors
    temp = env_status.get('temperature', 45)
    radiation = env_status.get('radiation', 25)
    emi = env_status.get('emi', 35)
    
    # Calculate overall system health
    health = 100 - (temp/60 + radiation/50 + emi/40) * 100 / 3
    
    if health > 85:
        status_color = MILITARY_COLORS['green_safe']
        status_text = 'OPERATIONAL'
        status_icon = '●'
    elif health > 70:
        status_color = MILITARY_COLORS['amber_caution']
        status_text = 'DEGRADED'
        status_icon = '▲'
    else:
        status_color = MILITARY_COLORS['red_danger']
        status_text = 'CRITICAL'
        status_icon = '■'
    
    # Create status display
    ax.text(0.5, 0.8, status_icon, ha='center', va='center',
           color=status_color, fontsize=60, fontweight='bold')
    ax.text(0.5, 0.5, status_text, ha='center', va='center',
           color=status_color, fontsize=16, fontweight='bold')
    ax.text(0.5, 0.35, f'{health:.0f}% HEALTH', ha='center', va='center',
           color=MILITARY_COLORS['white_text'], fontsize=12)
    
    # Add subsystem status
    subsystems = ['PUF Core', 'Crypto Engine', 'Environmental Mon.']
    subsystem_health = [health + np.random.normal(0, 5) for _ in subsystems]
    
    for i, (system, sys_health) in enumerate(zip(subsystems, subsystem_health)):
        y_pos = 0.15 - i * 0.05
        sys_health = np.clip(sys_health, 0, 100)
        
        if sys_health > 85:
            sys_color = MILITARY_COLORS['green_safe']
        elif sys_health > 70:
            sys_color = MILITARY_COLORS['amber_caution']
        else:
            sys_color = MILITARY_COLORS['red_danger']
        
        ax.text(0.1, y_pos, f'{system}:', ha='left', va='center',
               color=MILITARY_COLORS['gray_text'], fontsize=10)
        ax.text(0.9, y_pos, f'{sys_health:.0f}%', ha='right', va='center',
               color=sys_color, fontsize=10, fontweight='bold')


def _create_countermeasure_panel(ax: plt.Axes, attack_prob: np.ndarray, 
                               countermeasure_eff: np.ndarray) -> None:
    """Create attack probability and countermeasure effectiveness panel."""
    
    # Generate sample data if not provided
    if len(attack_prob) == 0:
        time_hours = np.linspace(0, 24, 25)  # 24 hours
        attack_prob = 20 + 10 * np.sin(time_hours / 4) + 5 * np.random.random(len(time_hours))
        countermeasure_eff = 85 + 5 * np.cos(time_hours / 6) + 3 * np.random.random(len(time_hours))
    else:
        time_hours = np.arange(len(attack_prob))
    
    # Create twin axes
    ax2 = ax.twinx()
    
    # Plot attack probability
    line1 = ax.fill_between(time_hours, attack_prob, alpha=0.7, 
                           color=MILITARY_COLORS['red_danger'], label='Attack Success Probability')
    ax.plot(time_hours, attack_prob, color=MILITARY_COLORS['red_danger'], linewidth=3)
    
    # Plot countermeasure effectiveness
    line2 = ax2.plot(time_hours, countermeasure_eff, color=MILITARY_COLORS['green_safe'], 
                    linewidth=3, label='Countermeasure Effectiveness', marker='o', markersize=3)
    
    # Add critical thresholds
    ax.axhline(y=30, color=MILITARY_COLORS['amber_caution'], linestyle='--', 
              linewidth=2, alpha=0.8, label='Attack Threshold')
    ax2.axhline(y=80, color=MILITARY_COLORS['blue_info'], linestyle='--', 
               linewidth=2, alpha=0.8, label='Min. Effectiveness')
    
    # Customize axes
    ax.set_title('ATTACK PROBABILITY & COUNTERMEASURE EFFECTIVENESS (24H)', 
                fontsize=16, fontweight='bold', color=MILITARY_COLORS['white_text'])
    ax.set_xlabel('Time (Hours)', fontsize=12, color=MILITARY_COLORS['white_text'])
    ax.set_ylabel('Attack Success Probability (%)', fontsize=12, 
                 color=MILITARY_COLORS['red_danger'])
    ax2.set_ylabel('Countermeasure Effectiveness (%)', fontsize=12, 
                  color=MILITARY_COLORS['green_safe'])
    
    ax.set_ylim(0, 60)
    ax2.set_ylim(60, 100)
    ax.grid(True, alpha=0.3)
    
    # Add current status annotation
    current_attack = attack_prob[-1]
    current_counter = countermeasure_eff[-1]
    
    status_text = f'CURRENT STATUS:\nAttack Risk: {current_attack:.1f}%\nDefense Level: {current_counter:.1f}%'
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, va='top', ha='left',
           color=MILITARY_COLORS['white_text'], fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=MILITARY_COLORS['bg_medium'], alpha=0.8))


def generate_sample_dashboard_data() -> Tuple[Dict, float, Dict, np.ndarray, np.ndarray]:
    """Generate sample data for testing the defense dashboard."""
    np.random.seed(42)
    
    # Mission timeline data
    time_months = np.linspace(0, 60, 61)
    reliability = 98 - 0.3 * time_months + 2 * np.sin(time_months / 6) + \
                 np.random.normal(0, 1, len(time_months))
    security_level = 95 - 0.2 * time_months + 1.5 * np.sin(time_months / 8) + \
                    np.random.normal(0, 0.8, len(time_months))
    
    mission_data = {
        'time_months': time_months,
        'reliability': np.clip(reliability, 70, 100),
        'security_level': np.clip(security_level, 70, 100)
    }
    
    # Current threat level
    threat_level = np.random.uniform(20, 80)
    
    # Environmental status
    env_status = {
        'temperature': np.random.uniform(35, 65),
        'radiation': np.random.uniform(15, 45),
        'emi': np.random.uniform(20, 50)
    }
    
    # Attack probability and countermeasures over 24 hours
    time_hours = np.linspace(0, 24, 25)
    attack_prob = 20 + 10 * np.sin(time_hours / 4) + 5 * np.random.random(len(time_hours))
    countermeasure_eff = 85 + 5 * np.cos(time_hours / 6) + 3 * np.random.random(len(time_hours))
    
    return mission_data, threat_level, env_status, attack_prob, countermeasure_eff


def main():
    """Main entry point for the defense dashboard module."""
    import argparse
    import matplotlib
    matplotlib.use('Agg')  # For headless environments
    
    parser = argparse.ArgumentParser(description="Generate defense operations dashboard")
    parser.add_argument('--output-dir', default='figures/dashboard/',
                       help='Output directory for generated dashboard')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format for dashboard')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for raster formats')
    
    args = parser.parse_args()
    
    print("Generating defense operations dashboard...")
    
    try:
        # Generate sample data
        mission_data, threat_level, env_status, attack_prob, countermeasure_eff = \
            generate_sample_dashboard_data()
        
        # Create dashboard
        fig = create_defense_dashboard(mission_data, threat_level, env_status,
                                     attack_prob, countermeasure_eff,
                                     output_dir=args.output_dir,
                                     save_format=args.format,
                                     dpi=args.dpi)
        
        print("✅ Defense dashboard created successfully!")
        print(f"🎯 Threat Level: {threat_level:.1f}%")
        print(f"🌡️  Environmental Status:")
        for factor, value in env_status.items():
            print(f"   - {factor.title()}: {value:.1f}")
        print(f"📁 Dashboard saved to: {args.output_dir}")
        
        plt.close(fig)  # Close to save memory
        
    except Exception as e:
        print(f"❌ Error creating defense dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())