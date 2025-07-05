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

# Military compliance standards and specifications
MILITARY_COMPLIANCE = {
    'standards': {
        'MIL-STD-810H': 'Environmental Engineering Considerations and Laboratory Tests',
        'MIL-STD-461G': 'Requirements for the Control of Electromagnetic Interference',
        'MIL-STD-704F': 'Aircraft Electric Power Characteristics',
        'MIL-STD-1275E': 'Characteristics of 28 Volt DC Electrical Systems',
        'FIPS-140-2': 'Security Requirements for Cryptographic Modules',
        'Common Criteria': 'Information Technology Security Evaluation',
        'DO-178C': 'Software Considerations in Airborne Systems',
        'DO-254': 'Design Assurance Guidance for Airborne Electronic Hardware'
    },
    'environmental_limits': {
        'temperature': {'min': -55, 'max': 125, 'unit': '°C', 'standard': 'MIL-STD-810H'},
        'altitude': {'min': -152, 'max': 12192, 'unit': 'm', 'standard': 'MIL-STD-810H'},
        'vibration': {'max': 20, 'unit': 'g RMS', 'standard': 'MIL-STD-810H'},
        'shock': {'max': 40, 'unit': 'g', 'standard': 'MIL-STD-810H'},
        'humidity': {'max': 95, 'unit': '% RH', 'standard': 'MIL-STD-810H'},
        'emi_susceptibility': {'max': 200, 'unit': 'V/m', 'standard': 'MIL-STD-461G'},
        'radiation': {'max': 100, 'unit': 'krad', 'standard': 'MIL-HDBK-814'}
    },
    'security_levels': {
        'UNCLASSIFIED': {'color': '#00ff41', 'priority': 1},
        'CONFIDENTIAL': {'color': '#ffaa00', 'priority': 2},
        'SECRET': {'color': '#ff6600', 'priority': 3},
        'TOP SECRET': {'color': '#ff0041', 'priority': 4}
    },
    'tamper_evidence': {
        'LEVEL_1': 'Basic visual inspection',
        'LEVEL_2': 'Destructive analysis required',
        'LEVEL_3': 'Sophisticated equipment required',
        'LEVEL_4': 'Nation-state level resources required'
    }
}

def create_defense_dashboard(
    mission_data: Dict[str, np.ndarray],
    threat_level: float,
    environmental_status: Dict[str, float],
    attack_probability: np.ndarray,
    countermeasure_effectiveness: np.ndarray,
    output_dir: str = 'figures/',
    save_format: str = 'png',
    dpi: int = 300,
    military_compliance: Optional[Dict[str, str]] = None
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
    military_compliance : Optional[Dict[str, str]], optional
        Military standards compliance status
        
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
    
    # Add military compliance panel
    if military_compliance is None:
        military_compliance = _generate_default_compliance(environmental_status, threat_level)
    
    _add_compliance_annotations(fig, military_compliance)
    
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


def _generate_default_compliance(env_status: Dict[str, float], threat_level: float) -> Dict[str, str]:
    """Generate default military compliance status based on environmental conditions."""
    
    # Military standards and their criteria
    compliance = {}
    
    # MIL-STD-810 Environmental Engineering
    temp = env_status.get('temperature', 45)
    if temp <= 55:
        compliance['MIL-STD-810'] = 'COMPLIANT'
    elif temp <= 70:
        compliance['MIL-STD-810'] = 'MARGINAL'
    else:
        compliance['MIL-STD-810'] = 'NON-COMPLIANT'
    
    # MIL-STD-461 Electromagnetic Interference
    emi = env_status.get('emi', 35)
    if emi <= 30:
        compliance['MIL-STD-461'] = 'COMPLIANT'
    elif emi <= 45:
        compliance['MIL-STD-461'] = 'MARGINAL'
    else:
        compliance['MIL-STD-461'] = 'NON-COMPLIANT'
    
    # FIPS 140-2 Cryptographic Module Security
    if threat_level <= 30:
        compliance['FIPS-140-2'] = 'LEVEL 3'
    elif threat_level <= 60:
        compliance['FIPS-140-2'] = 'LEVEL 2'
    else:
        compliance['FIPS-140-2'] = 'LEVEL 1'
    
    # DO-178C Software Considerations (based on system health)
    radiation = env_status.get('radiation', 25)
    system_health = 100 - (temp/60 + radiation/50 + emi/40) * 100 / 3
    
    if system_health >= 90:
        compliance['DO-178C'] = 'DAL A'
    elif system_health >= 80:
        compliance['DO-178C'] = 'DAL B'
    elif system_health >= 70:
        compliance['DO-178C'] = 'DAL C'
    else:
        compliance['DO-178C'] = 'DAL D'
    
    # NATO STANAG Agreement
    overall_compliance = sum([
        1 if status in ['COMPLIANT', 'LEVEL 3', 'DAL A'] else
        0.5 if status in ['MARGINAL', 'LEVEL 2', 'DAL B'] else 0
        for status in compliance.values()
    ]) / len(compliance)
    
    if overall_compliance >= 0.8:
        compliance['NATO-STANAG'] = 'APPROVED'
    elif overall_compliance >= 0.6:
        compliance['NATO-STANAG'] = 'CONDITIONAL'
    else:
        compliance['NATO-STANAG'] = 'RESTRICTED'
    
    return compliance


def _add_compliance_annotations(fig: plt.Figure, compliance: Dict[str, str]) -> None:
    """Add military compliance annotations to the dashboard."""
    
    # Create compliance status panel in top-right corner
    compliance_text = "MILITARY COMPLIANCE STATUS\n" + "="*35 + "\n"
    
    # Color coding for compliance status
    status_colors = {
        'COMPLIANT': MILITARY_COLORS['green_safe'],
        'MARGINAL': MILITARY_COLORS['amber_caution'],
        'NON-COMPLIANT': MILITARY_COLORS['red_danger'],
        'LEVEL 3': MILITARY_COLORS['green_safe'],
        'LEVEL 2': MILITARY_COLORS['amber_caution'],
        'LEVEL 1': MILITARY_COLORS['red_danger'],
        'DAL A': MILITARY_COLORS['green_safe'],
        'DAL B': MILITARY_COLORS['blue_info'],
        'DAL C': MILITARY_COLORS['amber_caution'],
        'DAL D': MILITARY_COLORS['red_danger'],
        'APPROVED': MILITARY_COLORS['green_safe'],
        'CONDITIONAL': MILITARY_COLORS['amber_caution'],
        'RESTRICTED': MILITARY_COLORS['red_danger']
    }
    
    # Standard descriptions
    standard_descriptions = {
        'MIL-STD-810': 'Environmental Engineering',
        'MIL-STD-461': 'EMI Requirements',
        'FIPS-140-2': 'Cryptographic Security',
        'DO-178C': 'Software Assurance',
        'NATO-STANAG': 'NATO Interoperability'
    }
    
    y_offset = 0.90
    for standard, status in compliance.items():
        color = status_colors.get(status, MILITARY_COLORS['white_text'])
        description = standard_descriptions.get(standard, 'Defense Standard')
        
        # Add standard name and status
        fig.text(0.75, y_offset, f"{standard}:", fontsize=11, fontweight='bold',
                color=MILITARY_COLORS['white_text'], ha='left', va='top')
        
        fig.text(0.95, y_offset, status, fontsize=11, fontweight='bold',
                color=color, ha='right', va='top')
        
        # Add description below
        fig.text(0.75, y_offset - 0.015, description, fontsize=9,
                color=MILITARY_COLORS['gray_text'], ha='left', va='top', style='italic')
        
        y_offset -= 0.05
    
    # Add compliance summary box
    compliant_count = sum(1 for status in compliance.values() 
                         if status in ['COMPLIANT', 'LEVEL 3', 'DAL A', 'APPROVED'])
    total_standards = len(compliance)
    compliance_rate = compliant_count / total_standards * 100
    
    if compliance_rate >= 80:
        summary_color = MILITARY_COLORS['green_safe']
        summary_status = 'MISSION READY'
    elif compliance_rate >= 60:
        summary_color = MILITARY_COLORS['amber_caution']
        summary_status = 'CONDITIONAL READY'
    else:
        summary_color = MILITARY_COLORS['red_danger']
        summary_status = 'NOT READY'
    
    # Add summary box
    fig.text(0.75, y_offset - 0.03, 
            f"OVERALL COMPLIANCE: {compliance_rate:.0f}%\nSTATUS: {summary_status}",
            fontsize=12, fontweight='bold', color=summary_color, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=MILITARY_COLORS['bg_medium'], 
                     edgecolor=summary_color, linewidth=2, alpha=0.9))
    
    # Add compliance legend
    legend_y = y_offset - 0.15
    fig.text(0.75, legend_y, "COMPLIANCE LEGEND:", fontsize=10, fontweight='bold',
            color=MILITARY_COLORS['white_text'], ha='left', va='top')
    
    legend_items = [
        ("● COMPLIANT/LEVEL 3", MILITARY_COLORS['green_safe']),
        ("● MARGINAL/LEVEL 2", MILITARY_COLORS['amber_caution']),
        ("● NON-COMPLIANT/LEVEL 1", MILITARY_COLORS['red_danger'])
    ]
    
    for i, (item, color) in enumerate(legend_items):
        fig.text(0.75, legend_y - 0.025 - i*0.02, item, fontsize=9,
                color=color, ha='left', va='top')


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


def create_military_compliance_dashboard(
    puf_performance: Dict[str, float],
    environmental_data: Dict[str, float],
    attack_assessment: Dict[str, float],
    mission_profile: Dict[str, str],
    output_dir: str = 'figures/dashboard/',
    save_format: str = 'png',
    dpi: int = 300
) -> plt.Figure:
    """
    Create comprehensive military compliance dashboard with PUF security assessment.
    
    Parameters
    ----------
    puf_performance : Dict[str, float]
        PUF performance metrics: {'reliability', 'uniqueness', 'attack_resistance'}
    environmental_data : Dict[str, float]
        Environmental conditions: {'temperature', 'radiation', 'emi', 'voltage'}
    attack_assessment : Dict[str, float]
        Attack vulnerability: {'ml_attacks', 'side_channel', 'physical_attacks'}
    mission_profile : Dict[str, str]
        Mission context: {'mission_type', 'security_clearance', 'deployment_environment'}
    output_dir : str, optional
        Output directory for dashboard
    save_format : str, optional
        File format ('png', 'pdf', 'svg')
    dpi : int, optional
        DPI for raster formats
        
    Returns
    -------
    plt.Figure
        Enhanced military compliance dashboard
    """
    
    # Generate comprehensive mission data
    mission_data = {
        'time_months': np.linspace(0, 60, 61),
        'reliability': np.full(61, puf_performance.get('reliability', 95)),
        'security_level': np.full(61, puf_performance.get('attack_resistance', 85))
    }
    
    # Calculate threat level based on attack assessment
    threat_level = np.mean([
        attack_assessment.get('ml_attacks', 0) * 100,
        attack_assessment.get('side_channel', 0) * 100, 
        attack_assessment.get('physical_attacks', 0) * 100
    ])
    
    # Generate attack probability timeline
    attack_prob = np.random.uniform(threat_level - 10, threat_level + 10, 25)
    countermeasure_eff = np.random.uniform(85, 95, 25)
    
    # Generate military compliance based on all factors
    compliance = _generate_enhanced_compliance(
        environmental_data, threat_level, puf_performance, mission_profile
    )
    
    # Create enhanced dashboard
    fig = create_defense_dashboard(
        mission_data, threat_level, environmental_data,
        attack_prob, countermeasure_eff, output_dir, save_format, dpi, compliance
    )
    
    # Add mission-specific annotations
    _add_mission_context_annotations(fig, mission_profile, puf_performance)
    
    # Save with military-specific filename
    os.makedirs(output_dir, exist_ok=True)
    military_filename = os.path.join(output_dir, f'military_compliance_dashboard.{save_format}')
    fig.savefig(military_filename, dpi=dpi, bbox_inches='tight', facecolor=MILITARY_COLORS['bg_dark'])
    
    return fig


def _generate_enhanced_compliance(
    env_data: Dict[str, float], 
    threat_level: float,
    puf_performance: Dict[str, float],
    mission_profile: Dict[str, str]
) -> Dict[str, str]:
    """Generate enhanced compliance assessment considering all factors."""
    
    compliance = _generate_default_compliance(env_data, threat_level)
    
    # Enhance based on PUF performance
    reliability = puf_performance.get('reliability', 95)
    uniqueness = puf_performance.get('uniqueness', 49.5)
    attack_resistance = puf_performance.get('attack_resistance', 85)
    
    # Adjust FIPS 140-2 based on PUF security
    if attack_resistance >= 90 and reliability >= 95:
        compliance['FIPS-140-2'] = 'LEVEL 4'
    elif attack_resistance >= 85 and reliability >= 90:
        if compliance['FIPS-140-2'] == 'LEVEL 1':
            compliance['FIPS-140-2'] = 'LEVEL 2'
    
    # Add mission-specific standards
    mission_type = mission_profile.get('mission_type', 'general')
    security_clearance = mission_profile.get('security_clearance', 'CONFIDENTIAL')
    
    if mission_type in ['satellite', 'aerospace']:
        # Space systems require additional standards
        if reliability >= 99 and env_data.get('radiation', 25) <= 30:
            compliance['AS9100'] = 'CERTIFIED'
        else:
            compliance['AS9100'] = 'NON-CERTIFIED'
    
    elif mission_type in ['naval', 'submarine']:
        # Naval systems have specific requirements
        if reliability >= 95 and env_data.get('temperature', 45) <= 50:
            compliance['MIL-STD-1399'] = 'COMPLIANT'
        else:
            compliance['MIL-STD-1399'] = 'NON-COMPLIANT'
    
    # Security clearance compliance
    if security_clearance == 'TOP_SECRET':
        min_attack_resistance = 95
    elif security_clearance == 'SECRET':
        min_attack_resistance = 90
    elif security_clearance == 'CONFIDENTIAL':
        min_attack_resistance = 85
    else:
        min_attack_resistance = 80
    
    if attack_resistance >= min_attack_resistance:
        compliance[f'SECURITY-{security_clearance}'] = 'AUTHORIZED'
    else:
        compliance[f'SECURITY-{security_clearance}'] = 'UNAUTHORIZED'
    
    return compliance


def _add_mission_context_annotations(
    fig: plt.Figure, 
    mission_profile: Dict[str, str],
    puf_performance: Dict[str, float]
) -> None:
    """Add mission context and PUF performance annotations."""
    
    # Mission context box
    mission_type = mission_profile.get('mission_type', 'General Defense')
    security_clearance = mission_profile.get('security_clearance', 'CONFIDENTIAL')
    deployment_env = mission_profile.get('deployment_environment', 'Field')
    
    context_text = f"""MISSION CONTEXT
Mission Type: {mission_type.upper()}
Security Level: {security_clearance}
Environment: {deployment_env}
"""
    
    fig.text(0.01, 0.85, context_text, fontsize=10, fontweight='bold',
            color=MILITARY_COLORS['blue_info'], ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=MILITARY_COLORS['bg_medium'], 
                     edgecolor=MILITARY_COLORS['blue_info'], linewidth=1, alpha=0.9))
    
    # PUF performance summary
    reliability = puf_performance.get('reliability', 95)
    uniqueness = puf_performance.get('uniqueness', 49.5)
    attack_resistance = puf_performance.get('attack_resistance', 85)
    
    # Determine overall PUF grade
    overall_score = (reliability + uniqueness + attack_resistance) / 3
    if overall_score >= 95:
        grade = 'A+'
        grade_color = MILITARY_COLORS['green_safe']
    elif overall_score >= 90:
        grade = 'A'
        grade_color = MILITARY_COLORS['green_safe']
    elif overall_score >= 85:
        grade = 'B+'
        grade_color = MILITARY_COLORS['blue_info']
    elif overall_score >= 80:
        grade = 'B'
        grade_color = MILITARY_COLORS['amber_caution']
    else:
        grade = 'C'
        grade_color = MILITARY_COLORS['red_danger']
    
    puf_text = f"""PUF SECURITY ASSESSMENT
Reliability: {reliability:.1f}%
Uniqueness: {uniqueness:.1f}%
Attack Resistance: {attack_resistance:.1f}%

OVERALL GRADE: {grade}
"""
    
    fig.text(0.01, 0.65, puf_text, fontsize=10, fontweight='bold',
            color=MILITARY_COLORS['white_text'], ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=MILITARY_COLORS['bg_medium'], 
                     edgecolor=grade_color, linewidth=2, alpha=0.9))
    
    # Add certification status
    cert_y = 0.35
    if overall_score >= 90:
        cert_status = "✓ CERTIFIED FOR DEPLOYMENT"
        cert_color = MILITARY_COLORS['green_safe']
    elif overall_score >= 80:
        cert_status = "⚠ CONDITIONAL APPROVAL"
        cert_color = MILITARY_COLORS['amber_caution']
    else:
        cert_status = "✗ NOT APPROVED"
        cert_color = MILITARY_COLORS['red_danger']
    
    fig.text(0.01, cert_y, cert_status, fontsize=14, fontweight='bold',
            color=cert_color, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=MILITARY_COLORS['bg_dark'], 
                     edgecolor=cert_color, linewidth=2, alpha=0.9))


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
        
        # Create basic dashboard
        fig1 = create_defense_dashboard(mission_data, threat_level, env_status,
                                       attack_prob, countermeasure_eff,
                                       output_dir=args.output_dir,
                                       save_format=args.format,
                                       dpi=args.dpi)
        
        print("✅ Basic defense dashboard created successfully!")
        plt.close(fig1)
        
        # Create enhanced military compliance dashboard
        print("\n🎖️  Generating enhanced military compliance dashboard...")
        
        # Sample PUF performance data
        puf_performance = {
            'reliability': np.random.uniform(85, 98),
            'uniqueness': np.random.uniform(48, 51),
            'attack_resistance': np.random.uniform(80, 95)
        }
        
        # Sample attack assessment
        attack_assessment = {
            'ml_attacks': np.random.uniform(0.1, 0.3),
            'side_channel': np.random.uniform(0.05, 0.2),
            'physical_attacks': np.random.uniform(0.1, 0.4)
        }
        
        # Sample mission profile
        mission_profiles = [
            {
                'mission_type': 'satellite',
                'security_clearance': 'SECRET',
                'deployment_environment': 'Space'
            },
            {
                'mission_type': 'naval',
                'security_clearance': 'TOP_SECRET',
                'deployment_environment': 'Submarine'
            },
            {
                'mission_type': 'field',
                'security_clearance': 'CONFIDENTIAL',
                'deployment_environment': 'Battlefield'
            }
        ]
        
        selected_profile = np.random.choice(mission_profiles)
        
        # Create military compliance dashboard
        fig2 = create_military_compliance_dashboard(
            puf_performance, env_status, attack_assessment, selected_profile,
            output_dir=args.output_dir, save_format=args.format, dpi=args.dpi
        )
        
        print("✅ Military compliance dashboard created successfully!")
        print(f"🎯 Threat Level: {threat_level:.1f}%")
        print(f"🛡️  PUF Performance:")
        for metric, value in puf_performance.items():
            print(f"   - {metric.replace('_', ' ').title()}: {value:.1f}")
        print(f"🚀 Mission Profile:")
        for key, value in selected_profile.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print(f"🌡️  Environmental Status:")
        for factor, value in env_status.items():
            print(f"   - {factor.title()}: {value:.1f}")
        print(f"📁 Dashboards saved to: {args.output_dir}")
        
        plt.close(fig2)  # Close to save memory
        
    except Exception as e:
        print(f"❌ Error creating defense dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def generate_military_compliance_report(
    compliance_status: Dict[str, str],
    puf_metrics: Dict[str, float],
    environmental_data: Dict[str, float],
    mission_profile: Dict[str, str],
    output_dir: str = 'figures/dashboard/'
) -> str:
    """
    Generate comprehensive military compliance assessment report.
    
    Parameters
    ----------
    compliance_status : Dict[str, str]
        Compliance status for each standard
    puf_metrics : Dict[str, float]
        PUF performance metrics
    environmental_data : Dict[str, float]
        Environmental test conditions
    mission_profile : Dict[str, str]
        Mission context and requirements
    output_dir : str, optional
        Output directory for report
        
    Returns
    -------
    str
        Path to generated compliance report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('MILITARY COMPLIANCE ASSESSMENT REPORT', 
                 fontsize=24, fontweight='bold', color=MILITARY_COLORS['white_text'])
    fig.patch.set_facecolor(MILITARY_COLORS['bg_dark'])
    
    # 1. Compliance Summary Matrix
    ax1 = axes[0, 0]
    ax1.set_facecolor(MILITARY_COLORS['bg_medium'])
    
    standards = list(compliance_status.keys())
    statuses = list(compliance_status.values())
    
    # Create compliance matrix visualization
    compliance_matrix = []
    status_colors = []
    for status in statuses:
        if 'COMPLIANT' in status or 'CERTIFIED' in status or 'AUTHORIZED' in status or 'LEVEL' in status:
            compliance_matrix.append(1)
            status_colors.append(MILITARY_COLORS['green_safe'])
        elif 'MARGINAL' in status or 'LEVEL 1' in status:
            compliance_matrix.append(0.5)
            status_colors.append(MILITARY_COLORS['amber_caution'])
        else:
            compliance_matrix.append(0)
            status_colors.append(MILITARY_COLORS['red_danger'])
    
    y_pos = np.arange(len(standards))
    bars = ax1.barh(y_pos, compliance_matrix, color=status_colors, alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(standards, color=MILITARY_COLORS['white_text'], fontsize=11)
    ax1.set_xlabel('Compliance Level', color=MILITARY_COLORS['white_text'], fontsize=12)
    ax1.set_title('COMPLIANCE STANDARDS SUMMARY', 
                  color=MILITARY_COLORS['white_text'], fontsize=14, fontweight='bold')
    
    # Add status labels on bars
    for bar, status in zip(bars, statuses):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                status, va='center', ha='left', color=MILITARY_COLORS['white_text'],
                fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, 1.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Environmental Limits vs Actual
    ax2 = axes[0, 1]
    ax2.set_facecolor(MILITARY_COLORS['bg_medium'])
    
    env_factors = ['temperature', 'radiation', 'emi']
    actual_values = [environmental_data.get(factor, 0) for factor in env_factors]
    limit_values = [MILITARY_COMPLIANCE['environmental_limits'][factor]['max'] 
                   for factor in env_factors]
    
    x = np.arange(len(env_factors))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, actual_values, width, label='Actual', 
                   color=MILITARY_COLORS['blue_info'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, limit_values, width, label='Limit', 
                   color=MILITARY_COLORS['amber_caution'], alpha=0.8)
    
    ax2.set_xlabel('Environmental Factors', color=MILITARY_COLORS['white_text'])
    ax2.set_ylabel('Values', color=MILITARY_COLORS['white_text'])
    ax2.set_title('ENVIRONMENTAL COMPLIANCE LIMITS', 
                  color=MILITARY_COLORS['white_text'], fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.title() for f in env_factors], color=MILITARY_COLORS['white_text'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', 
                color=MILITARY_COLORS['white_text'])
    
    # 3. Security Classification Matrix
    ax3 = axes[1, 0]
    ax3.set_facecolor(MILITARY_COLORS['bg_medium'])
    
    security_levels = ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET']
    current_clearance = mission_profile.get('security_clearance', 'CONFIDENTIAL')
    attack_resistance = puf_metrics.get('attack_resistance', 85)
    
    # Create security clearance visualization
    clearance_requirements = [75, 85, 90, 95]  # Min attack resistance for each level
    
    colors = [MILITARY_COMPLIANCE['security_levels'][level]['color'] 
             for level in security_levels]
    
    bars = ax3.bar(security_levels, clearance_requirements, color=colors, alpha=0.7)
    ax3.axhline(y=attack_resistance, color=MILITARY_COLORS['white_text'], 
               linestyle='--', linewidth=3, label=f'Current: {attack_resistance:.1f}%')
    
    ax3.set_ylabel('Min. Attack Resistance (%)', color=MILITARY_COLORS['white_text'])
    ax3.set_title('SECURITY CLEARANCE REQUIREMENTS', 
                  color=MILITARY_COLORS['white_text'], fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight current clearance
    for i, level in enumerate(security_levels):
        if level == current_clearance:
            bars[i].set_edgecolor(MILITARY_COLORS['white_text'])
            bars[i].set_linewidth(4)
    
    # 4. Mission Profile Summary
    ax4 = axes[1, 1]
    ax4.set_facecolor(MILITARY_COLORS['bg_medium'])
    ax4.axis('off')
    
    # Create mission profile text summary
    profile_text = "MISSION PROFILE SUMMARY\n\n"
    for key, value in mission_profile.items():
        profile_text += f"{key.replace('_', ' ').title()}: {value}\n"
    
    profile_text += f"\nPUF PERFORMANCE METRICS\n"
    for key, value in puf_metrics.items():
        if isinstance(value, (int, float)):
            profile_text += f"{key.replace('_', ' ').title()}: {value:.2f}%\n"
        else:
            profile_text += f"{key.replace('_', ' ').title()}: {value}\n"
    
    ax4.text(0.05, 0.95, profile_text, transform=ax4.transAxes, va='top', ha='left',
            color=MILITARY_COLORS['white_text'], fontsize=12, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1", facecolor=MILITARY_COLORS['bg_light'], alpha=0.8))
    
    # 5. Compliance Timeline
    ax5 = axes[2, 0]
    ax5.set_facecolor(MILITARY_COLORS['bg_medium'])
    
    # Simulate compliance testing timeline
    test_phases = ['Design Review', 'Environmental Test', 'Security Eval', 'Field Trial', 'Certification']
    completion = [100, 100, 85, 60, 30]  # Percentage completion
    
    colors = [MILITARY_COLORS['green_safe'] if c == 100 
             else MILITARY_COLORS['amber_caution'] if c > 50 
             else MILITARY_COLORS['red_danger'] for c in completion]
    
    bars = ax5.barh(test_phases, completion, color=colors, alpha=0.8)
    ax5.set_xlabel('Completion (%)', color=MILITARY_COLORS['white_text'])
    ax5.set_title('COMPLIANCE TESTING TIMELINE', 
                  color=MILITARY_COLORS['white_text'], fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add completion percentages
    for bar, pct in zip(bars, completion):
        ax5.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{pct}%', va='center', ha='left', 
                color=MILITARY_COLORS['white_text'], fontweight='bold')
    
    # 6. Risk Assessment Matrix
    ax6 = axes[2, 1]
    ax6.set_facecolor(MILITARY_COLORS['bg_medium'])
    
    # Risk categories and levels
    risk_categories = ['Environmental', 'Physical Attack', 'Side Channel', 'ML Attack', 'Supply Chain']
    risk_levels = [20, 35, 25, 40, 15]  # Risk percentages
    
    # Create risk heat map
    risk_colors = [MILITARY_COLORS['green_safe'] if r < 25 
                  else MILITARY_COLORS['amber_caution'] if r < 40 
                  else MILITARY_COLORS['red_danger'] for r in risk_levels]
    
    bars = ax6.bar(risk_categories, risk_levels, color=risk_colors, alpha=0.8)
    ax6.axhline(y=30, color=MILITARY_COLORS['white_text'], linestyle='--', 
               alpha=0.7, label='Acceptable Risk (30%)')
    
    ax6.set_ylabel('Risk Level (%)', color=MILITARY_COLORS['white_text'])
    ax6.set_title('THREAT RISK ASSESSMENT', 
                  color=MILITARY_COLORS['white_text'], fontweight='bold')
    ax6.set_xticklabels(risk_categories, rotation=45, ha='right', 
                       color=MILITARY_COLORS['white_text'])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add risk values on bars
    for bar, risk in zip(bars, risk_levels):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{risk}%', ha='center', va='bottom', 
                color=MILITARY_COLORS['white_text'], fontweight='bold')
    
    plt.tight_layout()
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'military_compliance_report_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=MILITARY_COLORS['bg_dark'])
    plt.close()
    
    return output_path


if __name__ == "__main__":
    exit(main())