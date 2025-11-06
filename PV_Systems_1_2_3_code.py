import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Set global style parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Garamond'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': 'white'
})

# Load the CSV data
file_path = 'csv_-1.11665_36.92927_fixed_23_0_PT5M_2024.csv'
df = pd.read_csv(file_path)

# Convert 'period_end' to datetime and set as index
df['period_end'] = pd.to_datetime(df['period_end'])
df.set_index('period_end', inplace=True)

# Ensure all required columns exist
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth']
for col in required_columns:
    if col not in df.columns:
        print(f"Warning: Column '{col}' is missing. Using default value 0.")
        df[col] = 0

# System sizes to analyze
system_configs = [
    {'num_panels': 200, 'label': '200 Panels'},
    {'num_panels': 2000, 'label': '2000 Panels'},
    {'num_panels': 2800, 'label': '2800 Panels'}
]

# System Design Parameters
panel_power_max = 550  # Watts

# OPTIMIZED: Use much coarser grid for faster computation
tilt_angles = np.arange(0, 91, 1)    # 15° steps - sufficient for analysis
azimuth_angles = np.arange(0, 361, 2)  # 45° steps - focus on cardinal directions

def calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad):
    """Calculate angle of incidence with numerical stability"""
    cos_aoi = (np.cos(zenith_rad) * np.cos(tilt_rad) +
               np.sin(zenith_rad) * np.sin(tilt_rad) *
               np.cos(azimuth_rad - surface_azimuth_rad))
    cos_aoi = np.clip(cos_aoi, -1.0, 1.0)
    aoi = np.degrees(np.arccos(cos_aoi))
    return np.clip(aoi, 0, 90)

def calculate_energy_for_config_simple(tilt, azimuth, data_subset, num_panels):
    """Calculate energy for single configuration - simplified"""
    tilt_rad = np.radians(tilt)
    azimuth_rad = np.radians(data_subset['azimuth'])
    surface_azimuth_rad = np.radians(azimuth)
    zenith_rad = np.radians(data_subset['zenith'])
    
    # Calculate AOI
    aoi = calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad)
    
    # Simplified irradiance calculation
    poa_direct = data_subset['dni'] * np.cos(np.radians(aoi))
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = data_subset['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_sky_diffuse = data_subset['ghi'] * data_subset['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_sky_diffuse
    
    # Simplified temperature model
    module_temp = 45 + poa_total / 1000 * 20
    
    # Power output
    temp_coeff = -0.003
    panel_power = panel_power_max * (1 + temp_coeff * (module_temp - 25))
    dc_power = panel_power * poa_total / 1000
    
    # System output
    inverter_efficiency = 0.96  # Modern inverters
    ac_power = dc_power * inverter_efficiency
    system_power = ac_power * num_panels
    
    # Energy calculation (5-minute intervals)
    energy_kwh = system_power * (5 / 60) / 1000
    
    return energy_kwh.sum()

def optimize_month_simple(month_data, num_panels):
    """Simple optimization without parallel processing"""
    max_energy = -np.inf
    optimal_tilt, optimal_azimuth = None, None
    
    total_configs = len(tilt_angles) * len(azimuth_angles)
    config_count = 0
    
    for tilt in tilt_angles:
        for azimuth in azimuth_angles:
            config_count += 1
            if config_count % 10 == 0:
                print(f"  Progress: {config_count}/{total_configs}")
                
            energy = calculate_energy_for_config_simple(tilt, azimuth, month_data, num_panels)
            
            if energy > max_energy:
                max_energy = energy
                optimal_tilt, optimal_azimuth = tilt, azimuth
    
    return optimal_tilt, optimal_azimuth, max_energy

def optimize_fixed_simple(full_data, num_panels):
    """Simple optimization for fixed configuration"""
    max_energy = -np.inf
    optimal_tilt, optimal_azimuth = None, None
    
    total_configs = len(tilt_angles) * len(azimuth_angles)
    config_count = 0
    
    for tilt in tilt_angles:
        for azimuth in azimuth_angles:
            config_count += 1
            if config_count % 10 == 0:
                print(f"  Progress: {config_count}/{total_configs}")
                
            energy = calculate_energy_for_config_simple(tilt, azimuth, full_data, num_panels)
            
            if energy > max_energy:
                max_energy = energy
                optimal_tilt, optimal_azimuth = tilt, azimuth
    
    return optimal_tilt, optimal_azimuth, max_energy

# Dictionary to store results for all systems
all_systems_results = {}

# Get unique months in the dataset
df['month'] = df.index.month
months = sorted(df['month'].unique())
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"Starting analysis with simplified approach...")
print(f"Grid size: {len(tilt_angles)} tilt angles × {len(azimuth_angles)} azimuth angles = {len(tilt_angles)*len(azimuth_angles)} configurations per optimization")

# Process each system configuration
for config in system_configs:
    num_panels = config['num_panels']
    label = config['label']
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SYSTEM: {label}")
    print(f"{'='*60}")
    
    # System capacity
    system_capacity_kw = (num_panels * panel_power_max) / 1000
    print(f"System Capacity: {system_capacity_kw:.2f} kW")
    
    # Initialize dictionary to store monthly results
    monthly_results = {}
    
    # Iterate over each month
    for month in months:
        print(f"\nProcessing month {month} ({month_names[month-1]}) for {label}...")
        
        # Filter data for the current month
        month_data = df[df['month'] == month].copy()
        
        # Skip if no data for this month
        if len(month_data) == 0:
            print(f"  No data for month {month}, skipping...")
            continue
        
        # Optimize for this month using simple approach
        optimal_tilt, optimal_azimuth, max_energy = optimize_month_simple(month_data, num_panels)
        
        # Store monthly results
        monthly_results[month] = {
            'optimal_tilt': optimal_tilt,
            'optimal_azimuth': optimal_azimuth,
            'max_energy': max_energy
        }
        
        print(f"  Month {month_names[month-1]}: Tilt={optimal_tilt}°, Azimuth={optimal_azimuth}°, Energy={max_energy:.2f} kWh")
    
    # FIND OPTIMAL FIXED TILT AND AZIMUTH FOR THE ENTIRE YEAR
    print(f"\nFinding optimal fixed tilt and azimuth for {label}...")
    
    optimal_fixed_tilt, optimal_fixed_azimuth, max_energy_fixed = optimize_fixed_simple(df, num_panels)
    
    print(f"Optimal Fixed: Tilt={optimal_fixed_tilt}°, Azimuth={optimal_fixed_azimuth}°, Annual Energy={max_energy_fixed:.2f} kWh")
    
    # Calculate energy for each month using fixed optimal configuration
    monthly_fixed_energy = {}
    for month in months:
        if month in monthly_results:
            month_data = df[df['month'] == month].copy()
            fixed_energy = calculate_energy_for_config_simple(optimal_fixed_tilt, optimal_fixed_azimuth, month_data, num_panels)
            monthly_fixed_energy[month] = fixed_energy
    
    # Store results for this system
    all_systems_results[label] = {
        'monthly_results': monthly_results.copy(),
        'monthly_fixed_energy': monthly_fixed_energy.copy(),
        'optimal_fixed_tilt': optimal_fixed_tilt,
        'optimal_fixed_azimuth': optimal_fixed_azimuth,
        'max_energy_fixed': max_energy_fixed,
        'num_panels': num_panels,
        'system_capacity_kw': system_capacity_kw
    }

print(f"\n{'='*60}")
print("ALL SYSTEMS PROCESSING COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")

# Save results for plotting
with open('all_systems_results.pkl', 'wb') as f:
    pickle.dump({
        'all_systems_results': all_systems_results,
        'months': months,
        'month_names': month_names
    }, f)

print("Results saved to 'all_systems_results.pkl' for plotting")

#-------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.gridspec import GridSpec

# Load saved results
with open('all_systems_results.pkl', 'rb') as f:
    data = pickle.load(f)

all_systems_results = data['all_systems_results']
months = data['months']
month_names = data['month_names']

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

# Color scheme for different systems
system_colors = {
    '200 Panels': '#1f77b4',
    '2000 Panels': '#ff7f0e', 
    '2800 Panels': '#2ca02c'
}

def create_combined_plots():
    """Create combined plots for all three systems"""
    
    # FIGURE 1: Monthly tilt-azimuth optimal configuration (single graph)
    fig1 = plt.figure(figsize=(16, 18))
    gs1 = GridSpec(2, 1, figure=fig1, height_ratios=[1, 1], hspace=0.4)
    
    # Subplot 1: Optimal Tilt and Azimuth Angles
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.set_facecolor('#f8f9fa')
    
    # Use the first system for tilt-azimuth plot (since it's the same for all systems)
    first_system = list(all_systems_results.keys())[0]
    monthly_results = all_systems_results[first_system]['monthly_results']
    
    months_available = [month for month in months if month in monthly_results]
    month_labels = [month_names[month-1] for month in months_available]
    
    monthly_tilts = [monthly_results[month]['optimal_tilt'] for month in months_available]
    monthly_azimuths = [monthly_results[month]['optimal_azimuth'] for month in months_available]
    
    # Tilt line
    color1 = '#1f77b4'
    line1 = ax1.plot(month_labels, monthly_tilts, 'o-', color=color1, linewidth=2.5, markersize=12, 
                     markerfacecolor=color1, markeredgecolor='white', markeredgewidth=2, label='Optimal Tilt Angle')
    ax1.set_ylabel('Optimal Tilt Angle (°)', color=color1, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=17)
    ax1.tick_params(axis='x', labelsize=18, rotation=1)
    ax1.set_ylim(0, 90)
    ax1.set_xlim(-0.5, len(month_labels)-0.5)
    
    # Azimuth line on secondary axis
    ax2 = ax1.twinx()
    color2 = '#d62728'
    line2 = ax2.plot(month_labels, monthly_azimuths, 's-', color=color2, linewidth=2.5, markersize=12, 
                     markerfacecolor=color2, markeredgecolor='white', markeredgewidth=2, label='Optimal Azimuth Angle')
    ax2.set_ylabel('Optimal Azimuth Angle (°)', color=color2, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=18)
    ax2.set_ylim(0, 360)
    
    # ADD ANNOTATIONS FOR TILT-AZIMUTH POINTS
    for i, (tilt, azimuth) in enumerate(zip(monthly_tilts, monthly_azimuths)):
        # Annotate tilt values
        ax1.annotate(f'{tilt}°', (i, tilt), xytext=(0, 15), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.18", facecolor='lightblue', alpha=0.57, edgecolor='navy'))
        
        # Annotate azimuth values
        ax2.annotate(f'{azimuth}°', (i, azimuth), xytext=(0, -20), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.18", facecolor='lightcoral', alpha=0.57, edgecolor='darkred'))
    
    # Grid and styling
    ax1.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Legend below the graph
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, 
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.25),
                       ncol=2,
                       fontsize=16,
                       framealpha=0.9,
                       facecolor='white',
                       edgecolor='gray')
    legend.get_frame().set_linewidth(1.5)
    
    # Subplot 2: Energy comparison for all systems
    ax3 = fig1.add_subplot(gs1[1, 0])
    ax3.set_facecolor('#f8f9fa')
    
    x_pos_systems = np.arange(len(month_labels))
    total_width = 0.8
    width_per_system = total_width / len(all_systems_results)
    
    # Plot monthly optimal vs fixed optimal for all systems
    for i, (system_label, system_data) in enumerate(all_systems_results.items()):
        monthly_results = system_data['monthly_results']
        monthly_fixed_energy = system_data['monthly_fixed_energy']
        
        monthly_opt_energies = [monthly_results[month]['max_energy'] for month in months_available]
        fixed_opt_energies = [monthly_fixed_energy[month] for month in months_available]
        
        offset = (i - (len(all_systems_results)-1)/2) * width_per_system
        
        # Plot bars without hatchlines - pure colors with edge shading
        bars_opt = ax3.bar(x_pos_systems + offset - width_per_system/4, monthly_opt_energies, 
                          width_per_system/2, 
                          label=f'{system_label} - Monthly Optimal Tilt-azimuth Configuration', 
                          color=system_colors[system_label], 
                          alpha=0.85,
                          edgecolor='black',
                          linewidth=0.8)
        
        bars_fixed = ax3.bar(x_pos_systems + offset + width_per_system/4, fixed_opt_energies, 
                            width_per_system/2, 
                            label=f'{system_label} - Fixed Optimal Tilt-azimuth Configuration (Tilt=3°, Azimuth=192°)', 
                            color=system_colors[system_label], 
                            alpha=0.6,
                            edgecolor='black',
                            linewidth=0.8)
    
    ax3.set_xlabel('Month', fontsize=18.5, fontfamily='Garamond', fontweight='bold')
    ax3.set_ylabel('Energy Production (kWh)', fontsize=18.5, fontfamily='Garamond', fontweight='bold')
    ax3.set_xticks(x_pos_systems)
    ax3.set_xticklabels(month_labels, rotation=1, fontsize=18)
    ax3.tick_params(axis='y', labelsize=17)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_axisbelow(True)
    
    # Legend below the graph with multiple columns
    ax3.legend(loc='upper center', 
               bbox_to_anchor=(0.5, -0.2), 
               ncol=2,  # Three columns to fit neatly
               fontsize=15, 
               framealpha=0.9,
               facecolor='white',
               edgecolor='gray')
    
    # Use subplots_adjust instead of tight_layout to fix the warning
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, hspace=0.4)
    plt.savefig("COMBINED_TILT_AZIMUTH_ENERGY_COMPARISON.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
    
    # FIGURE 2: kWh energy gains for all systems
    fig2 = plt.figure(figsize=(16, 18))  # Increased size to avoid text clipping
    
    # Subplot 1: Absolute energy gains
    ax1 = fig2.add_subplot(2, 1, 1)
    ax1.set_facecolor('#f8f9fa')
    
    x_pos_gains = np.arange(len(month_labels))
    width_gains = 0.27
    
    all_differences = []
    for i, (system_label, system_data) in enumerate(all_systems_results.items()):
        monthly_results = system_data['monthly_results']
        monthly_fixed_energy = system_data['monthly_fixed_energy']
        
        monthly_opt_energies = [monthly_results[month]['max_energy'] for month in months_available]
        fixed_opt_energies = [monthly_fixed_energy[month] for month in months_available]
        differences = [opt - fix for opt, fix in zip(monthly_opt_energies, fixed_opt_energies)]
        all_differences.extend(differences)
        
        offset = (i - 1) * width_gains
        
        # Bars without hatchlines - pure colors
        bars = ax1.bar(x_pos_gains + offset, differences, width_gains, 
                      label=f'{system_label} PV System', 
                      color=system_colors[system_label], 
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=0.65)
        
        # Add value labels - ENSURING MARCH IS LABELLED LIKE OTHER MONTHS
        for j, bar in enumerate(bars):
            height = bar.get_height()
            # Remove the condition to ensure ALL values are labelled, including March
            va = 'bottom' if height >= 0 else 'top'
            y_offset = max([abs(d) for d in differences]) * 0.01  # Slightly increased offset
            ax1.text(bar.get_x() + bar.get_width()/2., 
                    height + (y_offset if height >= 0 else -y_offset),
                    f'{height:.0f}', ha='center', va=va, fontsize=9, fontweight='bold')  # Increased font size
    
    ax1.set_xlabel('Month', fontsize=18.5, fontfamily='Garamond', fontweight='bold')
    ax1.set_ylabel('Energy Gain (kWh)', fontsize=18.5, fontfamily='Garamond', fontweight='bold')
    ax1.set_xticks(x_pos_gains)
    ax1.set_xticklabels(month_labels, rotation=1, fontsize=18)
    ax1.tick_params(axis='y', labelsize=17)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Legend below the graph
    ax1.legend(loc='upper center', 
               bbox_to_anchor=(0.5, -0.2),  # Adjusted for larger figure
               ncol=3,
               fontsize=16,
               framealpha=0.9,
               facecolor='white',
               edgecolor='gray')
    
    # Use subplots_adjust instead of tight_layout to fix the warning
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15, hspace=0.3)
    plt.savefig("COMBINED_ENERGY_GAINS_COMPARISON.pdf", format="pdf", bbox_inches="tight", dpi=500)
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR ALL SYSTEMS")
    print("="*80)
    
    for system_label, system_data in all_systems_results.items():
        monthly_results = system_data['monthly_results']
        monthly_fixed_energy = system_data['monthly_fixed_energy']
        max_energy_fixed = system_data['max_energy_fixed']
        
        monthly_opt_energies = [monthly_results[month]['max_energy'] for month in months_available]
        fixed_opt_energies = [monthly_fixed_energy[month] for month in months_available]
        
        total_monthly_opt = sum(monthly_opt_energies)
        total_fixed_opt = sum(fixed_opt_energies)
        total_gain = total_monthly_opt - total_fixed_opt
        annual_gain_percent = (total_gain / total_fixed_opt) * 100 if total_fixed_opt > 0 else 0
        
        print(f"\n{system_label}:")
        print(f"  System Capacity: {system_data['system_capacity_kw']:.2f} kW")
        print(f"  Fixed Optimal Tilt: {system_data['optimal_fixed_tilt']}°")
        print(f"  Fixed Optimal Azimuth: {system_data['optimal_fixed_azimuth']}°")
        print(f"  Total Annual Energy (Monthly Opt): {total_monthly_opt:.2f} kWh")
        print(f"  Total Annual Energy (Fixed Opt): {total_fixed_opt:.2f} kWh")
        print(f"  Annual Energy Gain: {total_gain:.2f} kWh ({annual_gain_percent:.2f}%)")

# Create the combined plots
create_combined_plots()
