import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from matplotlib.gridspec import GridSpec

# === Load and prepare satellite-based weather data ===
file_path = "csv_-34.9599769_138.6414601_fixed_22.5_270_PT5M.csv"
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'])
df.set_index('period_end', inplace=True)

# === Filter for the full year 2024 ===
df = df[df.index.year == 2024]

# === Ensure required columns are present ===
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        print(f"Warning: Column '{col}' is missing. Filling with zeros.")
        df[col] = 0

# === System configuration ===
panel_power_max = 185  # W
system_capacity_kw = 10.175
system_capacity_w = system_capacity_kw * 1000
num_panels = int(system_capacity_w / panel_power_max)
inverter_efficiency = 0.97
temp_coeff = -0.0034
stc_irradiance = 1000  # W/m²

# === Define tilt and azimuth ranges for optimization ===
tilt_angles = np.arange(0, 91, 1)  # 0 to 90 degrees
azimuth_angles = np.arange(0, 361, 2)  # 0 to 360 degrees

# === Initialize dictionaries to store results ===
monthly_results_pvlib = {}
monthly_results_osmmeps = {}
months = range(1, 13)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# === FIXED: Improved angle of incidence calculation with numerical stability ===
def calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad):
    """
    Calculate angle of incidence with numerical stability
    """
    # Calculate the cosine of the angle of incidence
    cos_aoi = (np.cos(zenith_rad) * np.cos(tilt_rad) +
               np.sin(zenith_rad) * np.sin(tilt_rad) *
               np.cos(azimuth_rad - surface_azimuth_rad))
    
    # Ensure numerical stability by clipping to valid range [-1, 1]
    cos_aoi = np.clip(cos_aoi, -1.0, 1.0)
    
    # Calculate angle of incidence in degrees
    aoi = np.degrees(np.arccos(cos_aoi))
    aoi = np.clip(aoi, 0, 90)
    
    return aoi

# === PVLib power calculation function ===
def calculate_power_pvlib(tilt, azimuth, data_subset):
    solar_position = pvlib.solarposition.get_solarposition(
        data_subset.index, -34.9599769, 138.6414601)
    
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=data_subset['dni'],
        ghi=data_subset['ghi'],
        dhi=data_subset['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    
    poa_irradiance = poa['poa_global']
    temp_cell = pvlib.temperature.sapm_cell(
        poa_irradiance, data_subset['air_temp'], data_subset['wind_speed_10m'], -3.47, -0.0594, 3
    )
    
    dc_power = (poa_irradiance / stc_irradiance) * num_panels * panel_power_max * (1 + temp_coeff * (temp_cell - 25))
    ac_power = dc_power * inverter_efficiency
    energy_kwh = (ac_power / 1000) * (5 / 60)  # 5-minute intervals to kWh
    
    return energy_kwh.sum()

# === FIXED: OSM-MEPS power calculation function with improved AOI ===
def calculate_power_osmmeps(tilt, azimuth, data_subset):
    tilt_rad = np.radians(tilt)
    azimuth_panel_rad = np.radians(azimuth)
    
    zenith_rad = np.radians(data_subset['zenith'])
    azimuth_rad = np.radians(data_subset['azimuth'])
    
    # Use the improved AOI calculation
    aoi = calculate_aoi(tilt_rad, azimuth_rad, azimuth_panel_rad, zenith_rad)
    cos_aoi = np.cos(np.radians(aoi))
    cos_aoi[cos_aoi < 0] = 0
    
    poa_direct = data_subset['dni'] * cos_aoi * (1 - data_subset['cloud_opacity'] / 100)
    poa_direct = poa_direct.clip(lower=0)
    poa_diffuse = data_subset['dhi'] * (1 + np.cos(tilt_rad)) / 2
    poa_sky_diffuse = data_subset['ghi'] * data_subset['albedo'] * (1 - np.cos(tilt_rad)) / 2
    poa_total = poa_direct + poa_diffuse + poa_sky_diffuse
    
    nominal_operating_cell_temp = 45
    module_temp = nominal_operating_cell_temp + poa_total / 1000 * (28 - data_subset['air_temp'])
    
    dc_power = panel_power_max * (1 + temp_coeff * (module_temp - nominal_operating_cell_temp))
    dc_power *= poa_total / stc_irradiance
    dc_power *= (1 - 0.002 * data_subset['relative_humidity'])
    
    ac_power = dc_power * inverter_efficiency
    scaled_power = ac_power * num_panels
    actual_power = scaled_power * (1 - 0.01)
    
    energy_kwh = actual_power * (5 / 60) / 1000
    return energy_kwh.sum()

# === Monthly optimization for both models ===
print("=== MONTHLY OPTIMIZATION ===")
for month in months:
    print(f"Processing month {month}...")
    month_data = df[df.index.month == month]
    
    if len(month_data) == 0:
        continue
    
    # PVLib optimization
    max_energy_pvlib = -np.inf
    optimal_tilt_pvlib, optimal_azimuth_pvlib = None, None
    
    # OSM-MEPS optimization
    max_energy_osmmeps = -np.inf
    optimal_tilt_osmmeps, optimal_azimuth_osmmeps = None, None
    
    for tilt in tilt_angles:
        for azimuth in azimuth_angles:
            # PVLib calculation
            energy_pvlib = calculate_power_pvlib(tilt, azimuth, month_data)
            if energy_pvlib > max_energy_pvlib:
                max_energy_pvlib = energy_pvlib
                optimal_tilt_pvlib, optimal_azimuth_pvlib = tilt, azimuth
            
            # OSM-MEPS calculation
            energy_osmmeps = calculate_power_osmmeps(tilt, azimuth, month_data)
            if energy_osmmeps > max_energy_osmmeps:
                max_energy_osmmeps = energy_osmmeps
                optimal_tilt_osmmeps, optimal_azimuth_osmmeps = tilt, azimuth
    
    # Store results
    monthly_results_pvlib[month] = {
        'optimal_tilt': optimal_tilt_pvlib,
        'optimal_azimuth': optimal_azimuth_pvlib,
        'max_energy': max_energy_pvlib
    }
    
    monthly_results_osmmeps[month] = {
        'optimal_tilt': optimal_tilt_osmmeps,
        'optimal_azimuth': optimal_azimuth_osmmeps,
        'max_energy': max_energy_osmmeps
    }
    
    print(f"Month {month}: PVLib - Tilt: {optimal_tilt_pvlib}°, Azimuth: {optimal_azimuth_pvlib}°, Energy: {max_energy_pvlib:.2f} kWh")
    print(f"Month {month}: OSM-MEPS - Tilt: {optimal_tilt_osmmeps}°, Azimuth: {optimal_azimuth_osmmeps}°, Energy: {max_energy_osmmeps:.2f} kWh")

# === Find optimal fixed configuration for entire year ===
print("\n=== FINDING OPTIMAL FIXED CONFIGURATION ===")

# PVLib fixed optimization
max_energy_fixed_pvlib = -np.inf
optimal_fixed_tilt_pvlib, optimal_fixed_azimuth_pvlib = None, None

# OSM-MEPS fixed optimization
max_energy_fixed_osmmeps = -np.inf
optimal_fixed_tilt_osmmeps, optimal_fixed_azimuth_osmmeps = None, None

for tilt in tilt_angles:
    for azimuth in azimuth_angles:
        # PVLib annual calculation
        energy_pvlib = calculate_power_pvlib(tilt, azimuth, df)
        if energy_pvlib > max_energy_fixed_pvlib:
            max_energy_fixed_pvlib = energy_pvlib
            optimal_fixed_tilt_pvlib, optimal_fixed_azimuth_pvlib = tilt, azimuth
        
        # OSM-MEPS annual calculation
        energy_osmmeps = calculate_power_osmmeps(tilt, azimuth, df)
        if energy_osmmeps > max_energy_fixed_osmmeps:
            max_energy_fixed_osmmeps = energy_osmmeps
            optimal_fixed_tilt_osmmeps, optimal_fixed_azimuth_osmmeps = tilt, azimuth

print(f"PVLib Fixed Optimal - Tilt: {optimal_fixed_tilt_pvlib}°, Azimuth: {optimal_fixed_azimuth_pvlib}°, Annual Energy: {max_energy_fixed_pvlib:.2f} kWh")
print(f"OSM-MEPS Fixed Optimal - Tilt: {optimal_fixed_tilt_osmmeps}°, Azimuth: {optimal_fixed_azimuth_osmmeps}°, Annual Energy: {max_energy_fixed_osmmeps:.2f} kWh")

# === Calculate monthly energy with fixed configurations ===
monthly_fixed_energy_pvlib = {}
monthly_fixed_energy_osmmeps = {}

for month in months:
    if month in monthly_results_pvlib:
        month_data = df[df.index.month == month]
        
        monthly_fixed_energy_pvlib[month] = calculate_power_pvlib(
            optimal_fixed_tilt_pvlib, optimal_fixed_azimuth_pvlib, month_data)
        
        monthly_fixed_energy_osmmeps[month] = calculate_power_osmmeps(
            optimal_fixed_tilt_osmmeps, optimal_fixed_azimuth_osmmeps, month_data)

# === Print comparison results ===
print("\n=== COMPARISON: MONTHLY OPTIMAL vs FIXED OPTIMAL ===")
print(f"{'Month':<8} {'Model':<10} {'Opt Tilt':<10} {'Opt Azim':<10} {'Mth Opt Energy':<15} {'Fixed Energy':<15} {'Gain':<10} {'Gain %':<8}")
print("-" * 95)

for month in months:
    if month in monthly_results_pvlib:
        # PVLib results
        mth_opt_pvlib = monthly_results_pvlib[month]['max_energy']
        fixed_pvlib = monthly_fixed_energy_pvlib[month]
        gain_pvlib = mth_opt_pvlib - fixed_pvlib
        gain_pct_pvlib = (gain_pvlib / fixed_pvlib) * 100 if fixed_pvlib > 0 else 0
        
        print(f"{month_names[month-1]:<8} {'PVLib':<10} {monthly_results_pvlib[month]['optimal_tilt']:<10} "
              f"{monthly_results_pvlib[month]['optimal_azimuth']:<10} {mth_opt_pvlib:<15.2f} {fixed_pvlib:<15.2f} "
              f"{gain_pvlib:<10.2f} {gain_pct_pvlib:<8.2f}%")
        
        # OSM-MEPS results
        mth_opt_osm = monthly_results_osmmeps[month]['max_energy']
        fixed_osm = monthly_fixed_energy_osmmeps[month]
        gain_osm = mth_opt_osm - fixed_osm
        gain_pct_osm = (gain_osm / fixed_osm) * 100 if fixed_osm > 0 else 0
        
        print(f"{month_names[month-1]:<8} {'OSM-MEPS':<10} {monthly_results_osmmeps[month]['optimal_tilt']:<10} "
              f"{monthly_results_osmmeps[month]['optimal_azimuth']:<10} {mth_opt_osm:<15.2f} {fixed_osm:<15.2f} "
              f"{gain_osm:<10.2f} {gain_pct_osm:<8.2f}%")
        print("-" * 95)

  #----------------------------------------------------------------------------------------------------------------------------------------



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def perform_statistical_analysis():
    """Perform paired t-test analysis and create comparative plots"""
    
    # Extract data for analysis
    months_available = [month for month in months if month in monthly_results_pvlib]
    month_labels = [month_names[month-1] for month in months_available]
    
    # Prepare data arrays
    monthly_opt_pvlib = [monthly_results_pvlib[month]['max_energy'] for month in months_available]
    fixed_pvlib = [monthly_fixed_energy_pvlib[month] for month in months_available]
    monthly_opt_osm = [monthly_results_osmmeps[month]['max_energy'] for month in months_available]
    fixed_osm = [monthly_fixed_energy_osmmeps[month] for month in months_available]
    
    # Convert to numpy arrays
    monthly_opt_pvlib = np.array(monthly_opt_pvlib)
    fixed_pvlib = np.array(fixed_pvlib)
    monthly_opt_osm = np.array(monthly_opt_osm)
    fixed_osm = np.array(fixed_osm)
    
    # Calculate differences
    differences_pvlib = monthly_opt_pvlib - fixed_pvlib
    differences_osm = monthly_opt_osm - fixed_osm
    
    # VERIFICATION CALCULATIONS
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    print(f"Months analyzed: {len(months_available)}")
    print(f"PVLib - Monthly optimal range: {monthly_opt_pvlib.min():.1f} to {monthly_opt_pvlib.max():.1f} kWh")
    print(f"PVLib - Fixed optimal range: {fixed_pvlib.min():.1f} to {fixed_pvlib.max():.1f} kWh")
    print(f"OSM-MEPS - Monthly optimal range: {monthly_opt_osm.min():.1f} to {monthly_opt_osm.max():.1f} kWh")
    print(f"OSM-MEPS - Fixed optimal range: {fixed_osm.min():.1f} to {fixed_osm.max():.1f} kWh")
    
    # Perform paired t-tests
    t_stat_pvlib, p_value_pvlib = stats.ttest_rel(monthly_opt_pvlib, fixed_pvlib)
    t_stat_osm, p_value_osm = stats.ttest_rel(monthly_opt_osm, fixed_osm)
    
    # Print statistical results
    print("\n" + "=" * 60)
    print("PAIRED T-TEST ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nPVLib Model:")
    print(f"  t-statistic: {t_stat_pvlib:.4f}")
    print(f"  p-value: {p_value_pvlib:.6f}")
    print(f"  Mean monthly gain: {np.mean(differences_pvlib):.2f} kWh")
    print(f"  Total annual gain: {np.sum(differences_pvlib):.2f} kWh")
    
    print(f"\nOSM-MEPS Model:")
    print(f"  t-statistic: {t_stat_osm:.4f}")
    print(f"  p-value: {p_value_osm:.6f}")
    print(f"  Mean monthly gain: {np.mean(differences_osm):.2f} kWh")
    print(f"  Total annual gain: {np.sum(differences_osm):.2f} kWh")
    
    # Statistical significance interpretation
    alpha = 0.05
    print(f"\nSTATISTICAL SIGNIFICANCE (alpha = {alpha}):")
    
    if p_value_pvlib < alpha:
        print(f"  PVLib: SIGNIFICANT (p < {alpha})")
    else:
        print(f"  PVLib: NOT SIGNIFICANT (p ≥ {alpha})")
    
    if p_value_osm < alpha:
        print(f"  OSM-MEPS: SIGNIFICANT (p < {alpha})")
    else:
        print(f"  OSM-MEPS: NOT SIGNIFICANT (p ≥ {alpha})")
    
    return monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels

def create_comparative_plots(monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels):
    """Create comparative plots for monthly vs fixed energy"""
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
    
    # Subplot 1: Monthly vs Fixed Energy Comparison - 4 BARS PER MONTH
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')
    
    x_pos = np.arange(len(month_labels))
    total_width = 0.8
    bar_width = total_width / 4  # 4 bars per month
    
    # Calculate positions for 4 bars per month
    positions = [
        x_pos - 1.5*bar_width,  # PVLib Monthly Optimal
        x_pos - 0.5*bar_width,  # PVLib Fixed Optimal  
        x_pos + 0.5*bar_width,  # OSM-MEPS Monthly Optimal
        x_pos + 1.5*bar_width   # OSM-MEPS Fixed Optimal
    ]
    
    # Plot 4 bars per month
    bars1 = ax1.bar(positions[0], monthly_opt_pvlib, bar_width,
                   color='orange', alpha=0.9, edgecolor='darkorange', linewidth=1,
                   label='PVLIB Monthly Optimal')
    
    bars2 = ax1.bar(positions[1], fixed_pvlib, bar_width,
                   color='darkorange', alpha=0.7, edgecolor='brown', linewidth=1,
                   label='PVLIB Fixed Optimal')
    
    bars3 = ax1.bar(positions[2], monthly_opt_osm, bar_width,
                   color='limegreen', alpha=0.9, edgecolor='green', linewidth=1,
                   label='OSM-MEPS Monthly Optimal')
    
    bars4 = ax1.bar(positions[3], fixed_osm, bar_width,
                   color='forestgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1,
                   label='OSM-MEPS Fixed Optimal')
    
    ax1.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_ylabel('Energy Production (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    
    # Enhanced grid for first plot - every 100 units
    y_max = max(monthly_opt_pvlib.max(), monthly_opt_osm.max())
    y_max_rounded = ((y_max // 100) + 1) * 100  
    y_ticks = np.arange(0, y_max_rounded + 300, 300)  
    
    ax1.set_yticks(y_ticks)
    ax1.set_ylim(0, y_max_rounded)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Legend for energy comparison
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, fontsize=13, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Subplot 2: Energy Gains
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#f8f9fa')
    
    # Plot energy gains - 2 bars per month
    gain_bar_width = total_width / 2
    gain_positions = [x_pos - gain_bar_width/2, x_pos + gain_bar_width/2]
    
    bars_gains_pvlib = ax2.bar(gain_positions[0], differences_pvlib, gain_bar_width,
                              color='orange', alpha=0.8, edgecolor='darkorange', linewidth=1,
                              label='PVLIB Energy Gain')
    bars_gains_osm = ax2.bar(gain_positions[1], differences_osm, gain_bar_width,
                            color='limegreen', alpha=0.8, edgecolor='green', linewidth=1,
                            label='OSM-MEPS Energy Gain')
    
    ax2.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_ylabel('Energy Gain (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(month_labels, fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    
    # Enhanced grid for second plot - every 100 units, NO NEGATIVE VALUES
    gain_max = max(np.max(differences_pvlib), np.max(differences_osm))
    gain_max_rounded = ((gain_max // 100) + 1) * 100  # Round up to nearest 100
    gain_ticks = np.arange(0, gain_max_rounded + 100, 100)  # Grid every 100 units, starting from 0
    
    ax2.set_yticks(gain_ticks)
    ax2.set_ylim(0, gain_max_rounded)  # Start from 0, no negative values
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels on gain bars
    for bars in [bars_gains_pvlib, bars_gains_osm]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label positive gains
                y_offset = gain_max_rounded * 0.01
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + y_offset,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=12, 
                        fontfamily='Garamond', fontweight='bold')
    
    # Legend for gains
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=13, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    plt.savefig("STATISTICAL_ANALYSIS_COMPARISON_Adelaide_FINAL.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()
    
    # Create summary statistics table - NO ROUNDING
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary_data = {
        'Metric': ['Total Annual Energy (Monthly Opt)', 'Total Annual Energy (Fixed Opt)', 
                   'Total Annual Gain', 'Gain Percentage'],
        'PVLib': [f"{np.sum(monthly_opt_pvlib):.2f} kWh", f"{np.sum(fixed_pvlib):.2f} kWh",
                 f"{np.sum(differences_pvlib):.2f} kWh",
                 f"{(np.sum(differences_pvlib)/np.sum(fixed_pvlib))*100:.2f}%"],
        'OSM-MEPS': [f"{np.sum(monthly_opt_osm):.2f} kWh", f"{np.sum(fixed_osm):.2f} kWh",
                    f"{np.sum(differences_osm):.2f} kWh",
                    f"{(np.sum(differences_osm)/np.sum(fixed_osm))*100:.2f}%"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

# Perform the analysis and create plots adelaide
monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels = perform_statistical_analysis()
create_comparative_plots(monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels)

#----------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# === PLOTTING CODE: Compass-Loop Oriented Tilt–Azimuth ===

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def normalize_azimuth(az):
    """Normalize to 0–360"""
    return az % 360

def compass_transform(az):
    """
    Transform azimuth so that:
    - 0°/360° (N) lies on X-axis (baseline)
    - 90° (E) is at top
    - 270° (W) is below
    - 180° (S) is below West (deepest)
    """
    az = normalize_azimuth(az)
    return np.where(az <= 180, az, az - 360)

def create_compass_loop_plot():
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.set_facecolor('#f8f9fa')

    months_available = [m for m in months if m in monthly_results_pvlib]
    month_labels = [month_names[m - 1] for m in months_available]

    # === Extract Data ===
    pvlib_tilt = [monthly_results_pvlib[m]['optimal_tilt'] for m in months_available]
    osm_tilt = [monthly_results_osmmeps[m]['optimal_tilt'] for m in months_available]
    pvlib_az = [compass_transform(monthly_results_pvlib[m]['optimal_azimuth']) for m in months_available]
    osm_az = [compass_transform(monthly_results_osmmeps[m]['optimal_azimuth']) for m in months_available]

    # === Twin Axis for Tilt ===
    ax2 = ax1.twinx()

    # === Colors ===
    color_pvlib_tilt = 'orange'
    color_osm_tilt = 'green'
    color_pvlib_azimuth = 'maroon'
    color_osm_azimuth = 'cyan'

    # === Plot AZIMUTH (Compass Loop Style) ===
    ax1.plot(month_labels, pvlib_az, 'o--', color=color_pvlib_azimuth,
             markerfacecolor=color_pvlib_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, label='PVLIB Optimal Azimuth')

    ax1.plot(month_labels, osm_az, 's--', color=color_osm_azimuth,
             markerfacecolor=color_osm_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, label='OSM-MEPS Optimal Azimuth')

    # === Plot TILT ===
    ax2.plot(month_labels, pvlib_tilt, 'o-', color=color_pvlib_tilt,
             markerfacecolor=color_pvlib_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, label='PVLIB Optimal Tilt')

    ax2.plot(month_labels, osm_tilt, 's-', color=color_osm_tilt,
             markerfacecolor=color_osm_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, label='OSM-MEPS Optimal Tilt')

    # === Axis Setup ===
    ax1.set_ylim(-270, 90)  # E at top, N baseline, W below, S below W
    ax1.axhline(0, color='black', linewidth=1.3)  # North line reference
    ax1.set_ylabel('Optimal Azimuth Angle (°)', fontsize=20, fontweight='bold')

    # Azimuth ticks every 30°, labeled with compass points
    az_ticks = np.arange(-270, 100, 30)
    az_labels = []
    for t in az_ticks:
        deg = t % 360
        if deg == 0:
            label = 'N (0°)'
        elif deg == 90:
            label = 'E (90°)'
        elif deg == 180:
            label = 'S (180°)'
        elif deg == 270:
            label = 'W (270°)'
        else:
            label = f'{deg:.0f}°'
        az_labels.append(label)

    ax1.set_yticks(az_ticks)
    ax1.set_yticklabels(az_labels)

    ax1.set_xlabel('Month', fontsize=20, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=18)

    # Dense gridlines for better readability
    ax1.grid(True, which='major', axis='both', linestyle='-', alpha=0.4, linewidth=0.7)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='y', linestyle='--', alpha=0.30, linewidth=0.4)

    # === Tilt Axis ===
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Optimal Tilt Angle (°)', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=18)

    # === Annotate points with dynamic positioning ===
    
    # Calculate annotation offsets based on data values and axis ranges
    az_range = 360  # -270 to 90 = 360 range
    tilt_range = 90  # 0 to 90 = 90 range
    
    # Annotate Azimuth points (ax1)
    for i, month in enumerate(month_labels):
        # Determine annotation direction based on position in axis
        pvlib_az_val = pvlib_az[i]
        osm_az_val = osm_az[i]
        
        # For PVLIB azimuth - position annotations to avoid overlap
        if pvlib_az_val > -50:  # If in upper half, annotate downward
            pvlib_az_offset = -5
            pvlib_va = 'top'
        else:  # If in lower half, annotate upward
            pvlib_az_offset = 15
            pvlib_va = 'bottom'
            
        # For OSM azimuth - opposite direction from PVLIB
        if osm_az_val > -50:
            osm_az_offset = 15
            osm_va = 'bottom'
        else:
            osm_az_offset = -15
            osm_va = 'top'
        
        # Get original azimuth values for display (not transformed)
        pvlib_az_orig = normalize_azimuth(monthly_results_pvlib[months_available[i]]['optimal_azimuth'])
        osm_az_orig = normalize_azimuth(monthly_results_osmmeps[months_available[i]]['optimal_azimuth'])
        
        ax1.text(month, pvlib_az_val + pvlib_az_offset, f'{pvlib_az_orig:.0f}°', 
                 color=color_pvlib_azimuth, fontsize=12, ha='center', va=pvlib_va,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color_pvlib_azimuth))
        
        ax1.text(month, osm_az_val + osm_az_offset, f'{osm_az_orig:.0f}°', 
                 color=color_osm_azimuth, fontsize=12, ha='center', va=osm_va,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color_osm_azimuth))

    # Annotate Tilt points (ax2)
    for i, month in enumerate(month_labels):
        pvlib_tilt_val = pvlib_tilt[i]
        osm_tilt_val = osm_tilt[i]
        
        # For tilt annotations, position based on relative values
        if pvlib_tilt_val > osm_tilt_val:
            pvlib_tilt_offset = 3
            osm_tilt_offset = -3
            pvlib_va = 'bottom'
            osm_va = 'top'
        else:
            pvlib_tilt_offset = -3
            osm_tilt_offset = 3
            pvlib_va = 'top'
            osm_va = 'bottom'
        
        ax2.text(month, pvlib_tilt_val + pvlib_tilt_offset, f'{pvlib_tilt_val:.0f}°', 
                 color=color_pvlib_tilt, fontsize=12, ha='center', va=pvlib_va,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color_pvlib_tilt))
        
        ax2.text(month, osm_tilt_val + osm_tilt_offset, f'{osm_tilt_val:.0f}°', 
                 color=color_osm_tilt, fontsize=12, ha='center', va=osm_va,
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor=color_osm_tilt))

    # === Legend ===
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.13),
               ncol=4, fontsize=16, framealpha=0.9,
               facecolor='white', edgecolor='gray')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)
    plt.savefig("ADELAIDE_COMPASS_LOOP_TILT_AZIMUTH_v2.pdf", format="pdf",
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# === Run final improved compass layout ===
create_compass_loop_plot()

#---------------------------------------------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# === PLOTTING CODE: Compass-Loop Oriented Tilt–Azimuth ===

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 20  # Increased by 3 steps (from 17)
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#fafafa'  # Changed to #fafafa
plt.rcParams['figure.facecolor'] = '#fafafa'  # Changed to #fafafa
plt.rcParams['font.weight'] = 'bold'  # Make all text bold

def normalize_azimuth(az):
    """Normalize to 0–360"""
    return az % 360

def compass_transform(az):
    """
    Transform azimuth so that:
    - 0°/360° (N) lies on X-axis (baseline)
    - 90° (E) is at top
    - 270° (W) is below
    - 180° (S) is below West (deepest)
    """
    az = normalize_azimuth(az)
    return np.where(az <= 180, az, az - 360)

def create_compass_loop_plot():
    fig, ax1 = plt.subplots(figsize=(10, 7))  # Changed to length 10, height 6
    ax1.set_facecolor('#fafafa')

    months_available = [m for m in months if m in monthly_results_pvlib]
    month_labels = [month_names[m - 1] for m in months_available]

    # === Extract Data ===
    pvlib_tilt = [monthly_results_pvlib[m]['optimal_tilt'] for m in months_available]
    osm_tilt = [monthly_results_osmmeps[m]['optimal_tilt'] for m in months_available]
    pvlib_az = [compass_transform(monthly_results_pvlib[m]['optimal_azimuth']) for m in months_available]
    osm_az = [compass_transform(monthly_results_osmmeps[m]['optimal_azimuth']) for m in months_available]

    # === Twin Axis for Tilt ===
    ax2 = ax1.twinx()
    ax2.set_facecolor('#fafafa')

    # === Colors ===
    color_pvlib_tilt = 'orange'
    color_osm_tilt = 'green'
    color_pvlib_azimuth = 'maroon'
    color_osm_azimuth = 'cyan'

    # === Plot AZIMUTH (Compass Loop Style) ===
    ax1.plot(month_labels, pvlib_az, 'o--', color=color_pvlib_azimuth,
             markerfacecolor=color_pvlib_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,  # Increased marker size
             label='PVLIB Optimal Azimuth')

    ax1.plot(month_labels, osm_az, 's--', color=color_osm_azimuth,
             markerfacecolor=color_osm_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,  # Increased marker size
             label='OSM-MEPS Optimal Azimuth')

    # === Plot TILT ===
    ax2.plot(month_labels, pvlib_tilt, 'o-', color=color_pvlib_tilt,
             markerfacecolor=color_pvlib_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,  # Increased marker size
             label='PVLIB Optimal Tilt')

    ax2.plot(month_labels, osm_tilt, 's-', color=color_osm_tilt,
             markerfacecolor=color_osm_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,  # Increased marker size
             label='OSM-MEPS Optimal Tilt')

    # === Axis Setup ===
    ax1.set_ylim(-270, 90)  # E at top, N baseline, W below, S below W
    ax1.axhline(0, color='black', linewidth=1.3)  # North line reference
    ax1.set_ylabel('Optimal Azimuth Angle (°)', fontsize=23, fontfamily='Garamond', fontweight='bold')  # Increased font size

    # Azimuth ticks every 30°, labeled with compass points
    az_ticks = np.arange(-270, 100, 30)
    az_labels = []
    for t in az_ticks:
        deg = t % 360
        if deg == 0:
            label = 'N\n0°'
        elif deg == 90:
            label = 'E\n90°'
        elif deg == 180:
            label = 'S\n180°'
        elif deg == 270:
            label = 'W\n270°'
        else:
            label = f'{deg:.0f}°'
        az_labels.append(label)

    ax1.set_yticks(az_ticks)
    ax1.set_yticklabels(az_labels)
    ax1.tick_params(axis='y', labelsize=17)  # Increased font size

    ax1.set_xlabel('Month', fontsize=23, fontfamily='Garamond', fontweight='bold')  # Increased font size
    ax1.tick_params(axis='x', labelsize=21)  # Increased font size

    # Dense gridlines for better readability
    ax1.grid(True, which='major', axis='both', linestyle='-', alpha=0.4, linewidth=0.7)
    ax1.minorticks_on()
    ax1.grid(True, which='minor', axis='y', linestyle='--', alpha=0.30, linewidth=0.4)

    # === Tilt Axis ===
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Optimal Tilt Angle (°)', fontsize=23, fontfamily='Garamond', fontweight='bold')  # Increased font size
    ax2.tick_params(axis='y', labelsize=21)  # Increased font size

    # === Annotate points with dynamic positioning ===
    
    # Calculate annotation offsets based on data values and axis ranges
    az_range = 360  # -270 to 90 = 360 range
    tilt_range = 90  # 0 to 90 = 90 range
    
    # Annotate Azimuth points (ax1)
    for i, month in enumerate(month_labels):
        # Determine annotation direction based on position in axis
        pvlib_az_val = pvlib_az[i]
        osm_az_val = osm_az[i]
        
        # For PVLIB azimuth - position annotations to avoid overlap
        if pvlib_az_val > -50:  # If in upper half, annotate downward
            pvlib_az_offset = -0.2
            pvlib_va = 'top'
        else:  # If in lower half, annotate upward
            pvlib_az_offset = 15
            pvlib_va = 'bottom'
            
        # For OSM azimuth - opposite direction from PVLIB
        if osm_az_val > -50:
            osm_az_offset = 15
            osm_va = 'bottom'
        else:
            osm_az_offset = -15
            osm_va = 'top'
        
        # Get original azimuth values for display (not transformed)
        pvlib_az_orig = normalize_azimuth(monthly_results_pvlib[months_available[i]]['optimal_azimuth'])
        osm_az_orig = normalize_azimuth(monthly_results_osmmeps[months_available[i]]['optimal_azimuth'])
        
        ax1.text(month, pvlib_az_val + pvlib_az_offset, f'{pvlib_az_orig:.0f}°', 
                 color=color_pvlib_azimuth, fontsize=15, ha='center', va=pvlib_va, fontweight='bold',  # Increased font size and bold
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='#fafafa', alpha=0.9, edgecolor=color_pvlib_azimuth, linewidth=1.5))  # Changed to #fafafa
        
        ax1.text(month, osm_az_val + osm_az_offset, f'{osm_az_orig:.0f}°', 
                 color=color_osm_azimuth, fontsize=15, ha='center', va=osm_va, fontweight='bold',  # Increased font size and bold
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='#fafafa', alpha=0.9, edgecolor=color_osm_azimuth, linewidth=1.5))  # Changed to #fafafa

    # Annotate Tilt points (ax2)
    for i, month in enumerate(month_labels):
        pvlib_tilt_val = pvlib_tilt[i]
        osm_tilt_val = osm_tilt[i]
        
        # For tilt annotations, position based on relative values
        if pvlib_tilt_val > osm_tilt_val:
            pvlib_tilt_offset = 3
            osm_tilt_offset = -3
            pvlib_va = 'bottom'
            osm_va = 'top'
        else:
            pvlib_tilt_offset = -3
            osm_tilt_offset = 3
            pvlib_va = 'top'
            osm_va = 'bottom'
        
        ax2.text(month, pvlib_tilt_val + pvlib_tilt_offset, f'{pvlib_tilt_val:.0f}°', 
                 color=color_pvlib_tilt, fontsize=15, ha='center', va=pvlib_va, fontweight='bold',  # Increased font size and bold
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='#fafafa', alpha=0.9, edgecolor=color_pvlib_tilt, linewidth=1.5))  # Changed to #fafafa
        
        ax2.text(month, osm_tilt_val + osm_tilt_offset, f'{osm_tilt_val:.0f}°', 
                 color=color_osm_tilt, fontsize=15, ha='center', va=osm_va, fontweight='bold',  # Increased font size and bold
                 bbox=dict(boxstyle="round,pad=0.1", facecolor='#fafafa', alpha=0.9, edgecolor=color_osm_tilt, linewidth=1.5))  # Changed to #fafafa

    # === Legend ===
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.13),
               ncol=2, fontsize=16, framealpha=0.9,  # Increased font size
               facecolor='white', edgecolor='gray')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)
    plt.savefig("ADELAIDE_COMPASS_LOOP_TILT_AZIMUTH_v2_PETER.pdf", format="pdf",
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# === Run final improved compass layout ===
create_compass_loop_plot()

#----------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def perform_statistical_analysis():
    """Perform paired t-test analysis and create comparative plots"""
    
    # Extract data for analysis
    months_available = [month for month in months if month in monthly_results_pvlib]
    month_labels = [month_names[month-1] for month in months_available]
    
    # Prepare data arrays
    monthly_opt_pvlib = [monthly_results_pvlib[month]['max_energy'] for month in months_available]
    fixed_pvlib = [monthly_fixed_energy_pvlib[month] for month in months_available]
    monthly_opt_osm = [monthly_results_osmmeps[month]['max_energy'] for month in months_available]
    fixed_osm = [monthly_fixed_energy_osmmeps[month] for month in months_available]
    
    # Convert to numpy arrays
    monthly_opt_pvlib = np.array(monthly_opt_pvlib)
    fixed_pvlib = np.array(fixed_pvlib)
    monthly_opt_osm = np.array(monthly_opt_osm)
    fixed_osm = np.array(fixed_osm)
    
    # Calculate differences
    differences_pvlib = monthly_opt_pvlib - fixed_pvlib
    differences_osm = monthly_opt_osm - fixed_osm
    
    # VERIFICATION CALCULATIONS
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    print(f"Months analyzed: {len(months_available)}")
    print(f"PVLib - Monthly optimal range: {monthly_opt_pvlib.min():.1f} to {monthly_opt_pvlib.max():.1f} kWh")
    print(f"PVLib - Fixed optimal range: {fixed_pvlib.min():.1f} to {fixed_pvlib.max():.1f} kWh")
    print(f"OSM-MEPS - Monthly optimal range: {monthly_opt_osm.min():.1f} to {monthly_opt_osm.max():.1f} kWh")
    print(f"OSM-MEPS - Fixed optimal range: {fixed_osm.min():.1f} to {fixed_osm.max():.1f} kWh")
    
    # Perform paired t-tests
    t_stat_pvlib, p_value_pvlib = stats.ttest_rel(monthly_opt_pvlib, fixed_pvlib)
    t_stat_osm, p_value_osm = stats.ttest_rel(monthly_opt_osm, fixed_osm)
    
    # Print statistical results
    print("\n" + "=" * 60)
    print("PAIRED T-TEST ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nPVLib Model:")
    print(f"  t-statistic: {t_stat_pvlib:.4f}")
    print(f"  p-value: {p_value_pvlib:.6f}")
    print(f"  Mean monthly gain: {np.mean(differences_pvlib):.2f} kWh")
    print(f"  Total annual gain: {np.sum(differences_pvlib):.2f} kWh")
    
    print(f"\nOSM-MEPS Model:")
    print(f"  t-statistic: {t_stat_osm:.4f}")
    print(f"  p-value: {p_value_osm:.6f}")
    print(f"  Mean monthly gain: {np.mean(differences_osm):.2f} kWh")
    print(f"  Total annual gain: {np.sum(differences_osm):.2f} kWh")
    
    # Statistical significance interpretation
    alpha = 0.05
    print(f"\nSTATISTICAL SIGNIFICANCE (alpha = {alpha}):")
    
    if p_value_pvlib < alpha:
        print(f"  PVLib: SIGNIFICANT (p < {alpha})")
    else:
        print(f"  PVLib: NOT SIGNIFICANT (p ≥ {alpha})")
    
    if p_value_osm < alpha:
        print(f"  OSM-MEPS: SIGNIFICANT (p < {alpha})")
    else:
        print(f"  OSM-MEPS: NOT SIGNIFICANT (p ≥ {alpha})")
    
    return monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels

def create_comparative_plots(monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels):
    """Create comparative plots for monthly vs fixed energy"""
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
    
    # Subplot 1: Monthly vs Fixed Energy Comparison - 4 BARS PER MONTH
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')
    
    x_pos = np.arange(len(month_labels))
    total_width = 0.8
    bar_width = total_width / 4  # 4 bars per month
    
    # Calculate positions for 4 bars per month
    positions = [
        x_pos - 1.5*bar_width,  # PVLib Monthly Optimal
        x_pos - 0.5*bar_width,  # PVLib Fixed Optimal  
        x_pos + 0.5*bar_width,  # OSM-MEPS Monthly Optimal
        x_pos + 1.5*bar_width   # OSM-MEPS Fixed Optimal
    ]
    
    # Plot 4 bars per month
    bars1 = ax1.bar(positions[0], monthly_opt_pvlib, bar_width,
                   color='orange', alpha=0.9, edgecolor='darkorange', linewidth=1,
                   label='PVLIB Monthly Optimal')
    
    bars2 = ax1.bar(positions[1], fixed_pvlib, bar_width,
                   color='darkorange', alpha=0.7, edgecolor='brown', linewidth=1,
                   label='PVLIB Fixed Optimal')
    
    bars3 = ax1.bar(positions[2], monthly_opt_osm, bar_width,
                   color='limegreen', alpha=0.9, edgecolor='green', linewidth=1,
                   label='OSM-MEPS Monthly Optimal')
    
    bars4 = ax1.bar(positions[3], fixed_osm, bar_width,
                   color='forestgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1,
                   label='OSM-MEPS Fixed Optimal')
    
    ax1.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_ylabel('Energy Production (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    
    # Enhanced grid for first plot - every 100 units
    y_max = max(monthly_opt_pvlib.max(), monthly_opt_osm.max())
    y_max_rounded = ((y_max // 100) + 1) * 100  
    y_ticks = np.arange(0, y_max_rounded + 300, 300)  
    
    ax1.set_yticks(y_ticks)
    ax1.set_ylim(0, y_max_rounded)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Legend for energy comparison
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, fontsize=13, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Subplot 2: Energy Gains
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#f8f9fa')
    
    # Plot energy gains - 2 bars per month
    gain_bar_width = total_width / 2
    gain_positions = [x_pos - gain_bar_width/2, x_pos + gain_bar_width/2]
    
    bars_gains_pvlib = ax2.bar(gain_positions[0], differences_pvlib, gain_bar_width,
                              color='orange', alpha=0.8, edgecolor='darkorange', linewidth=1,
                              label='PVLIB Energy Gain')
    bars_gains_osm = ax2.bar(gain_positions[1], differences_osm, gain_bar_width,
                            color='limegreen', alpha=0.8, edgecolor='green', linewidth=1,
                            label='OSM-MEPS Energy Gain')
    
    ax2.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_ylabel('Energy Gain (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(month_labels, fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    
    # Enhanced grid for second plot - every 100 units, NO NEGATIVE VALUES
    gain_max = max(np.max(differences_pvlib), np.max(differences_osm))
    gain_max_rounded = ((gain_max // 100) + 1) * 100  # Round up to nearest 100
    gain_ticks = np.arange(0, gain_max_rounded + 100, 100)  # Grid every 100 units, starting from 0
    
    ax2.set_yticks(gain_ticks)
    ax2.set_ylim(0, gain_max_rounded)  # Start from 0, no negative values
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels on gain bars
    for bars in [bars_gains_pvlib, bars_gains_osm]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only label positive gains
                y_offset = gain_max_rounded * 0.01
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + y_offset,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=12, 
                        fontfamily='Garamond', fontweight='bold')
    
    # Legend for gains
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=13, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    plt.savefig("STATISTICAL_ANALYSIS_COMPARISON_Adelaide_FINAL.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()
    
    # Create summary statistics table - NO ROUNDING
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary_data = {
        'Metric': ['Total Annual Energy (Monthly Opt)', 'Total Annual Energy (Fixed Opt)', 
                   'Total Annual Gain', 'Gain Percentage'],
        'PVLib': [f"{np.sum(monthly_opt_pvlib):.2f} kWh", f"{np.sum(fixed_pvlib):.2f} kWh",
                 f"{np.sum(differences_pvlib):.2f} kWh",
                 f"{(np.sum(differences_pvlib)/np.sum(fixed_pvlib))*100:.2f}%"],
        'OSM-MEPS': [f"{np.sum(monthly_opt_osm):.2f} kWh", f"{np.sum(fixed_osm):.2f} kWh",
                    f"{np.sum(differences_osm):.2f} kWh",
                    f"{(np.sum(differences_osm)/np.sum(fixed_osm))*100:.2f}%"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

# Perform the analysis and create plots adelaide
monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels = perform_statistical_analysis()
create_comparative_plots(monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels)

#----------------------------------------------------------------------------------------------------------------------------------------
