import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib

# === Load and prepare satellite-based weather data ===
file_path = "csv_40.886273_23.912687_fixed_23_180_PT5M.csv"
df = pd.read_csv(file_path)
df['period_end'] = pd.to_datetime(df['period_end'])
df.set_index('period_end', inplace=True)

# === Filter for 2024 ===
df = df[df.index.year == 2024]

# === Ensure required columns ===
required_columns = ['dni', 'ghi', 'dhi', 'air_temp', 'albedo', 'zenith', 'azimuth',
                    'cloud_opacity', 'relative_humidity', 'wind_speed_10m']
for col in required_columns:
    if col not in df.columns:
        print(f"Warning: Column '{col}' is missing. Filling with zeros.")
        df[col] = 0

# === System configuration ===
panel_power_max = 390  # W
system_capacity_kw = 1010.88
system_capacity_w = system_capacity_kw * 1000
num_panels = int(system_capacity_w / panel_power_max)
inverter_efficiency = 0.99
temp_coeff = -0.005
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
        data_subset.index, 40.886273, 23.912687)
    
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
    module_temp = nominal_operating_cell_temp + poa_total / 800 * (28 - data_subset['air_temp'])
    
    dc_power = panel_power_max * (1 + temp_coeff * (module_temp - nominal_operating_cell_temp))
    dc_power *= poa_total / stc_irradiance
    dc_power *= (1 - 0.002 * data_subset['relative_humidity'])
    
    ac_power = dc_power * inverter_efficiency
    scaled_power = ac_power * num_panels
    actual_power = scaled_power * (1 - 0.05)  # Post-system empirical loss factors
    
    # Apply extra derating for April to September (dust and soiling)
    april_to_sep_mask = data_subset.index.month.isin([4, 5, 6, 7, 8, 9])
    actual_power_derated = actual_power.copy()
    actual_power_derated[april_to_sep_mask] *= (1 - 0.15)
    
    energy_kwh = actual_power_derated * (5 / 60) / 1000
    return energy_kwh.sum()

# === Monthly optimization for both models ===
print("=== MONTHLY OPTIMIZATION FOR GREECE SYSTEM ===")
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
print("\n=== FINDING OPTIMAL FIXED CONFIGURATION FOR GREECE ===")

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


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------


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
    print(f"PVLib - Monthly optimal range: {monthly_opt_pvlib.min()/1000:.1f} to {monthly_opt_pvlib.max()/1000:.1f} MWh")
    print(f"PVLib - Fixed optimal range: {fixed_pvlib.min()/1000:.1f} to {fixed_pvlib.max()/1000:.1f} MWh")
    print(f"OSM-MEPS - Monthly optimal range: {monthly_opt_osm.min()/1000:.1f} to {monthly_opt_osm.max()/1000:.1f} MWh")
    print(f"OSM-MEPS - Fixed optimal range: {fixed_osm.min()/1000:.1f} to {fixed_osm.max()/1000:.1f} MWh")
    print(f"PVLib - Gain range: {differences_pvlib.min():.1f} to {differences_pvlib.max():.1f} kWh")
    print(f"OSM-MEPS - Gain range: {differences_osm.min():.1f} to {differences_osm.max():.1f} kWh")
    
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
    """Create comparative plots - first in KWh, second (gains) in kWh"""
    
    # Convert only the first plot data to MWh
    monthly_opt_pvlib_mwh = monthly_opt_pvlib 
    fixed_pvlib_mwh = fixed_pvlib 
    monthly_opt_osm_mwh = monthly_opt_osm 
    fixed_osm_mwh = fixed_osm 
    
    # Keep gains in kWh for second plot
    differences_pvlib_kwh = differences_pvlib
    differences_osm_kwh = differences_osm
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 15))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
    
    # Subplot 1: Monthly vs Fixed Energy Comparison - 4 BARS PER MONTH (MWh)
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
    
    # Plot 4 bars per month in MWh
    bars1 = ax1.bar(positions[0], monthly_opt_pvlib_mwh, bar_width,
                   color='orange', alpha=0.9, edgecolor='darkorange', linewidth=1,
                   label='PVLIB Monthly Optimal')
    
    bars2 = ax1.bar(positions[1], fixed_pvlib_mwh, bar_width,
                   color='darkorange', alpha=0.7, edgecolor='brown', linewidth=1,
                   label='PVLIB Fixed Optimal')
    
    bars3 = ax1.bar(positions[2], monthly_opt_osm_mwh, bar_width,
                   color='limegreen', alpha=0.9, edgecolor='green', linewidth=1,
                   label='OSM-MEPS Monthly Optimal')
    
    bars4 = ax1.bar(positions[3], fixed_osm_mwh, bar_width,
                   color='forestgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1,
                   label='OSM-MEPS Fixed Optimal')
    
    ax1.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_ylabel('Energy Production (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(month_labels, fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    
    # Enhanced grid for first plot - like graph paper with multiple grid lines
    y_max = max(monthly_opt_pvlib_mwh.max(), monthly_opt_osm_mwh.max())
    y_max_rounded = ((y_max // 50000) + 1.1) * 50000  # Round up to nearest 50 MWh
    
    # Major grid lines every 50 MWh (thicker, more prominent)
    major_y_ticks = np.arange(0, y_max_rounded + 50000, 50000)
    ax1.set_yticks(major_y_ticks)
    
    # Minor grid lines every 10 MWh (lighter, for easier reading)
    minor_y_ticks = np.arange(0, y_max_rounded + 10000, 10000)
    ax1.set_ylim(0, y_max_rounded)
    
    # Add both major and minor grid lines
    ax1.grid(True, alpha=0.6, linestyle='-', linewidth=0.8, which='major')  # Major grid - more visible
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, which='minor')  # Minor grid - lighter
    
    # Enable minor ticks for the grid
    ax1.minorticks_on()
    ax1.set_axisbelow(True)
    
    # Legend for energy comparison
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, fontsize=13.5, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Subplot 2: Energy Gains in kWh
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#f8f9fa')
    
    # Plot energy gains - 2 bars per month in kWh
    gain_bar_width = total_width / 2
    gain_positions = [x_pos - gain_bar_width/2, x_pos + gain_bar_width/2]
    
    bars_gains_pvlib = ax2.bar(gain_positions[0], differences_pvlib_kwh, gain_bar_width,
                              color='orange', alpha=0.8, edgecolor='darkorange', linewidth=1,
                              label='PVLIB Energy Gain')
    bars_gains_osm = ax2.bar(gain_positions[1], differences_osm_kwh, gain_bar_width,
                            color='limegreen', alpha=0.8, edgecolor='green', linewidth=1,
                            label='OSM-MEPS Energy Gain')
    
    ax2.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_ylabel('Energy Gain (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(month_labels, fontsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    
    # INCREASED GRID SIZE: Enhanced grid for second plot - every 1000 kWh units, NO NEGATIVE VALUES
    gain_max = max(np.max(differences_pvlib_kwh), np.max(differences_osm_kwh))
    gain_max_rounded = ((gain_max // 1000) + 1.4) * 1000  # Round up to nearest 1000 kWh
    gain_ticks = np.arange(0, gain_max_rounded + 1500, 1500)  # Grid every 1000 kWh, starting from 0
    
    ax2.set_yticks(gain_ticks)
    ax2.set_ylim(0, gain_max_rounded)  # Start from 0, no negative values
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels on gain bars - FORCE LABELS FOR MARCH AND SEPTEMBER
    for i, (bar_pvlib, bar_osm) in enumerate(zip(bars_gains_pvlib, bars_gains_osm)):
        height_pvlib = bar_pvlib.get_height()
        height_osm = bar_osm.get_height()
        month_name = month_labels[i]
        
        # ALWAYS label PVLib bars for March and September regardless of height
        if month_name in ['Mar', 'Sep']:
            y_offset = gain_max_rounded * 0.001
            ax2.text(bar_pvlib.get_x() + bar_pvlib.get_width()/2., 
                    height_pvlib + y_offset,
                    f'{height_pvlib:.0f}', ha='center', va='bottom', fontsize=12, 
                    fontfamily='Garamond', fontweight='bold')
        
        # Label other PVLib bars normally
        elif height_pvlib > 1:
            y_offset = gain_max_rounded * 0.001
            ax2.text(bar_pvlib.get_x() + bar_pvlib.get_width()/2., 
                    height_pvlib + y_offset,
                    f'{height_pvlib:.0f}', ha='center', va='bottom', fontsize=12, 
                    fontfamily='Garamond', fontweight='bold')
        
        # Label all OSM-MEPS bars normally
        if height_osm > 1:
            y_offset = gain_max_rounded * 0.001
            ax2.text(bar_osm.get_x() + bar_osm.get_width()/2., 
                    height_osm + y_offset,
                    f'{height_osm:.0f}', ha='center', va='bottom', fontsize=12, 
                    fontfamily='Garamond', fontweight='bold')
    
    # Legend for gains
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=13.5, framealpha=0.9,
              facecolor='white', edgecolor='gray')
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1)
    plt.savefig("STATISTICAL_ANALYSIS_COMPARISON_GREECE_final.pdf", format="pdf", 
                bbox_inches="tight", dpi=600, facecolor=fig.get_facecolor())
    plt.show()
    
    # Create summary statistics table - mixed units
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary_data = {
        'Metric': ['Total Annual Energy (Monthly Opt)', 'Total Annual Energy (Fixed Opt)', 
                   'Total Annual Gain', 'Gain Percentage'],
        'PVLib': [f"{np.sum(monthly_opt_pvlib)/1000:.2f} MWh", f"{np.sum(fixed_pvlib)/1000:.2f} MWh",
                 f"{np.sum(differences_pvlib):.0f} kWh",
                 f"{(np.sum(differences_pvlib)/np.sum(fixed_pvlib))*100:.2f}%"],
        'OSM-MEPS': [f"{np.sum(monthly_opt_osm)/1000:.2f} MWh", f"{np.sum(fixed_osm)/1000:.2f} MWh",
                    f"{np.sum(differences_osm):.0f} kWh",
                    f"{(np.sum(differences_osm)/np.sum(fixed_osm))*100:.2f}%"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

# Perform the analysis and create plots greece
monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels = perform_statistical_analysis()
create_comparative_plots(monthly_opt_pvlib, fixed_pvlib, monthly_opt_osm, fixed_osm, differences_pvlib, differences_osm, month_labels)


#-----------------------------------------------------------------------------------------------------------------------------------------------------

# === PLOTTING CODE: Monthly Tilt-Azimuth Configuration for Greece ===

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def normalize_azimuth(azimuth):
    """Normalize azimuth to 0-359 range, converting 360 to 0"""
    return azimuth % 360

def create_monthly_tilt_azimuth_plot_greece():
    """Create professional monthly tilt-azimuth plot with marker points for Greece system"""
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.35)
    
    # Subplot 1: PVLib Optimal Angles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#f8f9fa')
    
    months_available = [month for month in months if month in monthly_results_pvlib]
    month_labels = [month_names[month-1] for month in months_available]
    
    # PVLib data - normalize azimuth values (360 becomes 0)
    monthly_tilts_pvlib = [monthly_results_pvlib[month]['optimal_tilt'] for month in months_available]
    monthly_azimuths_pvlib = [normalize_azimuth(monthly_results_pvlib[month]['optimal_azimuth']) for month in months_available]
    
    # Tilt line - PVLib (Orange)
    color1 = 'orange'
    line1 = ax1.plot(month_labels, monthly_tilts_pvlib, 'o-', color=color1, linewidth=2.5, markersize=12, 
                     markerfacecolor=color1, markeredgecolor='white', markeredgewidth=2)
    ax1.set_ylabel('Optimal Tilt Angle (°)', color=color1, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_ylim(0, 90)
    ax1.set_xlim(-0.5, len(month_labels)-0.5)
    
    # Azimuth line on secondary axis - PVLib (Maroon)
    ax2 = ax1.twinx()
    color2 = 'maroon'
    line2 = ax2.plot(month_labels, monthly_azimuths_pvlib, 's-', color=color2, linewidth=2.5, markersize=10, 
                     markerfacecolor=color2, markeredgecolor='white', markeredgewidth=2)
    ax2.set_ylabel('Optimal Azimuth Angle (°)', color=color2, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=16)
    ax2.set_ylim(0, 359)  # Maximum azimuth set to 359
    # Set proper azimuth ticks (0=N, 90=E, 180=S, 270=W)
    ax2.set_yticks([0, 90, 180, 270])
    ax2.set_yticklabels(['0° (N)', '90° (E)', '180° (S)', '270° (W)'])
    
    # Add annotations for PVLib points
    for i, (tilt, azimuth) in enumerate(zip(monthly_tilts_pvlib, monthly_azimuths_pvlib)):
        ax1.annotate(f'{tilt}°', (i, tilt), xytext=(0, 12), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightblue', alpha=0.5, edgecolor='navy'))
        
        ax2.annotate(f'{azimuth}°', (i, azimuth), xytext=(0, -11), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcoral', alpha=0.5, edgecolor='darkred'))
    
    # Grid and styling for PVLib plot
    ax1.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Subplot 2: OSM-MEPS Optimal Angles
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#f8f9fa')
    
    # OSM-MEPS data - normalize azimuth values (360 becomes 0)
    monthly_tilts_osm = [monthly_results_osmmeps[month]['optimal_tilt'] for month in months_available]
    monthly_azimuths_osm = [normalize_azimuth(monthly_results_osmmeps[month]['optimal_azimuth']) for month in months_available]
    
    # Tilt line - OSM-MEPS (Green)
    color3 = 'green'
    line3 = ax3.plot(month_labels, monthly_tilts_osm, 'o-', color=color3, linewidth=2.5, markersize=12, 
                     markerfacecolor=color3, markeredgecolor='white', markeredgewidth=2)
    ax3.set_ylabel('Optimal Tilt Angle (°)', color=color3, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=16)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.set_ylim(0, 90)
    ax3.set_xlim(-0.5, len(month_labels)-0.5)
    
    # Azimuth line on secondary axis - OSM-MEPS (Cyan)
    ax4 = ax3.twinx()
    color4 = 'cyan'
    line4 = ax4.plot(month_labels, monthly_azimuths_osm, 's-', color=color4, linewidth=2.5, markersize=10, 
                     markerfacecolor=color4, markeredgecolor='white', markeredgewidth=2)
    ax4.set_ylabel('Optimal Azimuth Angle (°)', color=color4, fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax4.tick_params(axis='y', labelcolor=color4, labelsize=16)
    ax4.set_ylim(0, 359)  # Maximum azimuth set to 359
    # Set proper azimuth ticks (0=N, 90=E, 180=S, 270=W)
    ax4.set_yticks([0, 90, 180, 270])
    ax4.set_yticklabels(['0° (N)', '90° (E)', '180° (S)', '270° (W)'])
    
    # Add annotations for OSM-MEPS points
    for i, (tilt, azimuth) in enumerate(zip(monthly_tilts_osm, monthly_azimuths_osm)):
        ax3.annotate(f'{tilt}°', (i, tilt), xytext=(0, 12), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightgreen', alpha=0.5, edgecolor='darkgreen'))
        
        ax4.annotate(f'{azimuth}°', (i, azimuth), xytext=(0, -11), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcyan', alpha=0.5, edgecolor='darkcyan'))
    
    # Grid and styling for OSM-MEPS plot
    ax3.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    ax3.set_axisbelow(True)
    
    # Adjust layout and save
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.4)
    plt.savefig("GREECE_MONTHLY_OPTIMAL_ANGLES_COMPARISON.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# Create the monthly tilt-azimuth plot for Greece
create_monthly_tilt_azimuth_plot_greece()

#-------------------------------------------------------------------------------------------------------------------


# === PLOTTING CODE: Combined Monthly Tilt-Azimuth Configuration ===

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

def normalize_azimuth(azimuth):
    """Normalize azimuth to 0-359 range, converting 360 to 0"""
    return azimuth % 360

def create_combined_tilt_azimuth_plot():
    """Create combined monthly tilt-azimuth plot for both models using dual y-axes"""
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_facecolor('#f8f9fa')
    
    months_available = [month for month in months if month in monthly_results_pvlib]
    month_labels = [month_names[month-1] for month in months_available]
    
    # Extract data for both models
    monthly_tilts_pvlib = [monthly_results_pvlib[month]['optimal_tilt'] for month in months_available]
    monthly_azimuths_pvlib = [normalize_azimuth(monthly_results_pvlib[month]['optimal_azimuth']) for month in months_available]
    monthly_tilts_osm = [monthly_results_osmmeps[month]['optimal_tilt'] for month in months_available]
    monthly_azimuths_osm = [normalize_azimuth(monthly_results_osmmeps[month]['optimal_azimuth']) for month in months_available]
    
    # Create second y-axis for azimuth
    ax2 = ax1.twinx()
    
    # Tilt lines on left y-axis (ax1)
    color_pvlib_tilt = 'orange'
    color_osm_tilt = 'green'
    color_pvlib_azimuth = 'maroon'
    color_osm_azimuth = 'cyan'
    
    # Plot tilt data on left y-axis
    line1 = ax1.plot(month_labels, monthly_tilts_pvlib, 'o-', color=color_pvlib_tilt, linewidth=2.5, markersize=12, 
                     markerfacecolor=color_pvlib_tilt, markeredgecolor='white', markeredgewidth=2,
                     label='PVLIB Optimal Tilt')
    
    line2 = ax1.plot(month_labels, monthly_tilts_osm, 's-', color=color_osm_tilt, linewidth=2.5, markersize=10, 
                     markerfacecolor=color_osm_tilt, markeredgecolor='white', markeredgewidth=2,
                     label='OSM-MEPS Optimal Tilt')
    
    # Plot azimuth data on right y-axis
    line3 = ax2.plot(month_labels, monthly_azimuths_pvlib, 'o--', color=color_pvlib_azimuth, linewidth=2.5, markersize=12, 
                     markerfacecolor=color_pvlib_azimuth, markeredgecolor='white', markeredgewidth=2,
                     label='PVLIB Optimal Azimuth')
    
    line4 = ax2.plot(month_labels, monthly_azimuths_osm, 's--', color=color_osm_azimuth, linewidth=2.5, markersize=10, 
                     markerfacecolor=color_osm_azimuth, markeredgecolor='white', markeredgewidth=2,
                     label='OSM-MEPS Optimal Azimuth')
    
    # Configure left y-axis (tilt)
    ax1.set_ylabel('Optimal Tilt Angle (°)', fontsize=18, fontfamily='Garamond', fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_ylim(0, 90)
    ax1.set_xlim(-0.5, len(month_labels)-0.5)
    ax1.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    
    # Configure right y-axis (azimuth)
    ax2.set_ylabel('Optimal Azimuth Angle (°)', fontsize=18, fontfamily='Garamond', fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax2.set_ylim(0, 359)
    
    # Set proper azimuth ticks (0=N, 90=E, 180=S, 270=W)
    ax2.set_yticks([0, 90, 180, 270])
    ax2.set_yticklabels(['0° (N)', '90° (E)', '180° (S)', '270° (W)'])
    
    # Add annotations for tilt points
    for i, (tilt_pvlib, tilt_osm) in enumerate(zip(monthly_tilts_pvlib, monthly_tilts_osm)):
        ax1.annotate(f'{tilt_pvlib}°', (i, tilt_pvlib), xytext=(0, 12), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightblue', alpha=0.5, edgecolor='navy'))
        
        ax1.annotate(f'{tilt_osm}°', (i, tilt_osm), xytext=(0, -11), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightgreen', alpha=0.5, edgecolor='darkgreen'))
    
    # Add annotations for azimuth points
    for i, (azimuth_pvlib, azimuth_osm) in enumerate(zip(monthly_azimuths_pvlib, monthly_azimuths_osm)):
        ax2.annotate(f'{azimuth_pvlib}°', (i, azimuth_pvlib), xytext=(0, 12), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcoral', alpha=0.5, edgecolor='darkred'))
        
        ax2.annotate(f'{azimuth_osm}°', (i, azimuth_osm), xytext=(0, -11), textcoords="offset points", 
                     ha='center', fontsize=10, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcyan', alpha=0.5, edgecolor='darkcyan'))
    
    # Grid and styling
    ax1.grid(True, alpha=0.5, linestyle='-', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper center', bbox_to_anchor=(0.5, -0.13),
               ncol=4, fontsize=14, framealpha=0.9,
               facecolor='white', edgecolor='gray')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("COMBINED_MONTHLY_OPTIMAL_ANGLES_tilt_azimuth.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# Create the combined monthly tilt-azimuth plot  GREECE
create_combined_tilt_azimuth_plot()


#------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# === PLOTTING CODE: Compass-Loop Oriented Tilt–Azimuth (E=Baseline) ===

# Set global style parameters for professional aesthetic
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

def compass_transform_e_baseline(az):
    """
    Transform azimuth so that:
    - E (90°) is at baseline (x-axis level)
    - W (270°) is at both top and bottom extremes
    - Circular compass layout
    """
    az = normalize_azimuth(az)
    # Transform to have 90° (E) at 0, 270° (W) at ±180
    transformed = (az - 90) % 360
    if transformed > 180:
        transformed = transformed - 360
    return transformed

def create_compass_loop_plot():
    """Create combined monthly tilt-azimuth plot with E=baseline orientation"""
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.set_facecolor('#f8f9fa')

    months_available = [m for m in months if m in monthly_results_pvlib]
    month_labels = [month_names[m - 1] for m in months_available]

    # === Extract Data ===
    pvlib_tilt = [monthly_results_pvlib[m]['optimal_tilt'] for m in months_available]
    osm_tilt = [monthly_results_osmmeps[m]['optimal_tilt'] for m in months_available]
    pvlib_az = [compass_transform_e_baseline(monthly_results_pvlib[m]['optimal_azimuth']) for m in months_available]
    osm_az = [compass_transform_e_baseline(monthly_results_osmmeps[m]['optimal_azimuth']) for m in months_available]

    print("PVLIB Transformed Azimuth values:", pvlib_az)
    print("OSM Transformed Azimuth values:", osm_az)
    print("PVLIB Tilt values:", pvlib_tilt)
    print("OSM Tilt values:", osm_tilt)

    # === Twin Axis for Tilt ===
    ax2 = ax1.twinx()

    # === Colors ===
    color_pvlib_tilt = 'orange'
    color_osm_tilt = 'green'
    color_pvlib_azimuth = 'maroon'
    color_osm_azimuth = 'cyan'

    # === ENHANCED GRID LINES - ADDED FIRST ===
    
    # Major grid lines for azimuth (every 45 degrees)
    az_major_ticks = np.arange(-180, 181, 45)
    for tick in az_major_ticks:
        ax1.axhline(y=tick, color='lightgray', linewidth=0.8, linestyle='-', alpha=0.55)
    
    # Minor grid lines for azimuth (every 15 degrees)  
    az_minor_ticks = np.arange(-180, 181, 15)
    for tick in az_minor_ticks:
        if tick not in az_major_ticks:  # Avoid duplicating major grid lines
            ax1.axhline(y=tick, color='lightgray', linewidth=0.5, linestyle='--', alpha=0.45)
    
    # Major grid lines for tilt (every 15 degrees)
    tilt_major_ticks = np.arange(0, 91, 15)
    for tick in tilt_major_ticks:
        ax2.axhline(y=tick, color='lightblue', linewidth=0.6, linestyle='-', alpha=0.55)
    
    # Minor grid lines for tilt (every 5 degrees)
    tilt_minor_ticks = np.arange(0, 91, 5)
    for tick in tilt_minor_ticks:
        if tick not in tilt_major_ticks:  # Avoid duplicating major grid lines
            ax2.axhline(y=tick, color='lightblue', linewidth=0.5, linestyle='--', alpha=0.45)
    
    # Vertical grid lines for months
    for i in range(len(month_labels)):
        ax1.axvline(x=i, color='lightgray', linewidth=0.5, linestyle='-', alpha=0.65)

    # === Plot AZIMUTH ===
    line3 = ax1.plot(month_labels, pvlib_az, 'o--', color=color_pvlib_azimuth,
             markerfacecolor=color_pvlib_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,
             label='PVLIB Optimal Azimuth')

    line4 = ax1.plot(month_labels, osm_az, 's--', color=color_osm_azimuth,
             markerfacecolor=color_osm_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,
             label='OSM-MEPS Optimal Azimuth')

    # === Plot TILT ===
    line1 = ax2.plot(month_labels, pvlib_tilt, 'o-', color=color_pvlib_tilt,
             markerfacecolor=color_pvlib_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,
             label='PVLIB Optimal Tilt')

    line2 = ax2.plot(month_labels, osm_tilt, 's-', color=color_osm_tilt,
             markerfacecolor=color_osm_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,
             label='OSM-MEPS Optimal Tilt')

    # === Axis Setup - E (90°) as Baseline ===
    ax1.set_ylim(-180, 180)  # W (270°) at both top and bottom
    ax1.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.7)  # E (90°) baseline
    
    ax1.set_ylabel('Optimal Azimuth Angle (°)', fontsize=18, fontfamily='Garamond', fontweight='bold', color='black')

    # Ticks every 45 degrees with compass points
    az_ticks = np.arange(-180, 181, 45)
    az_labels = []
    for t in az_ticks:
        # Convert back to original azimuth for labeling
        orig_az = (t + 90) % 360
        if orig_az == 90:
            label = 'E\n90°'
        elif orig_az == 0:
            label = 'N\n0°'
        elif orig_az == 180:
            label = 'S\n180°'
        elif orig_az == 270:
            label = 'W\n270°'
        elif orig_az == 45:
            label = '45°'
        elif orig_az == 135:
            label = '135°'
        elif orig_az == 225:
            label = '225°'
        elif orig_az == 315:
            label = '315°'
        else:
            label = f'{orig_az}°'
        az_labels.append(label)

    ax1.set_yticks(az_ticks)
    ax1.set_yticklabels(az_labels)
    ax1.tick_params(axis='y', labelsize=14, labelcolor='black')

    # === Tilt Axis ===
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Optimal Tilt Angle (°)', fontsize=18, fontfamily='Garamond', fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelsize=16, labelcolor='black')
    ax2.set_yticks(np.arange(0, 91, 15))  # Set major ticks for tilt

    # === X-axis Configuration ===
    ax1.set_xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
    ax1.set_xlim(-0.5, len(month_labels)-0.5)
    ax1.tick_params(axis='x', labelsize=16)

    # === Annotations ===
    
    # Annotate Azimuth points (ax1)
    for i, (azimuth_pvlib, azimuth_osm) in enumerate(zip(pvlib_az, osm_az)):
        # Get original azimuth values for display
        pvlib_az_orig = normalize_azimuth(monthly_results_pvlib[months_available[i]]['optimal_azimuth'])
        osm_az_orig = normalize_azimuth(monthly_results_osmmeps[months_available[i]]['optimal_azimuth'])
        
        # Dynamic positioning based on transformed values
        if azimuth_pvlib > 0:
            pvlib_voffset = 24
            pvlib_va = 'bottom'
        else:
            pvlib_voffset = -5
            pvlib_va = 'top'
            
        if azimuth_osm > 0:
            osm_voffset = -5
            osm_va = 'bottom'
        else:
            osm_voffset = -7
            osm_va = 'bottom'
        
        ax1.annotate(f'{pvlib_az_orig:.0f}°', (i, azimuth_pvlib), 
                     xytext=(0, pvlib_voffset), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcoral', alpha=0.8, edgecolor=color_pvlib_azimuth))
        
        ax1.annotate(f'{osm_az_orig:.0f}°', (i, azimuth_osm), 
                     xytext=(0, osm_voffset), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcyan', alpha=0.8, edgecolor=color_osm_azimuth))

    # Annotate Tilt points (ax2)
    for i, (tilt_pvlib, tilt_osm) in enumerate(zip(pvlib_tilt, osm_tilt)):
        # Simple vertical positioning for tilt
        if tilt_pvlib > tilt_osm:
            pvlib_voffset = -2
            osm_voffset = -2
        else:
            pvlib_voffset = -10
            osm_voffset = 15
        
        ax2.annotate(f'{tilt_pvlib:.0f}°', (i, tilt_pvlib), 
                     xytext=(-8, pvlib_voffset), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightblue', alpha=0.8, edgecolor=color_pvlib_tilt))
        
        ax2.annotate(f'{tilt_osm:.0f}°', (i, tilt_osm), 
                     xytext=(8, osm_voffset), textcoords="offset points", 
                     ha='center', fontsize=11, fontfamily='Garamond', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightgreen', alpha=0.8, edgecolor=color_osm_tilt))

    # === Legend ===
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, 
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=4, fontsize=16, framealpha=0.9,
               facecolor='white', edgecolor='gray')

    # === Adjust layout and save ===
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig("GREECE_SERRES_C_TILT-AZIMUTH_PVLIB_OSM_MEPS_COMPASS_ELSEVIER.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# === Run the E=baseline plot ===
create_compass_loop_plot()


#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# === PLOTTING CODE: Compass-Loop Oriented Tilt–Azimuth (E=Baseline) ===

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 20  # Increased by 3 steps (from 17)
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['figure.facecolor'] = '#fafafa'
plt.rcParams['font.weight'] = 'bold'  # Make all text bold

def normalize_azimuth(az):
    """Normalize to 0–360"""
    return az % 360

def compass_transform_e_baseline(az):
    """
    Transform azimuth so that:
    - E (90°) is at baseline (x-axis level)
    - W (270°) is at both top and bottom extremes
    - Circular compass layout
    """
    az = normalize_azimuth(az)
    # Transform to have 90° (E) at 0, 270° (W) at ±180
    transformed = (az - 90) % 360
    if transformed > 180:
        transformed = transformed - 360
    return transformed

def create_compass_loop_plot():
    """Create combined monthly tilt-azimuth plot with E=baseline orientation"""
    
    fig, ax1 = plt.subplots(figsize=(10, 7))  # Changed to length 10, height 6
    ax1.set_facecolor('#fafafa')

    months_available = [m for m in months if m in monthly_results_pvlib]
    month_labels = [month_names[m - 1] for m in months_available]

    # === Extract Data ===
    pvlib_tilt = [monthly_results_pvlib[m]['optimal_tilt'] for m in months_available]
    osm_tilt = [monthly_results_osmmeps[m]['optimal_tilt'] for m in months_available]
    pvlib_az = [compass_transform_e_baseline(monthly_results_pvlib[m]['optimal_azimuth']) for m in months_available]
    osm_az = [compass_transform_e_baseline(monthly_results_osmmeps[m]['optimal_azimuth']) for m in months_available]

    print("PVLIB Transformed Azimuth values:", pvlib_az)
    print("OSM Transformed Azimuth values:", osm_az)
    print("PVLIB Tilt values:", pvlib_tilt)
    print("OSM Tilt values:", osm_tilt)

    # === Twin Axis for Tilt ===
    ax2 = ax1.twinx()
    ax2.set_facecolor('#fafafa')

    # === Colors ===
    color_pvlib_tilt = 'orange'
    color_osm_tilt = 'green'
    color_pvlib_azimuth = 'maroon'
    color_osm_azimuth = 'cyan'

    # === ENHANCED GRID LINES - ADDED FIRST ===
    
    # Major grid lines for azimuth (every 45 degrees)
    az_major_ticks = np.arange(-180, 181, 45)
    for tick in az_major_ticks:
        ax1.axhline(y=tick, color='lightgray', linewidth=0.8, linestyle='-', alpha=0.55)
    
    # Minor grid lines for azimuth (every 15 degrees)  
    az_minor_ticks = np.arange(-180, 181, 15)
    for tick in az_minor_ticks:
        if tick not in az_major_ticks:  # Avoid duplicating major grid lines
            ax1.axhline(y=tick, color='lightgray', linewidth=0.5, linestyle='--', alpha=0.45)
    
    # Major grid lines for tilt (every 15 degrees)
    tilt_major_ticks = np.arange(0, 91, 15)
    for tick in tilt_major_ticks:
        ax2.axhline(y=tick, color='lightblue', linewidth=0.6, linestyle='-', alpha=0.55)
    
    # Minor grid lines for tilt (every 5 degrees)
    tilt_minor_ticks = np.arange(0, 91, 5)
    for tick in tilt_minor_ticks:
        if tick not in tilt_major_ticks:  # Avoid duplicating major grid lines
            ax2.axhline(y=tick, color='lightblue', linewidth=0.5, linestyle='--', alpha=0.45)
    
    # Vertical grid lines for months
    for i in range(len(month_labels)):
        ax1.axvline(x=i, color='lightgray', linewidth=0.5, linestyle='-', alpha=0.65)

    # === Plot AZIMUTH ===
    line3 = ax1.plot(month_labels, pvlib_az, 'o--', color=color_pvlib_azimuth,
             markerfacecolor=color_pvlib_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,
             label='PVLIB Optimal Azimuth')

    line4 = ax1.plot(month_labels, osm_az, 's--', color=color_osm_azimuth,
             markerfacecolor=color_osm_azimuth, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,
             label='OSM-MEPS Optimal Azimuth')

    # === Plot TILT ===
    line1 = ax2.plot(month_labels, pvlib_tilt, 'o-', color=color_pvlib_tilt,
             markerfacecolor=color_pvlib_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=12,
             label='PVLIB Optimal Tilt')

    line2 = ax2.plot(month_labels, osm_tilt, 's-', color=color_osm_tilt,
             markerfacecolor=color_osm_tilt, markeredgecolor='white',
             markeredgewidth=2, linewidth=3.2, markersize=10,
             label='OSM-MEPS Optimal Tilt')

    # === Axis Setup - E (90°) as Baseline ===
    ax1.set_ylim(-180, 180)  # W (270°) at both top and bottom
    ax1.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.7)  # E (90°) baseline
    
    ax1.set_ylabel('Optimal Azimuth Angle (°)', fontsize=21, fontfamily='Garamond', fontweight='bold', color='black')  # Increased font size

    # Ticks every 45 degrees with compass points
    az_ticks = np.arange(-180, 181, 45)
    az_labels = []
    for t in az_ticks:
        # Convert back to original azimuth for labeling
        orig_az = (t + 90) % 360
        if orig_az == 90:
            label = 'E\n90°'
        elif orig_az == 0:
            label = 'N\n0°'
        elif orig_az == 180:
            label = 'S\n180°'
        elif orig_az == 270:
            label = 'W\n270°'
        elif orig_az == 45:
            label = '45°'
        elif orig_az == 135:
            label = '135°'
        elif orig_az == 225:
            label = '225°'
        elif orig_az == 315:
            label = '315°'
        else:
            label = f'{orig_az}°'
        az_labels.append(label)

    ax1.set_yticks(az_ticks)
    ax1.set_yticklabels(az_labels)
    ax1.tick_params(axis='y', labelsize=17, labelcolor='black')  # Increased font size

    # === Tilt Axis ===
    ax2.set_ylim(0, 90)
    ax2.set_ylabel('Optimal Tilt Angle (°)', fontsize=21, fontfamily='Garamond', fontweight='bold', color='black')  # Increased font size
    ax2.tick_params(axis='y', labelsize=19, labelcolor='black')  # Increased font size
    ax2.set_yticks(np.arange(0, 91, 15))  # Set major ticks for tilt

    # === X-axis Configuration ===
    ax1.set_xlabel('Month', fontsize=21, fontfamily='Garamond', fontweight='bold')  # Increased font size
    ax1.set_xlim(-0.5, len(month_labels)-0.5)
    ax1.tick_params(axis='x', labelsize=19)  # Increased font size

    # === Annotations ===
    
    # Annotate Azimuth points (ax1)
    for i, (azimuth_pvlib, azimuth_osm) in enumerate(zip(pvlib_az, osm_az)):
        # Get original azimuth values for display
        pvlib_az_orig = normalize_azimuth(monthly_results_pvlib[months_available[i]]['optimal_azimuth'])
        osm_az_orig = normalize_azimuth(monthly_results_osmmeps[months_available[i]]['optimal_azimuth'])
        
        # Dynamic positioning based on transformed values
        if azimuth_pvlib > 0:
            pvlib_voffset = 24
            pvlib_va = 'bottom'
        else:
            pvlib_voffset = -5
            pvlib_va = 'top'
            
        if azimuth_osm > 0:
            osm_voffset = -5
            osm_va = 'bottom'
        else:
            osm_voffset = -7
            osm_va = 'bottom'
        
        ax1.annotate(f'{pvlib_az_orig:.0f}°', (i, azimuth_pvlib), 
                     xytext=(0, pvlib_voffset), textcoords="offset points", 
                     ha='center', fontsize=14, fontfamily='Garamond', fontweight='bold',  # Increased font size
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcoral', alpha=0.8, edgecolor=color_pvlib_azimuth))
        
        ax1.annotate(f'{osm_az_orig:.0f}°', (i, azimuth_osm), 
                     xytext=(0, osm_voffset), textcoords="offset points", 
                     ha='center', fontsize=14, fontfamily='Garamond', fontweight='bold',  # Increased font size
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightcyan', alpha=0.8, edgecolor=color_osm_azimuth))

    # Annotate Tilt points (ax2)
    for i, (tilt_pvlib, tilt_osm) in enumerate(zip(pvlib_tilt, osm_tilt)):
        # Simple vertical positioning for tilt
        if tilt_pvlib > tilt_osm:
            pvlib_voffset = -2
            osm_voffset = -2
        else:
            pvlib_voffset = -10
            osm_voffset = 15
        
        ax2.annotate(f'{tilt_pvlib:.0f}°', (i, tilt_pvlib), 
                     xytext=(-8, pvlib_voffset), textcoords="offset points", 
                     ha='center', fontsize=14, fontfamily='Garamond', fontweight='bold',  # Increased font size
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightblue', alpha=0.8, edgecolor=color_pvlib_tilt))
        
        ax2.annotate(f'{tilt_osm:.0f}°', (i, tilt_osm), 
                     xytext=(8, osm_voffset), textcoords="offset points", 
                     ha='center', fontsize=14, fontfamily='Garamond', fontweight='bold',  # Increased font size
                     bbox=dict(boxstyle="round,pad=0.12", facecolor='lightgreen', alpha=0.8, edgecolor=color_osm_tilt))

    # === Legend ===
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, 
               loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, fontsize=16, framealpha=0.9,  # Increased font size
               facecolor='white', edgecolor='gray')

    # === Adjust layout and save ===
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig("GREECE_SERRES_C_TILT-AZIMUTH_PVLIB_OSM_MEPS_COMPASS_ELSEVIER_PETER.pdf", format="pdf", 
                bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.show()

# === Run the E=baseline plot ===
create_compass_loop_plot()

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
