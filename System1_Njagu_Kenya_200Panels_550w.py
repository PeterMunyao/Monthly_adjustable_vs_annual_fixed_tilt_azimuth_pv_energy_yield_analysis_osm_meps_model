import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# Set global style parameters for professional aesthetic
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

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

# System Design Parameters
panel_power_max = 550  # Watts
num_panels = 200

# System capacity
system_capacity_kw = (num_panels * panel_power_max) / 1000

# Define tilt and azimuth ranges to analyze
tilt_angles = np.arange(0, 91, 1)  # 0 to 90 degrees in steps of 5
azimuth_angles = np.arange(0, 360, 2)  # 0 to 360 degrees in steps of 10

# Initialize dictionary to store monthly results
monthly_results = {}

# Get unique months in the dataset
df['month'] = df.index.month
months = sorted(df['month'].unique())

# FIXED: Improved angle of incidence calculation with numerical stability
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

# Iterate over each month
for month in months:
    print(f"Processing month {month}...")
    
    # Filter data for the current month
    month_data = df[df['month'] == month].copy()
    
    # Skip if no data for this month
    if len(month_data) == 0:
        continue
    
    # Initialize variables for monthly optimization
    max_energy = -np.inf
    min_energy = np.inf
    optimal_tilt, optimal_azimuth = None, None
    worst_tilt, worst_azimuth = None, None
    
    # Iterate over tilt and azimuth angles for this month
    for tilt in tilt_angles:
        for azimuth in azimuth_angles:
            tilt_rad = np.radians(tilt)
            azimuth_rad = np.radians(month_data['azimuth'])
            surface_azimuth_rad = np.radians(azimuth)
            zenith_rad = np.radians(month_data['zenith'])
            
            # FIXED: Use improved AOI calculation
            aoi = calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad)
            
            # Irradiance components
            month_data['poa_direct'] = month_data['dni'] * np.cos(np.radians(aoi))
            month_data['poa_direct'] = month_data['poa_direct'].clip(lower=0)
            month_data['poa_diffuse'] = month_data['dhi'] * (1 + np.cos(tilt_rad)) / 2
            month_data['poa_sky_diffuse'] = month_data['ghi'] * month_data['albedo'] * (1 - np.cos(tilt_rad)) / 2
            month_data['poa_total'] = month_data['poa_direct'] + month_data['poa_diffuse'] + month_data['poa_sky_diffuse']
            
            # Module temperature
            nominal_operating_cell_temp = 45
            month_data['module_temp'] = nominal_operating_cell_temp + month_data['poa_total'] / 800 * (28 - month_data['air_temp'])
            
            # Power output
            temp_coeff = -0.0045
            month_data['panel_power'] = panel_power_max * (1 + temp_coeff * (month_data['module_temp'] - nominal_operating_cell_temp))
            stc_irradiance = 1000
            month_data['dc_power'] = month_data['panel_power'] * month_data['poa_total'] / stc_irradiance
        
            # Inverter efficiency
            inverter_efficiency = 0.85
            month_data['ac_power'] = month_data['dc_power'] * inverter_efficiency
            month_data['scaled_power'] = month_data['ac_power'] * num_panels
            
            # Convert to energy in kWh for every 5-minute interval
            energy_kwh = month_data['scaled_power'] * (5 / 60) / 1000
            
            # Calculate total energy production for this month
            total_energy = energy_kwh.sum()
            
            # Update max and min energy configurations for this month
            if total_energy > max_energy:
                max_energy = total_energy
                optimal_tilt, optimal_azimuth = tilt, azimuth
                best_energy_profile = energy_kwh.copy()
                
            if total_energy < min_energy:
                min_energy = total_energy
                worst_tilt, worst_azimuth = tilt, azimuth
                worst_energy_profile = energy_kwh.copy()
    
    # Store monthly results
    monthly_results[month] = {
        'optimal_tilt': optimal_tilt,
        'optimal_azimuth': optimal_azimuth,
        'max_energy': max_energy,
        'worst_tilt': worst_tilt,
        'worst_azimuth': worst_azimuth,
        'min_energy': min_energy,
        'best_energy_profile': best_energy_profile,
        'worst_energy_profile': worst_energy_profile
    }
    
    print(f"Month {month}: Optimal Tilt: {optimal_tilt}°, Optimal Azimuth: {optimal_azimuth}°, Max Energy: {max_energy:.2f} kWh")

# Calculate total annual energy from optimal monthly configurations
total_annual_energy = sum(monthly_results[month]['max_energy'] for month in monthly_results)
print(f"\n=== TOTAL ANNUAL ENERGY FROM OPTIMAL MONTHLY CONFIGURATIONS: {total_annual_energy:.2f} kWh ===")

# FIND OPTIMAL FIXED TILT AND AZIMUTH FOR THE ENTIRE YEAR
print("\n=== FINDING OPTIMAL FIXED TILT AND AZIMUTH FOR ENTIRE YEAR ===")

max_energy_fixed = -np.inf
optimal_fixed_tilt, optimal_fixed_azimuth = None, None

for tilt in tilt_angles:
    for azimuth in azimuth_angles:
        tilt_rad = np.radians(tilt)
        azimuth_rad = np.radians(df['azimuth'])
        surface_azimuth_rad = np.radians(azimuth)
        zenith_rad = np.radians(df['zenith'])
        
        # FIXED: Use improved AOI calculation
        aoi = calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad)
        
        # Irradiance components
        df['poa_direct'] = df['dni'] * np.cos(np.radians(aoi))
        df['poa_direct'] = df['poa_direct'].clip(lower=0)
        df['poa_diffuse'] = df['dhi'] * (1 + np.cos(tilt_rad)) / 2
        df['poa_sky_diffuse'] = df['ghi'] * df['albedo'] * (1 - np.cos(tilt_rad)) / 2
        df['poa_total'] = df['poa_direct'] + df['poa_diffuse'] + df['poa_sky_diffuse']
        
        # Module temperature
        nominal_operating_cell_temp = 45
        df['module_temp'] = nominal_operating_cell_temp + df['poa_total'] / 800 * (28 - df['air_temp'])
        
        # Power output
        temp_coeff = -0.0045
        df['panel_power'] = panel_power_max * (1 + temp_coeff * (df['module_temp'] - nominal_operating_cell_temp))
        stc_irradiance = 1000
        df['dc_power'] = df['panel_power'] * df['poa_total'] / stc_irradiance
        
        # Inverter efficiency
        inverter_efficiency = 0.85
        df['ac_power'] = df['dc_power'] * inverter_efficiency
        df['scaled_power'] = df['ac_power'] * num_panels
        
        # Convert to energy in kWh for every 5-minute interval
        energy_kwh = df['scaled_power'] * (5 / 60) / 1000
        
        # Calculate total energy production for the entire year
        total_energy = energy_kwh.sum()
        
        # Update max energy configuration for fixed system
        if total_energy > max_energy_fixed:
            max_energy_fixed = total_energy
            optimal_fixed_tilt, optimal_fixed_azimuth = tilt, azimuth
            fixed_energy_profile = energy_kwh.copy()

print(f"Optimal Fixed Tilt: {optimal_fixed_tilt}°, Optimal Fixed Azimuth: {optimal_fixed_azimuth}°, Max Annual Energy: {max_energy_fixed:.2f} kWh")

# Calculate energy for each month using fixed optimal configuration
monthly_fixed_energy = {}
for month in months:
    if month in monthly_results:
        month_data = df[df['month'] == month].copy()
        
        # Calculate energy for this month using fixed optimal configuration
        tilt_rad = np.radians(optimal_fixed_tilt)
        azimuth_rad = np.radians(month_data['azimuth'])
        surface_azimuth_rad = np.radians(optimal_fixed_azimuth)
        zenith_rad = np.radians(month_data['zenith'])
        
        # FIXED: Use improved AOI calculation
        aoi = calculate_aoi(tilt_rad, azimuth_rad, surface_azimuth_rad, zenith_rad)
        
        # Irradiance components
        month_data['poa_direct'] = month_data['dni'] * np.cos(np.radians(aoi))
        month_data['poa_direct'] = month_data['poa_direct'].clip(lower=0)
        month_data['poa_diffuse'] = month_data['dhi'] * (1 + np.cos(tilt_rad)) / 2
        month_data['poa_sky_diffuse'] = month_data['ghi'] * month_data['albedo'] * (1 - np.cos(tilt_rad)) / 2
        month_data['poa_total'] = month_data['poa_direct'] + month_data['poa_diffuse'] + month_data['poa_sky_diffuse']
        
        # Module temperature
        nominal_operating_cell_temp = 45
        month_data['module_temp'] = nominal_operating_cell_temp + month_data['poa_total'] / 800 * (28 - month_data['air_temp'])
        
        # Power output
        temp_coeff = -0.0045
        month_data['panel_power'] = panel_power_max * (1 + temp_coeff * (month_data['module_temp'] - nominal_operating_cell_temp))
        stc_irradiance = 1000
        month_data['dc_power'] = month_data['panel_power'] * month_data['poa_total'] / stc_irradiance
        
        # Inverter efficiency
        inverter_efficiency = 0.85
        month_data['ac_power'] = month_data['dc_power'] * inverter_efficiency
        month_data['scaled_power'] = month_data['ac_power'] * num_panels
        
        # Convert to energy in kWh for every 5-minute interval
        energy_kwh = month_data['scaled_power'] * (5 / 60) / 1000
        
        monthly_fixed_energy[month] = energy_kwh.sum()

# Print comparison results
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print("\n=== COMPARISON: MONTHLY OPTIMAL vs FIXED OPTIMAL ===")
print(f"{'Month':<12} {'Optimal Tilt':<12} {'Optimal Azimuth':<15} {'Monthly Opt Energy':<18} {'Fixed Opt Energy':<16} {'Difference':<12} {'Gain %':<8}")
print("-" * 95)

total_gain = 0
for month in months:
    if month in monthly_results:
        monthly_opt_energy = monthly_results[month]['max_energy']
        fixed_opt_energy = monthly_fixed_energy[month]
        difference = monthly_opt_energy - fixed_opt_energy
        gain_percent = (difference / fixed_opt_energy) * 100 if fixed_opt_energy > 0 else 0
        total_gain += difference
        
        print(f"{month_names[month-1]:<12} {monthly_results[month]['optimal_tilt']:<12} {monthly_results[month]['optimal_azimuth']:<15} "
              f"{monthly_opt_energy:<18.2f} {fixed_opt_energy:<16.2f} {difference:<12.2f} {gain_percent:<8.2f}%")

annual_gain_percent = ((total_annual_energy - max_energy_fixed) / max_energy_fixed) * 100

print("-" * 95)
print(f"{'ANNUAL TOTALS':<12} {'Fixed: ' + str(optimal_fixed_tilt):<12} {'Fixed: ' + str(optimal_fixed_azimuth):<15} "
      f"{total_annual_energy:<18.2f} {max_energy_fixed:<16.2f} {total_annual_energy - max_energy_fixed:<12.2f} {annual_gain_percent:<8.2f}%")


# Statistical Significance Analysis Function
def statistical_significance_test(monthly_results, monthly_fixed_energy, months, month_names):
    """
    Perform statistical significance testing between monthly optimal and fixed optimal configurations
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)
    
    # Extract the data
    monthly_optimal_energy = []
    fixed_optimal_energy = []
    monthly_tilts = []
    monthly_azimuths = []
    monthly_labels = []
    
    for month in months:
        if month in monthly_results and month in monthly_fixed_energy:
            monthly_optimal_energy.append(monthly_results[month]['max_energy'])
            fixed_optimal_energy.append(monthly_fixed_energy[month])
            monthly_tilts.append(monthly_results[month]['optimal_tilt'])
            monthly_azimuths.append(monthly_results[month]['optimal_azimuth'])
            monthly_labels.append(month_names[month-1])
    
    # Convert to numpy arrays for easier calculations
    monthly_optimal = np.array(monthly_optimal_energy)
    fixed_optimal = np.array(fixed_optimal_energy)
    
    # Calculate differences
    differences = monthly_optimal - fixed_optimal
    percentage_differences = (differences / fixed_optimal) * 100
    
    print(f"VERIFICATION - This should match your comparison table:")
    print(f"{'Month':<12} {'Mth Opt (kWh)':<14} {'Fixed Opt (kWh)':<15} {'Diff (kWh)':<12} {'Gain %':<8}")
    print("-" * 65)
    
    total_monthly_opt = 0
    total_fixed_opt = 0
    for i, month_label in enumerate(monthly_labels):
        print(f"{month_label:<12} {monthly_optimal[i]:<14.2f} {fixed_optimal[i]:<15.2f} {differences[i]:<12.2f} {percentage_differences[i]:<8.2f}%")
        total_monthly_opt += monthly_optimal[i]
        total_fixed_opt += fixed_optimal[i]
    
    print("-" * 65)
    print(f"{'TOTALS':<12} {total_monthly_opt:<14.2f} {total_fixed_opt:<15.2f} {total_monthly_opt-total_fixed_opt:<12.2f} {((total_monthly_opt-total_fixed_opt)/total_fixed_opt)*100:<8.2f}%")
    
    # Statistical Tests
    print(f"\nSTATISTICAL ANALYSIS:")
    
    # 1. Paired t-test
    t_stat, p_value = stats.ttest_rel(monthly_optimal, fixed_optimal)
    print(f"Paired t-test Results:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Degrees of freedom = {len(monthly_optimal)-1}")
    
    # 2. Calculate confidence intervals
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin_error = t_critical * se_diff
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error
    
    print(f"\nConfidence Intervals (95%):")
    print(f"Mean monthly difference: {mean_diff:.2f} kWh")
    print(f"Standard deviation: {std_diff:.2f} kWh")
    print(f"Standard error: {se_diff:.2f} kWh")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] kWh")
    print(f"Margin of error: ±{margin_error:.2f} kWh")
    
    # 3. Effect size
    cohens_d = mean_diff / std_diff
    print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
    
    # Interpretation
    print(f"\nSTATISTICAL SIGNIFICANCE:")
    alpha = 0.05
    
    if p_value < alpha:
        print(f"✓ STATISTICALLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"  The improvement from monthly adjustment is statistically significant")
    else:
        print(f"✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"  The improvement may be due to random variation")
    
    # Practical significance analysis
    print(f"\nPRACTICAL SIGNIFICANCE ANALYSIS:")
    print(f"Average monthly gain: {mean_diff:.2f} kWh ({np.mean(percentage_differences):.2f}%)")
    print(f"Total annual gain: {np.sum(differences):.2f} kWh ({(np.sum(differences)/np.sum(fixed_optimal))*100:.2f}%)")
    
    # Monthly breakdown
    print(f"\nMONTHLY PERFORMANCE ANALYSIS:")
    print(f"{'Month':<12} {'Gain (kWh)':<12} {'Gain (%)':<10} {'Tilt':<6} {'Azimuth':<8} {'Significance':<12}")
    print("-" * 65)
    
    significance_count = {'High': 0, 'Medium': 0, 'Low': 0, 'Negative': 0}
    
    for i, month_label in enumerate(monthly_labels):
        gain_kwh = differences[i]
        gain_pct = percentage_differences[i]
        
        # Significance based on percentage gain
        if gain_pct > 5:
            significance = "High"
        elif gain_pct > 2:
            significance = "Medium" 
        elif gain_pct > 0:
            significance = "Low"
        else:
            significance = "Negative"
        
        significance_count[significance] += 1
            
        print(f"{month_label:<12} {gain_kwh:<12.2f} {gain_pct:<10.2f} {monthly_tilts[i]:<6} {monthly_azimuths[i]:<8} {significance:<12}")
    
    print("-" * 65)
    print(f"Summary: {significance_count['High']} high, {significance_count['Medium']} medium, {significance_count['Low']} low gain months")
    
    return monthly_optimal, fixed_optimal, differences, p_value, monthly_tilts, monthly_azimuths, monthly_labels

###-----------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

# ---- Extract data from your previous analysis ----
months_available = [month for month in months if month in monthly_results]
month_labels = [month_names[month-1] for month in months_available]

monthly_opt_energies = [monthly_results[month]['max_energy'] for month in months_available]
fixed_opt_energies = [monthly_fixed_energy[month] for month in months_available]

monthly_tilts = [monthly_results[month]['optimal_tilt'] for month in months_available]
monthly_azimuths = [monthly_results[month]['optimal_azimuth'] for month in months_available]

differences = [opt - fix for opt, fix in zip(monthly_opt_energies, fixed_opt_energies)]

x_pos = np.arange(len(month_labels))
bar_width = 0.48

# ---- PLOT 1: Comparison of monthly optimal vs fixed optimal energy ----
plt.figure(figsize=(14, 7))
brown_color = '#8B4513'
tan_color = '#D2B48C'

plt.bar(x_pos - bar_width/2, monthly_opt_energies, bar_width, label='Monthly Optimal Tilt-azimuth', color=brown_color, alpha=0.8)
plt.bar(x_pos + bar_width/2, fixed_opt_energies, bar_width, 
        label=f'Fixed Optimal Tilt-azimuth (Tilt: {optimal_fixed_tilt}°, Azimuth: {optimal_fixed_azimuth}°)', 
        color=tan_color, alpha=0.8)

plt.xlabel('Month', fontsize=18, fontweight='bold', fontfamily='Garamond')
plt.ylabel('Energy Production (kWh)', fontsize=18, fontweight='bold', fontfamily='Garamond')
plt.xticks(x_pos, month_labels, rotation=1, fontsize=18)
plt.yticks(fontsize=16)
plt.grid(True, alpha=0.3)

# Add value labels
for i, (opt_energy, fixed_energy) in enumerate(zip(monthly_opt_energies, fixed_opt_energies)):
    plt.text(i - bar_width/2, opt_energy + max(monthly_opt_energies)*0.01, f'{opt_energy:.0f}', 
             ha='center', va='bottom', fontsize=12, fontfamily='Garamond', fontweight='bold')
    plt.text(i + bar_width/2, fixed_energy + max(fixed_opt_energies)*0.01, f'{fixed_energy:.0f}', 
             ha='center', va='bottom', fontsize=12, fontfamily='Garamond', fontweight='bold')

# Move legend below the graph
plt.legend(fontsize=16, framealpha=0.9, facecolor='white', edgecolor='gray', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("NJAGU_MONTHLY_vs_FIXED_OPTIMAL_COMPARISON.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

# ---- PLOT 2: Monthly energy gains ----
plt.figure(figsize=(14, 7))
plt.bar(month_labels, differences, color='#2E8B57', alpha=0.7, width=0.55)
plt.xlabel('Month', fontsize=18, fontfamily='Garamond', fontweight='bold')
plt.ylabel('Energy Gain (kWh)', fontsize=18, fontfamily='Garamond', fontweight='bold')
plt.xticks(rotation=1, fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, alpha=0.3)

# Add value labels
for i, diff in enumerate(differences):
    plt.text(i, diff + max(differences)*0.01, f'{diff:.1f}', ha='center', va='bottom', fontsize=14, fontfamily='Garamond', fontweight='bold')

plt.tight_layout()
plt.savefig("STATISTICAL_ANALYSIS_ENERGY_GAINS.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

# ---- PLOT 3: Monthly optimal tilt and azimuth ----
fig, ax1 = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#f8f9fa')
ax1.set_facecolor('#f8f9fa')

# Tilt line
color1 = '#1f77b4'
ax1.set_xlabel('Month', fontsize=20, fontfamily='Garamond', fontweight='bold')
ax1.set_ylabel('Optimal Tilt Angle (°)', color=color1, fontsize=20, fontfamily='Garamond', fontweight='bold')
line1 = ax1.plot(month_labels, monthly_tilts, 'o-', color=color1, linewidth=2.5, markersize=14, markerfacecolor=color1, markeredgecolor='white', markeredgewidth=3, label='Optimal Tilt Angle')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=18)
ax1.tick_params(axis='x', labelsize=18, rotation=1)
ax1.set_ylim(0, 90)
ax1.set_xlim(-0.5, len(month_labels)-0.5)

# Azimuth line on secondary axis
ax2 = ax1.twinx()
color2 = '#d62728'
ax2.set_ylabel('Optimal Azimuth Angle (°)', color=color2, fontsize=20, fontfamily='Garamond', fontweight='bold')
line2 = ax2.plot(month_labels, monthly_azimuths, 's-', color=color2, linewidth=2.5, markersize=14, markerfacecolor=color2, markeredgecolor='white', markeredgewidth=3, label='Optimal Azimuth Angle')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=18)
ax2.set_ylim(0, 360)

# Grid and ticks
ax1.grid(True, alpha=0.5, linestyle='-', linewidth=1.0)
ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.7, which='minor')
ax1.minorticks_on()
ax1.set_yticks([0, 15, 30, 45, 60, 75, 90])
ax1.set_yticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax2.set_yticks([0, 90, 180, 270, 360])
ax2.set_yticks([45, 135, 225, 315], minor=True)

# Legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
legend = ax1.legend(lines, labels, loc='upper left', fontsize=16, framealpha=0.9, facecolor='white', edgecolor='gray')
legend.get_frame().set_linewidth(2)

# Annotations
for i, (tilt, azimuth) in enumerate(zip(monthly_tilts, monthly_azimuths)):
    ax1.annotate(f'{tilt}°', (i, tilt), xytext=(0,10), textcoords="offset points", ha='center', fontsize=11, fontfamily='Garamond', bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.55))
    ax2.annotate(f'{azimuth}°', (i, azimuth), xytext=(0,-10), textcoords="offset points", ha='center', fontsize=11, fontfamily='Garamond', bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcoral', alpha=0.55))

plt.tight_layout()
plt.savefig("NJAGU_MONTHLY_OPTIMAL_ANGLES_COMBINED.pdf", format="pdf", bbox_inches='tight', facecolor='#f8f9fa', dpi=300)
plt.show()

#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def statistical_significance_test(monthly_results, monthly_fixed_energy, months, month_names):
    """
    Perform statistical significance testing between monthly optimal and fixed optimal configurations
    """
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS - CORRECTED")
    print("="*80)
    
    # Extract the SAME data that was used in your comparison table
    monthly_optimal_energy = []
    fixed_optimal_energy = []
    monthly_tilts = []
    monthly_azimuths = []
    monthly_labels = []
    
    for month in months:
        if month in monthly_results and month in monthly_fixed_energy:
            monthly_optimal_energy.append(monthly_results[month]['max_energy'])
            fixed_optimal_energy.append(monthly_fixed_energy[month])
            monthly_tilts.append(monthly_results[month]['optimal_tilt'])
            monthly_azimuths.append(monthly_results[month]['optimal_azimuth'])
            monthly_labels.append(month_names[month-1])
    
    # Convert to numpy arrays for easier calculations
    monthly_optimal = np.array(monthly_optimal_energy)
    fixed_optimal = np.array(fixed_optimal_energy)
    
    # Calculate differences (this should match your comparison table)
    differences = monthly_optimal - fixed_optimal
    percentage_differences = (differences / fixed_optimal) * 100
    
    print(f"VERIFICATION - This should match your comparison table:")
    print(f"{'Month':<12} {'Mth Opt (kWh)':<14} {'Fixed Opt (kWh)':<15} {'Diff (kWh)':<12} {'Gain %':<8}")
    print("-" * 65)
    
    total_monthly_opt = 0
    total_fixed_opt = 0
    for i, month_label in enumerate(monthly_labels):
        print(f"{month_label:<12} {monthly_optimal[i]:<14.2f} {fixed_optimal[i]:<15.2f} {differences[i]:<12.2f} {percentage_differences[i]:<8.2f}%")
        total_monthly_opt += monthly_optimal[i]
        total_fixed_opt += fixed_optimal[i]
    
    print("-" * 65)
    print(f"{'TOTALS':<12} {total_monthly_opt:<14.2f} {total_fixed_opt:<15.2f} {total_monthly_opt-total_fixed_opt:<12.2f} {((total_monthly_opt-total_fixed_opt)/total_fixed_opt)*100:<8.2f}%")
    
    # Statistical Tests
    print(f"\nSTATISTICAL ANALYSIS:")
    
    # 1. Paired t-test
    t_stat, p_value = stats.ttest_rel(monthly_optimal, fixed_optimal)
    print(f"Paired t-test Results:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Degrees of freedom = {len(monthly_optimal)-1}")
    
    # 2. Calculate confidence intervals
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin_error = t_critical * se_diff
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error
    
    print(f"\nConfidence Intervals (95%):")
    print(f"Mean monthly difference: {mean_diff:.2f} kWh")
    print(f"Standard deviation: {std_diff:.2f} kWh")
    print(f"Standard error: {se_diff:.2f} kWh")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] kWh")
    print(f"Margin of error: ±{margin_error:.2f} kWh")
    
    # 3. Effect size
    cohens_d = mean_diff / std_diff
    print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
    
    # Interpretation
    print(f"\nSTATISTICAL SIGNIFICANCE:")
    alpha = 0.05
    
    if p_value < alpha:
        print(f"✓ STATISTICALLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"  The improvement from monthly adjustment is statistically significant")
    else:
        print(f"✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f})")
        print(f"  The improvement may be due to random variation")
    
    # Practical significance analysis
    print(f"\nPRACTICAL SIGNIFICANCE ANALYSIS:")
    print(f"Average monthly gain: {mean_diff:.2f} kWh ({np.mean(percentage_differences):.2f}%)")
    print(f"Total annual gain: {np.sum(differences):.2f} kWh ({(np.sum(differences)/np.sum(fixed_optimal))*100:.2f}%)")
    
    # Monthly breakdown with CORRECT data
    print(f"\nMONTHLY PERFORMANCE ANALYSIS (CORRECTED):")
    print(f"{'Month':<12} {'Gain (kWh)':<12} {'Gain (%)':<10} {'Tilt':<6} {'Azimuth':<8} {'Significance':<12}")
    print("-" * 65)
    
    significance_count = {'High': 0, 'Medium': 0, 'Low': 0, 'Negative': 0}
    
    for i, month_label in enumerate(monthly_labels):
        gain_kwh = differences[i]
        gain_pct = percentage_differences[i]
        
        # Significance based on percentage gain
        if gain_pct > 5:
            significance = "High"
        elif gain_pct > 2:
            significance = "Medium" 
        elif gain_pct > 0:
            significance = "Low"
        else:
            significance = "Negative"
        
        significance_count[significance] += 1
            
        print(f"{month_label:<12} {gain_kwh:<12.2f} {gain_pct:<10.2f} {monthly_tilts[i]:<6} {monthly_azimuths[i]:<8} {significance:<12}")
    
    print("-" * 65)
    print(f"Summary: {significance_count['High']} high, {significance_count['Medium']} medium, {significance_count['Low']} low gain months")
    
    return monthly_optimal, fixed_optimal, differences, p_value

def plot_corrected_analysis(monthly_optimal, fixed_optimal, differences, monthly_labels, monthly_tilts, monthly_azimuths):
    """
    Create accurate visualizations for the statistical analysis
    """
   

# Run the corrected analysis
monthly_optimal, fixed_optimal, differences, p_value = statistical_significance_test(
    monthly_results, monthly_fixed_energy, months, month_names
)

# Get additional data for plotting
monthly_tilts = [monthly_results[month]['optimal_tilt'] for month in months if month in monthly_results]
monthly_azimuths = [monthly_results[month]['optimal_azimuth'] for month in months if month in monthly_results]
monthly_labels = [month_names[month-1] for month in months if month in monthly_results]

plot_corrected_analysis(monthly_optimal, fixed_optimal, differences, monthly_labels, monthly_tilts, monthly_azimuths)

# Final assessment
print(f"\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

mean_gain = np.mean(differences)
mean_percentage_gain = np.mean([(diff/fixed)*100 for diff, fixed in zip(differences, fixed_optimal)])

if p_value < 0.05:
    if mean_percentage_gain > 3.0:
        print("✓ STRONG RECOMMENDATION: Implement monthly adjustment")
        print(f"  • Statistically significant (p = {p_value:.6f})")
        print(f"  • Substantial average gain: {mean_percentage_gain:.2f}%")
    
    else:
        print("○ MODERATE RECOMMENDATION: Consider seasonal adjustment")
        print(f"  • Statistically significant but modest gain: {mean_percentage_gain:.2f}%")
        print(f"  • Focus on high-gain months only")
else:
    print("● LIMITED BENEFIT: Monthly adjustment may not be worthwhile")
    print(f"  • Not statistically significant (p = {p_value:.6f})")

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
