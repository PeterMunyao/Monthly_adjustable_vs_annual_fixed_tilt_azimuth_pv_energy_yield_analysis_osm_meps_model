# Solar Panel Optimization Analysis

## Project Overview

This Python-based solar energy analysis tool performs comprehensive optimization of photovoltaic (PV) system configurations by comparing two different solar modeling approaches (PVLib and OSM-MEPS) to determine optimal tilt and azimuth angles for maximum energy generation.

## Key Features

- **Dual-Model Comparison**: Simultaneously analyzes PV system performance using both PVLib and custom OSM-MEPS models
- **Monthly Optimization**: Finds optimal tilt and azimuth angles for each month of the year
- **Fixed Configuration Analysis**: Determines best year-round fixed panel orientation
- **Performance Comparison**: Quantifies energy gains from monthly adjustments vs fixed positioning


## Technical Implementation

### Core Components

#### 1. Data Processing
- Loads and processes ground validated, satellite-based weather data from Solcast DNV company (5-minute intervals)
- Handles meteorological parameters: DNI, GHI, DHI, temperature, albedo, cloud opacity
- Filters data for specific time periods (e.g., full year 2024)

#### 2. Solar Modeling Algorithms

**PVLib Model:**
- Uses industry-standard pvlib library for irradiance calculations
- Implements SAPM (Sandia Array Performance Model) for temperature effects
- Comprehensive solar position and irradiance modeling

**OSM-MEPS Model:**
- Custom physics-based model with simplified Plane of Array (POA) calculations, which generalizes to other geographical areas with use of high quality data eg Solcast DNV data
- Incorporates cloud opacity and humidity correction factors and dynamic albedo effects (data from Solcast)


## Outputs Generated

### Numerical Results
- Monthly optimal tilt and azimuth angles for both models
- Energy production comparisons
- Statistical analysis metrics to validate that the optimal monthly tilt-azimuth configurations are not random noise or by chance.

### Visualizations
- Compass-oriented tilt-azimuth plots with E(90°) baseline
- Annotated data points 

### Prerequisites
```bash
pip install pandas numpy matplotlib pvlib
