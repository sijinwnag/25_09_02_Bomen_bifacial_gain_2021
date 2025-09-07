# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This workspace contains **bifacial solar panel gain analysis** for the Bomen Solar Farm using 2021 operational data. The primary focus is on **validating PVsyst simulation models** against measured electrical performance data to assess the benefits of bifacial photovoltaic systems.

**Core Analysis Workflow:**
- **Data Integration** - Load measured electrical power data (5-minute intervals) and PVsyst simulation results (hourly)
- **Data Processing** - Convert power to energy, resample to daily totals, filter for maintenance-free days
- **Model Validation** - Compare simulation vs. measured performance using statistical metrics
- **Optimization** - Find optimal scaling factors to minimize bias (MBE ≈ 0)
- **Performance Assessment** - Calculate RMSE, CRMSE, MAPE, and seasonal performance metrics

## Essential Development Commands

### Environment Setup
```bash
# Primary environment setup
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Key dependencies for solar analysis
pip install pandas numpy matplotlib scikit-learn seaborn jupyter pvlib
```

### Data Analysis Execution
```bash
# Navigate to code directory
cd Code

# Launch Jupyter notebook for interactive analysis
jupyter notebook

# Run batch evaluation on multiple PVsyst files (site-level)
python batch_pvsyst_evaluation.py

# Run batch evaluation for individual inverters
python batch_pvsyst_evaluation_inv.py --inverter "2-1"

# Generate maintenance filter (if needed)
python maintenance_filter.py
```

### Key Notebook Workflows
```bash
# Main analysis notebooks (execute in order)
jupyter notebook 25_09_02_Data_visualiser_matching.ipynb          # Site-level analysis
jupyter notebook 25_09_05_Data_visualiser_matching_inv.ipynb      # Individual inverter analysis
```

## High-Level Architecture

### Data Processing Pipeline
**PVsyst Simulation → Measured Data → Validation → Metrics**

1. **Data Loading**: CSV parsing with robust datetime handling and encoding detection
2. **Data Alignment**: Temporal synchronization between simulation (hourly) and measurements (5-minute)
3. **Energy Conversion**: Power-to-energy transformation with proper time weighting
4. **Filtering**: Maintenance-free days filter to exclude operational disruptions
5. **Optimization**: Binary search algorithm to find optimal scaling factors
6. **Validation**: Comprehensive statistical analysis with seasonal breakdown

### Three-Stage Validation Framework
**Load → Process → Validate**

1. **Data Integration**: Robust CSV parsing with multiple datetime format support
2. **Statistical Optimization**: Binary search algorithm achieving MBE ≤ 1e-13 tolerance
3. **Performance Metrics**: RMSE, CRMSE, MAPE, nRMSE with seasonal analysis

### Class-Based Architecture
Both Python scripts use object-oriented design:
```python
class PVsystBatchEvaluator:
    def __init__(self, project_root=None, inverter=None)
    def run_batch_evaluation()
    def process_single_file()
    def find_optimal_scaling_factor()
```

## Critical Data Architecture

### Physical System Configuration
- **Location**: Bomen Solar Farm (-35.0708°, 147.3842°), NSW, Australia  
- **Analysis Period**: 2021 full year with maintenance filtering
- **Data Resolution**: 5-minute measurements, hourly simulations
- **System Type**: Bifacial photovoltaic with single-axis tracking

### Data Sources & Formats
```
Raw Data Sources → Processing → Analysis → Results Export
     ↓                ↓           ↓            ↓
PVsyst CSV     → Date parsing → Energy      Results/
(semicolon;)     Latin-1        conversion    *.csv
Electrical     → Timestamp    → Daily        *.xlsx  
Pickle (.pkl)    indexing       totals       *.log
```

### Key File Locations
- **Electrical Data**: `Data/full_site_pow_5min.pkl` (5-minute power measurements)
- **Individual Data**: `Data/full_inv_pow_5min.pkl` (per-inverter measurements)
- **PVsyst Files**: `Data/PVsyst/param optimisation/*.CSV` (simulation results)
- **Per-Inverter**: `Data/PVsyst/per_inv/{inverter}/*.CSV` (individual simulations)
- **Maintenance Filter**: `Results/remaining_dates_2021.txt` (maintenance-free dates)

### Core Functions & Patterns

**CSV Parsing Pattern** (handles PVsyst format complexity):
```python
# Robust PVsyst CSV loading
df = pd.read_csv(
    file_path,
    delimiter=';',
    skiprows=list(range(10)) + [11],  # Skip metadata + units
    header=0,
    encoding='latin-1',
    low_memory=False
)

# Multi-format datetime parsing
df['timestamp'] = pd.to_datetime(df[date_col], format='%d/%m/%y %H:%M', errors='coerce')
if df['timestamp'].isna().any():
    df['timestamp'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
```

**Energy Conversion Pattern**:
```python
# 5-minute power to energy conversion
df['Energy_MWh'] = df['Power'] * (5/60)  # 5 minutes = 5/60 hours
daily_energy = df['Energy_MWh'].resample('D').sum()

# Hourly simulation to daily energy
simulation_df['EArray_MWh'] = simulation_df['EArray'] * 1.0  # 1 hour
daily_simulated = simulation_df['EArray_MWh'].resample('D').sum()
```

**Optimization Algorithm**:
```python
def find_optimal_scaling_factor(min_factor=0.5, max_factor=2.0, max_iterations=100):
    # Binary search to achieve MBE ≤ 1e-13
    while iterations < max_iterations:
        mid_factor = (min_factor + max_factor) / 2
        mbe = calculate_mbe(mid_factor)
        if abs(mbe) < target_mbe_tolerance:
            return mid_factor, mbe, iterations
```

## Development Workflow Patterns

### Batch Processing Pattern
Both scripts follow consistent workflow:
1. **Initialize** - Auto-detect project root, set up logging
2. **Load Data** - Electrical measurements and maintenance filter
3. **Process Files** - Iterate through PVsyst CSV files
4. **Optimize** - Find optimal scaling factor for each file
5. **Export** - Save results to CSV with comprehensive metrics

### Error Handling Strategy
- **Robust CSV parsing** with multiple encoding attempts
- **Flexible datetime parsing** with format fallbacks  
- **Comprehensive logging** with progress tracking
- **Graceful degradation** when files fail to process

### Results Export Structure
```
Results/
├── pvsyst_batch_evaluation_results.csv          # Site-level results
├── pvsyst_batch_evaluation_results_{inverter}.csv  # Per-inverter results
├── *.xlsx                                       # Excel format results
├── remaining_dates_2021.txt                     # Maintenance-free days filter
└── *.log                                        # Execution logs
```

## Key Analysis Parameters

### Optimal Performance Targets
```python
# Statistical targets achieved by optimization
performance_targets = {
    'mbe_tolerance': 1e-13,        # Mean Bias Error ≤ 1e-13 MW
    'max_iterations': 100,         # Binary search limit
    'scaling_range': [0.5, 2.0],   # Scale factor bounds
}

# Expected results format
evaluation_metrics = {
    'RMSE': 'Root Mean Square Error (MWh/day)',
    'MBE': 'Mean Bias Error (≈0 after optimization)', 
    'CRMSE': 'Centralized Root Mean Square Error',
    'MAPE': 'Mean Absolute Percentage Error (%)',
    'nRMSE': 'Normalized RMSE (% of measurement range)'
}
```

### Data Filtering Configuration
```python
# Maintenance-free filtering (critical for accuracy)
maintenance_filter = {
    'file': 'Results/remaining_dates_2021.txt',
    'format': 'One date per line (YYYY-MM-DD)',
    'purpose': 'Exclude days with maintenance/outages'
}

# Data quality filters
data_filters = {
    'remove_zero_values': False,    # Keep zero generation (nighttime)
    'match_timestamps': True,       # Only overlapping dates  
    'dropna_values': True          # Remove missing measurements
}
```

## Development Guidelines

### Working with PVsyst Data
- **Always use semicolon delimiter** for CSV files
- **Skip first 10 rows + row 11** (metadata and units)
- **Use latin-1 encoding** to handle special characters
- **Apply multiple datetime format attempts** for robustness
- **Validate EArray column exists** before processing

### Statistical Analysis Patterns  
- **Optimize scaling before metrics** to minimize bias
- **Calculate both scaled and unscaled metrics** for comparison
- **Include seasonal breakdown** (Summer/Autumn/Winter/Spring)
- **Use maintenance-free days filter** for operational accuracy
- **Validate convergence** of optimization algorithm

### Error Prevention
- **Check file existence** before processing
- **Validate data shapes** after resampling
- **Monitor optimization convergence** 
- **Log all processing steps** for debugging
- **Handle missing columns gracefully**

## Project Structure

### Main Directories
- **`Code/`** - Analysis scripts and Jupyter notebooks
- **`Data/`** - Electrical measurements and PVsyst simulation files
  - **`Data/PVsyst/param optimisation/`** - Site-level simulation files
  - **`Data/PVsyst/per_inv/{inverter}/`** - Individual inverter simulations
- **`Results/`** - Analysis outputs, metrics, and plots
- **`.venv/`** - Python virtual environment

### Essential Files
- **`requirements.txt`** - Python dependencies (comprehensive list with 300+ packages)
- **`Code/batch_pvsyst_evaluation.py`** - Site-level batch processor
- **`Code/batch_pvsyst_evaluation_inv.py`** - Individual inverter processor  
- **`Code/maintenance_filter.py`** - Data filtering utilities
- **`Code/25_09_02_Data_visualiser_matching.ipynb`** - Interactive site analysis
- **`Code/25_09_05_Data_visualiser_matching_inv.ipynb`** - Interactive inverter analysis

This framework enables systematic **bifacial solar panel performance validation** through rigorous statistical comparison of PVsyst simulations against measured data, providing quantitative assessment of model accuracy and bifacial gain benefits.