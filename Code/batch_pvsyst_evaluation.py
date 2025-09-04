#!/usr/bin/env python3
"""
Batch PVsyst Evaluation Script

This script processes multiple PVsyst CSV files and generates evaluation metrics 
for each one, comparing against measured electrical data. The script reproduces 
the evaluation results from notebook section "3.2.3.2. Plot the data after filtering".

Requirements:
- PVsyst CSV files in Data/PVsyst/param optimisation/
- Measured electrical power data (5-minute intervals)
- Maintenance-free days file: Results/remaining_dates_2021.txt

Output:
- CSV file with evaluation metrics for each PVsyst file
- Columns: file directory, RMSE, MBE, CRMSE, The optimised scale factor
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_evaluation.log'),
        logging.StreamHandler()
    ]
)

class PVsystBatchEvaluator:
    """Batch processor for PVsyst simulation files evaluation"""
    
    def __init__(self, project_root=None):
        """Initialize the batch evaluator with project paths"""
        if project_root is None:
            # Auto-detect project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
        
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "Data"
        self.results_dir = self.project_root / "Results"
        self.pvsyst_dir = self.data_dir / "PVsyst" / "param optimisation"
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.target_mbe_tolerance = 1e-13
        self.max_iterations = 100
        
        logging.info(f"Project root: {self.project_root}")
        logging.info(f"PVsyst directory: {self.pvsyst_dir}")
        
    def load_maintenance_free_days(self):
        """Load maintenance-free days from text file"""
        maintenance_file = self.results_dir / "remaining_dates_2021.txt"
        
        if not maintenance_file.exists():
            logging.error(f"Maintenance file not found: {maintenance_file}")
            return None
            
        try:
            with open(maintenance_file, 'r') as f:
                maintenance_free_days = [line.strip() for line in f.readlines() if line.strip()]
            
            maintenance_free_dates = pd.to_datetime(maintenance_free_days)
            logging.info(f"Loaded {len(maintenance_free_dates)} maintenance-free days")
            return maintenance_free_dates
            
        except Exception as e:
            logging.error(f"Error loading maintenance-free days: {e}")
            return None
    
    def load_measured_power_data(self):
        """
        Load measured electrical power data from the specific pickle file
        """
        # Load the specific electrical data file
        pickle_file = self.data_dir / "full_site_pow_5min.pkl"
        
        if not pickle_file.exists():
            logging.error(f"Required electrical data file not found: {pickle_file}")
            return None
            
        try:
            logging.info(f"Loading measured electrical data from: {pickle_file}")
            df = pd.read_pickle(pickle_file)
            
            logging.info(f"Raw data shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            # Convert power from W to MW (as per notebook)
            if 'Power' in df.columns:
                df['Power'] = df['Power'] / 1e3  # Convert W to MW
                logging.info(f"Converted Power from W to MW. Range: {df['Power'].min():.3f} to {df['Power'].max():.3f} MW")
            else:
                logging.error("'Power' column not found in electrical data")
                return None
                
            # Set timestamp as index
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                logging.info(f"Set Timestamp as index. Date range: {df.index.min()} to {df.index.max()}")
            else:
                logging.error("'Timestamp' column not found in electrical data")
                return None
                
            logging.info(f"Successfully loaded {len(df)} measured data points")
            return df
            
        except Exception as e:
            logging.error(f"Error loading electrical data file: {e}")
            return None
    
    def load_pvsyst_csv(self, file_path):
        """Load and parse a single PVsyst CSV file"""
        try:
            logging.info(f"Loading PVsyst file: {file_path.name}")
            
            # Load CSV with the same parameters as in the notebook
            df = pd.read_csv(
                file_path,
                delimiter=';',
                skiprows=list(range(10)) + [11],  # Skip metadata (0-9) and units row (11)
                header=0,  # Row 10 becomes the header after skipping
                encoding='latin-1',
                low_memory=False,
                na_values=['', ' ', 'nan', 'NaN']
            )
            
            # Get the first column (should be date/time)
            date_col = df.columns[0]
            
            # Parse timestamps
            df['timestamp'] = pd.to_datetime(
                df[date_col], 
                format='%d/%m/%y %H:%M',  # Try 2-digit year format
                errors='coerce'
            )
            
            # If some dates failed to parse, try flexible parser
            if df['timestamp'].isna().any():
                df['timestamp'] = pd.to_datetime(
                    df[date_col],
                    dayfirst=True,
                    errors='coerce'
                )
            
            # Drop the original date column and clean data
            df = df.drop(columns=[date_col])
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Check if EArray column exists and convert if needed
            if 'EArray' not in df.columns:
                logging.error(f"EArray column not found in {file_path.name}")
                return None
                
            if pd.api.types.is_object_dtype(df['EArray']):
                df['EArray'] = pd.to_numeric(
                    df['EArray'].str.replace(',', '.'), 
                    errors='coerce'
                )
            
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"Successfully loaded {len(df)} data points from {file_path.name}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading {file_path.name}: {e}")
            return None
    
    
    def process_to_daily_energy(self, measured_df, simulation_df):
        """Convert both datasets to daily energy totals and filter"""
        
        if measured_df is None:
            logging.error("No measured electrical data available. Cannot proceed with evaluation.")
            return None, None
            
        # Process real measured data (5-minute intervals)
        logging.info("Processing measured electrical data...")
        
        # Convert Power (MW) to Energy (MWh) for 5-minute data
        measured_df['Energy_MWh'] = measured_df['Power'] * (5/60)  # 5 minutes = 5/60 hours
        logging.info(f"Created energy column. Sample values: {measured_df['Energy_MWh'].head().values}")
        
        # Resample to daily energy totals
        daily_actual_energy = measured_df['Energy_MWh'].resample('D').sum()
        logging.info(f"Resampled to daily totals: {len(daily_actual_energy)} days")
        logging.info(f"Daily energy range: {daily_actual_energy.min():.2f} to {daily_actual_energy.max():.2f} MWh/day")
        
        # Convert simulation to daily energy
        logging.info("Processing PVsyst simulation data...")
        simulation_df['EArray_MWh'] = simulation_df['EArray'] * 1.0  # 1 hour energy
        daily_simulated_energy = simulation_df['EArray_MWh'].resample('D').sum()
        logging.info(f"Simulation daily energy range: {daily_simulated_energy.min():.2f} to {daily_simulated_energy.max():.2f} MWh/day")
        
        return daily_actual_energy, daily_simulated_energy
    
    def filter_maintenance_data(self, daily_actual, daily_simulated, maintenance_dates):
        """Filter data to only include maintenance-free days"""
        
        # Create combined DataFrame
        metrics_df = pd.DataFrame()
        metrics_df['Actual'] = daily_actual
        metrics_df['Simulated'] = daily_simulated
        
        # Filter for matching dates
        initial_rows = len(metrics_df)
        metrics_df = metrics_df.dropna()
        logging.info(f"After filtering for matching dates: {len(metrics_df)} data points (removed {initial_rows - len(metrics_df)} NaN values)")
        
        # Filter for maintenance-free days if available
        if maintenance_dates is not None:
            metrics_df['date'] = metrics_df.index.date
            initial_rows = len(metrics_df)
            metrics_df = metrics_df[metrics_df['date'].isin([date.date() for date in maintenance_dates])]
            logging.info(f"After filtering for maintenance-free days: {len(metrics_df)} data points (removed {initial_rows - len(metrics_df)} points)")
        else:
            logging.warning("No maintenance filter applied - using all available data")
        
        # Keep zero values as per notebook logic
        logging.info("Keeping zero values in dataset")
        
        return metrics_df
    
    def calculate_mbe(self, metrics_df, scale_factor):
        """Calculate Mean Bias Error for a given scale factor"""
        scaled_data = metrics_df['Simulated'] * scale_factor
        return np.mean(scaled_data - metrics_df['Actual'])
    
    def find_optimal_scaling_factor(self, metrics_df, min_factor=0.5, max_factor=2.0):
        """Binary search to find optimal scaling factor that minimizes MBE"""
        
        iterations = 0
        best_factor = None
        best_mbe = float('inf')
        
        while iterations < self.max_iterations:
            iterations += 1
            mid_factor = (min_factor + max_factor) / 2
            
            # Calculate MBE for the current factor
            mbe = self.calculate_mbe(metrics_df, mid_factor)
            
            # Track the best factor found
            if abs(mbe) < abs(best_mbe):
                best_factor = mid_factor
                best_mbe = mbe
            
            # Check if we've reached the target precision
            if abs(mbe) < self.target_mbe_tolerance:
                logging.info(f"Converged! Found factor with MBE below tolerance after {iterations} iterations.")
                return mid_factor, mbe, iterations
            
            # Adjust search range based on MBE sign
            if mbe > 0:  # MBE is positive, need to decrease factor
                max_factor = mid_factor
            else:  # MBE is negative, need to increase factor
                min_factor = mid_factor
            
            # Check if search range is too small
            if max_factor - min_factor < 1e-15:
                logging.info(f"Reached numerical precision limit after {iterations} iterations.")
                return best_factor, best_mbe, iterations
        
        logging.info(f"Reached maximum iterations ({self.max_iterations}).")
        return best_factor, best_mbe, iterations
    
    def calculate_evaluation_metrics(self, metrics_df, optimal_factor):
        """Calculate evaluation metrics after applying optimal scaling"""
        
        # Apply optimal scaling factor
        metrics_df['Simulated_scaled'] = metrics_df['Simulated'] * optimal_factor
        
        # Calculate error metrics
        rmse = np.sqrt(mean_squared_error(metrics_df['Actual'], metrics_df['Simulated_scaled']))
        
        # CRMSE (Centralized Root Mean Square Error)
        actual_centered = metrics_df['Actual'] - metrics_df['Actual'].mean()
        simulated_centered = metrics_df['Simulated_scaled'] - metrics_df['Simulated_scaled'].mean()
        crmse = np.sqrt(np.mean((actual_centered - simulated_centered)**2))
        
        # MBE (Mean Bias Error) - should be very close to zero
        mbe = np.mean(metrics_df['Simulated_scaled'] - metrics_df['Actual'])
        
        return rmse, mbe, crmse
    
    def process_single_file(self, file_path, measured_data, maintenance_dates):
        """Process a single PVsyst file and return evaluation metrics"""
        
        try:
            # Load PVsyst simulation data
            simulation_df = self.load_pvsyst_csv(file_path)
            if simulation_df is None:
                return None
            
            # Convert to daily energy
            daily_actual, daily_simulated = self.process_to_daily_energy(measured_data, simulation_df)
            
            # Check if data processing was successful
            if daily_actual is None or daily_simulated is None:
                logging.error(f"Failed to process daily energy data for {file_path.name}")
                return None
            
            # Filter for maintenance-free days
            metrics_df = self.filter_maintenance_data(daily_actual, daily_simulated, maintenance_dates)
            
            if len(metrics_df) < 10:
                logging.warning(f"Very few data points ({len(metrics_df)}) for {file_path.name}")
            
            # Find optimal scaling factor
            optimal_factor, final_mbe, iterations = self.find_optimal_scaling_factor(metrics_df)
            
            # Calculate evaluation metrics
            rmse, mbe, crmse = self.calculate_evaluation_metrics(metrics_df, optimal_factor)
            
            logging.info(f"Results for {file_path.name}: RMSE={rmse:.3f}, MBE={mbe:.6e}, CRMSE={crmse:.3f}, Scale={optimal_factor:.6f}")
            
            return {
                'file directory': str(file_path),
                'RMSE': rmse,
                'MBE': mbe,
                'CRMSE': crmse,
                'The optimised scale factor': optimal_factor,
                'data_points': len(metrics_df),
                'iterations': iterations
            }
            
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def run_batch_evaluation(self):
        """Main function to run batch evaluation on all PVsyst files"""
        
        logging.info("Starting batch evaluation of PVsyst files")
        
        # Load measured power data
        measured_data = self.load_measured_power_data()
        
        if measured_data is None:
            logging.error("Could not load measured electrical data. Aborting batch evaluation.")
            return None
        
        # Load maintenance-free days
        maintenance_dates = self.load_maintenance_free_days()
        
        # Find all PVsyst CSV files
        pvsyst_files = list(self.pvsyst_dir.glob("*.CSV"))
        
        if not pvsyst_files:
            logging.error(f"No CSV files found in {self.pvsyst_dir}")
            return
        
        logging.info(f"Found {len(pvsyst_files)} PVsyst CSV files to process")
        
        # Process each file
        results = []
        for i, file_path in enumerate(pvsyst_files, 1):
            logging.info(f"Processing file {i}/{len(pvsyst_files)}: {file_path.name}")
            
            result = self.process_single_file(file_path, measured_data, maintenance_dates)
            if result:
                results.append(result)
        
        if not results:
            logging.error("No results obtained from any files")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        output_file = self.results_dir / "pvsyst_batch_evaluation_results.csv"
        results_df[['file directory', 'RMSE', 'MBE', 'CRMSE', 'The optimised scale factor']].to_csv(
            output_file, index=False
        )
        
        logging.info(f"Results saved to: {output_file}")
        
        # Print summary statistics
        logging.info("\n" + "="*50)
        logging.info("BATCH EVALUATION SUMMARY")
        logging.info("="*50)
        logging.info(f"Files processed successfully: {len(results)}")
        logging.info(f"Average RMSE: {results_df['RMSE'].mean():.3f} ± {results_df['RMSE'].std():.3f}")
        logging.info(f"Average CRMSE: {results_df['CRMSE'].mean():.3f} ± {results_df['CRMSE'].std():.3f}")
        logging.info(f"Average scaling factor: {results_df['The optimised scale factor'].mean():.6f} ± {results_df['The optimised scale factor'].std():.6f}")
        logging.info(f"Average data points per file: {results_df['data_points'].mean():.1f}")
        
        return results_df

def main():
    """Main function to run the batch evaluation"""
    
    print("PVsyst Batch Evaluation Tool")
    print("="*40)
    
    try:
        # Initialize evaluator
        evaluator = PVsystBatchEvaluator()
        
        # Run batch evaluation
        results = evaluator.run_batch_evaluation()
        
        if results is not None:
            print(f"\nBatch evaluation completed successfully!")
            print(f"Results saved to: {evaluator.results_dir}/pvsyst_batch_evaluation_results.csv")
        else:
            print("\nBatch evaluation failed!")
            
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        print(f"\nFatal error: {e}")

if __name__ == "__main__":
    main()