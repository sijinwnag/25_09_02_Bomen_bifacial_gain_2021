#!/usr/bin/env python3
"""
Batch PVsyst Evaluation Script for Individual Inverters

This script processes PVsyst CSV files for individual inverters and generates 
evaluation metrics comparing against measured electrical data for the selected inverter.
The script is based on the analysis workflow from the Jupyter notebook 
"25_09_05_Data_visualiser_matching_inv.ipynb".

Features:
- Interactive inverter selection from available electrical data
- Flexible optimization folder selection (e.g., Perez, Hay models)
- Support for custom folder paths for different optimization scenarios

Requirements:
- PVsyst CSV files in optimization folder (e.g., Data/PVsyst/per_inv/{inverter}/{subfolder}/)
- Measured individual inverter power data from full_inv_pow_5min.pkl
- Maintenance-free days file: Results/remaining_dates_2021.txt

Output:
- CSV file with evaluation metrics for the selected inverter
- Columns: file directory, inverter_id, model_name, Uc, Uv, RMSE, MBE, CRMSE, The optimised scale factor
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
import re
from pathlib import Path
from sklearn.metrics import mean_squared_error
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Configure logging (will be updated with inverter-specific filename later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_evaluation_inv.log'),
        logging.StreamHandler()
    ]
)

class PVsystBatchEvaluator:
    """Batch processor for PVsyst simulation files evaluation for individual inverters"""
    
    def __init__(self, project_root=None, inverter=None, optimization_folder=None, apply_clipping=None, clipping_threshold=None):
        """Initialize the batch evaluator with project paths, inverter selection, and optimization folder"""
        if project_root is None:
            # Auto-detect project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
        
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "Data"
        self.results_dir = self.project_root / "Results"
        
        # Select inverter and optimization folder if not provided
        if inverter is None or optimization_folder is None:
            self.inverter, self.optimization_folder = self.select_inverter_and_folder()
        else:
            self.inverter = inverter
            self.optimization_folder = optimization_folder
        
        # Configure power clipping if not provided
        if apply_clipping is None or clipping_threshold is None:
            self.apply_clipping, self.clipping_threshold = self.configure_power_clipping()
        else:
            self.apply_clipping = apply_clipping
            self.clipping_threshold = clipping_threshold
        
        # Set PVsyst directory based on selected optimization folder
        if self.optimization_folder:
            self.pvsyst_dir = Path(self.optimization_folder)
        else:
            # Fallback to default structure if no optimization folder specified
            self.pvsyst_dir = self.data_dir / "PVsyst" / "per_inv" / self.inverter
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.target_mbe_tolerance = 1e-13
        self.max_iterations = 100
        
        # Update logging to include inverter
        self._setup_logging()
        
        logging.info(f"Project root: {self.project_root}")
        logging.info(f"Selected inverter: {self.inverter}")
        logging.info(f"Optimization folder: {self.optimization_folder}")
        logging.info(f"PVsyst directory: {self.pvsyst_dir}")
        logging.info(f"Power clipping enabled: {self.apply_clipping}")
        if self.apply_clipping:
            logging.info(f"Clipping threshold: {self.clipping_threshold} MW")
        
    def _setup_logging(self):
        """Setup inverter-specific logging"""
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Add new handlers with inverter-specific log file
        log_file = f'batch_evaluation_{self.inverter.replace("-", "_")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )
    
    def configure_power_clipping(self):
        """Configure power clipping settings through user input"""
        print(f"\n{'='*60}")
        print("POWER CLIPPING CONFIGURATION")
        print(f"{'='*60}")
        print("Power clipping limits simulation power values to prevent unrealistic peaks.")
        print("This is applied to raw power data before daily energy conversion.")
        
        # Ask user if they want to apply clipping
        while True:
            try:
                clipping_choice = input("\nApply power clipping to simulation data? (y/n): ").strip().lower()
                
                if clipping_choice in ['y', 'yes']:
                    apply_clipping = True
                    break
                elif clipping_choice in ['n', 'no']:
                    apply_clipping = False
                    clipping_threshold = None
                    print("Power clipping disabled.")
                    return apply_clipping, clipping_threshold
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                raise SystemExit(1)
        
        # If clipping is enabled, ask for threshold
        while True:
            try:
                threshold_input = input("Enter clipping threshold in MW (default 2.392): ").strip()
                
                if not threshold_input:
                    # Use default
                    clipping_threshold = 2.392
                    break
                else:
                    # Validate user input
                    clipping_threshold = float(threshold_input)
                    if clipping_threshold <= 0:
                        print("Clipping threshold must be a positive number. Please try again.")
                        continue
                    break
                    
            except ValueError:
                print("Please enter a valid number for the clipping threshold.")
                continue
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                raise SystemExit(1)
        
        print(f"Power clipping enabled with threshold: {clipping_threshold} MW")
        return apply_clipping, clipping_threshold
    
    def validate_optimization_folder(self, folder_path):
        """Validate optimization folder exists and contains CSV files"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logging.error(f"Optimization folder does not exist: {folder_path}")
            return False
            
        if not folder_path.is_dir():
            logging.error(f"Path is not a directory: {folder_path}")
            return False
            
        # Check for CSV files
        csv_files = list(folder_path.glob("*.CSV")) + list(folder_path.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in optimization folder: {folder_path}")
            return False
            
        logging.info(f"Found {len(csv_files)} CSV files in optimization folder: {folder_path}")
        return True
    
    def extract_parameters_from_filename(self, filename):
        """Extract Uc, Uv values and model name from PVsyst filename"""
        try:
            # Pattern: "Bomen solar farm 2021 {Uc} {Uv} {model_type} model.csv"
            # Example: "Bomen solar farm 2021 11 0.0 Perez model.csv"
            pattern = r"Bomen solar farm 2021 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\w+) model\.csv"
            
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                uc_str, uv_str, model_name = match.groups()
                
                # Convert to appropriate numeric types
                uc_value = float(uc_str) if '.' in uc_str else int(uc_str)
                uv_value = float(uv_str) if '.' in uv_str else int(uv_str)
                
                logging.debug(f"Extracted from {filename}: Uc={uc_value}, Uv={uv_value}, Model={model_name}")
                return uc_value, uv_value, model_name
            else:
                logging.warning(f"Could not extract parameters from filename: {filename}")
                return None, None, None
                
        except Exception as e:
            logging.error(f"Error extracting parameters from filename {filename}: {e}")
            return None, None, None
    
    def detect_transpositional_model(self, results):
        """Detect transpositional model type from folder path or results"""
        try:
            # First try to extract from folder path (more reliable)
            if hasattr(self, 'optimization_folder') and self.optimization_folder:
                folder_name = Path(self.optimization_folder).name.lower()
                if folder_name in ['perez', 'hay', 'isotropic']:
                    model_name = folder_name.capitalize()
                    logging.info(f"Model type detected from folder path: {model_name}")
                    return model_name
            
            # Fallback: analyze results to determine most common model
            model_names = [r.get('model_name') for r in results if r and r.get('model_name')]
            if model_names:
                # Use most common model type
                model_counts = Counter(model_names)
                most_common_model = model_counts.most_common(1)[0][0]
                
                # Log information about model distribution
                if len(model_counts) > 1:
                    logging.info(f"Multiple models detected: {dict(model_counts)}. Using most common: {most_common_model}")
                else:
                    logging.info(f"Model type detected from filenames: {most_common_model}")
                
                return most_common_model
            else:
                logging.warning("No model type could be detected from folder path or filenames")
                return "Unknown"
                
        except Exception as e:
            logging.error(f"Error detecting transpositional model: {e}")
            return "Unknown"
    
    def select_inverter_and_folder(self):
        """Prompt user to select an inverter and optimization folder from available options"""
        # Load the electrical data file to get available inverters
        electrical_data_file = self.data_dir / "full_inv_pow_5min.pkl"
        
        if not electrical_data_file.exists():
            raise FileNotFoundError(f"Electrical data file not found: {electrical_data_file}")
        
        try:
            # Load just the header to get column names
            import pandas as pd
            df = pd.read_pickle(electrical_data_file)
            available_inverters = [col for col in df.columns if col != 'Timestamp']
            available_inverters.sort()  # Sort for consistent ordering
            
            print("Available inverters in electrical data:")
            for i, inverter in enumerate(available_inverters, 1):
                print(f"  {i:2d}. {inverter}")
            
            # Also check which inverters have PVsyst simulation files
            per_inv_dir = self.data_dir / "PVsyst" / "per_inv"
            if per_inv_dir.exists():
                pvsyst_inverters = [d.name for d in per_inv_dir.iterdir() if d.is_dir()]
                pvsyst_inverters.sort()
                
                print(f"\nInverters with PVsyst simulation files:")
                for inverter in pvsyst_inverters:
                    marker = "✓" if inverter in available_inverters else "✗ (no electrical data)"
                    print(f"  {marker} {inverter}")
                
                # Filter to inverters that have both electrical data and PVsyst files
                valid_inverters = [inv for inv in available_inverters if inv in pvsyst_inverters]
                
                if not valid_inverters:
                    raise ValueError("No inverters found with both electrical data and PVsyst simulation files")
                
                print(f"\nValid inverters (have both electrical data and PVsyst files): {len(valid_inverters)}")
                for inverter in valid_inverters:
                    print(f"  ✓ {inverter}")
            else:
                valid_inverters = available_inverters
                print(f"\nPVsyst per_inv directory not found. Using all available inverters from electrical data.")
            
            # Get user selection for inverter
            selected_inverter = None
            while True:
                try:
                    selection = input(f"\nSelect inverter to analyze (enter inverter ID, e.g., '2-1'): ").strip()
                    
                    if selection in valid_inverters:
                        print(f"Selected inverter: {selection}")
                        selected_inverter = selection
                        break
                    else:
                        print(f"Invalid selection. Please choose from: {', '.join(valid_inverters)}")
                        
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    raise SystemExit(1)
            
            # Now prompt for optimization folder
            print(f"\n{'='*60}")
            print("OPTIMIZATION FOLDER SELECTION")
            print(f"{'='*60}")
            
            # Check for available optimization subfolders
            default_inv_dir = per_inv_dir / selected_inverter
            available_subfolders = []
            
            if default_inv_dir.exists():
                subfolders = [d for d in default_inv_dir.iterdir() if d.is_dir()]
                if subfolders:
                    print(f"\nAvailable optimization folders for inverter {selected_inverter}:")
                    for i, subfolder in enumerate(subfolders, 1):
                        csv_count = len(list(subfolder.glob("*.CSV")) + list(subfolder.glob("*.csv")))
                        print(f"  {i}. {subfolder.name} ({csv_count} CSV files)")
                        available_subfolders.append(subfolder)
                    
                    print(f"  {len(subfolders) + 1}. Custom folder (enter full path)")
            
            # Get user selection for optimization folder
            selected_folder = None
            while True:
                try:
                    if available_subfolders:
                        choice = input(f"\nSelect optimization folder (1-{len(available_subfolders) + 1}) or enter full path: ").strip()
                        
                        # Check if it's a number selection
                        try:
                            choice_num = int(choice)
                            if 1 <= choice_num <= len(available_subfolders):
                                selected_folder = str(available_subfolders[choice_num - 1])
                                break
                            elif choice_num == len(available_subfolders) + 1:
                                # User wants to enter custom path
                                custom_path = input("Enter full path to optimization folder: ").strip()
                                custom_path = custom_path.strip('"\'')
                                if self.validate_optimization_folder(custom_path):
                                    selected_folder = custom_path
                                    break
                                else:
                                    print("Please try again with a valid folder path.")
                                    continue
                            else:
                                print(f"Invalid selection. Please choose 1-{len(available_subfolders) + 1} or enter a full path.")
                                continue
                        except ValueError:
                            # Not a number, treat as custom path
                            choice = choice.strip('"\'')
                            if self.validate_optimization_folder(choice):
                                selected_folder = choice
                                break
                            else:
                                print("Please try again with a valid folder path.")
                                continue
                    else:
                        # No subfolders found, ask for custom path
                        custom_path = input("Enter full path to optimization folder: ").strip()
                        custom_path = custom_path.strip('"\'')
                        if self.validate_optimization_folder(custom_path):
                            selected_folder = custom_path
                            break
                        else:
                            print("Please try again with a valid folder path.")
                            continue
                        
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    raise SystemExit(1)
            
            print(f"Selected optimization folder: {selected_folder}")
            return selected_inverter, selected_folder
                    
        except Exception as e:
            logging.error(f"Error during inverter and folder selection: {e}")
            raise
    
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
        Load measured electrical power data for the selected inverter from the pickle file
        """
        # Load the individual inverter electrical data file
        pickle_file = self.data_dir / "full_inv_pow_5min.pkl"
        
        if not pickle_file.exists():
            logging.error(f"Required electrical data file not found: {pickle_file}")
            return None
            
        try:
            logging.info(f"Loading measured electrical data from: {pickle_file}")
            df = pd.read_pickle(pickle_file)
            
            logging.info(f"Raw data shape: {df.shape}")
            logging.info(f"Available inverters: {[col for col in df.columns if col != 'Timestamp']}")
            
            # Check if selected inverter exists in data
            if self.inverter not in df.columns:
                logging.error(f"Selected inverter '{self.inverter}' not found in electrical data")
                logging.error(f"Available inverters: {[col for col in df.columns if col != 'Timestamp']}")
                return None
                
            # Set timestamp as index first
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.set_index('Timestamp', inplace=True)
                logging.info(f"Set Timestamp as index. Date range: {df.index.min()} to {df.index.max()}")
            else:
                logging.error("'Timestamp' column not found in electrical data")
                return None
            
            # Filter to selected inverter only and convert kW to MW (as per notebook logic)
            inverter_df = df[[self.inverter]].copy()
            inverter_df = inverter_df / 1000.0  # Convert kW to MW
            
            # Rename column to 'Power' for consistency with original script
            inverter_df.columns = ['Power']
            
            logging.info(f"Filtered to inverter {self.inverter}. Power range: {inverter_df['Power'].min():.3f} to {inverter_df['Power'].max():.3f} MW")
            logging.info(f"Successfully loaded {len(inverter_df)} measured data points for inverter {self.inverter}")
            
            return inverter_df
            
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
            
            # Apply clipping to raw power values BEFORE energy conversion
            if self.apply_clipping:
                original_max = df['EArray'].max()
                clipped_count = (df['EArray'] > self.clipping_threshold).sum()
                df['EArray'] = df['EArray'].clip(upper=self.clipping_threshold)
                logging.info(f"Applied {self.clipping_threshold} MW clipping to {file_path.name}: {clipped_count} values clipped")
                logging.info(f"Maximum power reduced from {original_max:.2f} MW to {df['EArray'].max():.2f} MW")
            else:
                logging.info(f"No clipping applied to {file_path.name} (clipping disabled)")
            
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
            
            # Extract parameters from filename
            uc_value, uv_value, model_name = self.extract_parameters_from_filename(file_path.name)
            
            logging.info(f"Results for {file_path.name}: Inverter={self.inverter}, Model={model_name}, Uc={uc_value}, Uv={uv_value}, RMSE={rmse:.3f}, MBE={mbe:.6e}, CRMSE={crmse:.3f}, Scale={optimal_factor:.6f}")
            
            return {
                'file directory': str(file_path),
                'inverter_id': self.inverter,
                'model_name': model_name,
                'Uc': uc_value,
                'Uv': uv_value,
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
        
        # Detect transpositional model type for filename
        model_type = self.detect_transpositional_model(results)
        
        # Save results to CSV with inverter and model-specific filename
        inverter_safe = self.inverter.replace("-", "_")
        output_file = self.results_dir / f"pvsyst_batch_evaluation_results_{inverter_safe}_{model_type}.csv"
        results_df[['file directory', 'inverter_id', 'model_name', 'Uc', 'Uv', 'RMSE', 'MBE', 'CRMSE', 'The optimised scale factor']].to_csv(
            output_file, index=False
        )
        
        logging.info(f"Results saved to: {output_file}")
        logging.info(f"Transpositional model type: {model_type}")
        
        # Print summary statistics
        logging.info("\n" + "="*50)
        logging.info("BATCH EVALUATION SUMMARY")
        logging.info("="*50)
        logging.info(f"Files processed successfully: {len(results)}")
        logging.info(f"Average RMSE: {results_df['RMSE'].mean():.3f} ± {results_df['RMSE'].std():.3f}")
        logging.info(f"Average CRMSE: {results_df['CRMSE'].mean():.3f} ± {results_df['CRMSE'].std():.3f}")
        logging.info(f"Average scaling factor: {results_df['The optimised scale factor'].mean():.6f} ± {results_df['The optimised scale factor'].std():.6f}")
        logging.info(f"Average data points per file: {results_df['data_points'].mean():.1f}")
        
        return results_df, str(output_file)

def main():
    """Main function to run the batch evaluation"""
    
    print("PVsyst Batch Evaluation Tool for Individual Inverters")
    print("="*60)
    
    try:
        # Initialize evaluator (will prompt for inverter and optimization folder selection)
        evaluator = PVsystBatchEvaluator()
        
        # Run batch evaluation
        evaluation_result = evaluator.run_batch_evaluation()
        
        if evaluation_result is not None:
            results_df, actual_filename = evaluation_result
            print(f"\nBatch evaluation completed successfully!")
            print(f"Results saved to: {actual_filename}")
        else:
            print("\nBatch evaluation failed!")
            
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        print(f"\nFatal error: {e}")

if __name__ == "__main__":
    main()