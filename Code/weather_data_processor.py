#!/usr/bin/env python3
"""
Weather Data Processing Script for Bomen Solar Farm

This script processes weather data from three monitoring stations (CP01, CP02, CP03)
at the Bomen Solar Farm with multiple processing options including robust median,
simple averaging, and individual station processing.

Author: Claude Code Assistant
Created: 2025
Version: 1.0

Usage:
    python weather_data_processor.py --year 2021 --method robust_median
    python weather_data_processor.py --year 2022 --method average --output custom.csv
    python weather_data_processor.py --interactive
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    """
    Main class for processing weather data from Bomen Solar Farm monitoring stations.
    
    Processes weather data (air temperature, GHI, POA irradiance) from three stations
    and integrates hourly wind speed data from external CSV files.
    
    Supports multiple processing methods:
    - robust_median: MAD-based outlier detection with median processing
    - average: Simple averaging across all stations
    - CP01, CP02, CP03: Individual station processing
    
    Features:
    - Multi-station weather data processing with quality control
    - Hourly wind speed integration with 5-minute temporal alignment
    - Comprehensive metadata export with data quality statistics
    """
    
    def __init__(self, base_path: str = None, output_path: str = None, wind_data_path: str = None):
        """Initialize the weather data processor."""
        if base_path is None:
            # Default path to the data directory
            self.base_path = Path("C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/PV-syst-data/Bomen/data_from_server")
        else:
            self.base_path = Path(base_path)
        
        # Default output path to Results directory
        if output_path is None:
            self.output_path = Path("C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Results")
        else:
            self.output_path = Path(output_path)
        
        # Default wind speed data path
        if wind_data_path is None:
            self.wind_data_path = Path("C:/Users/z5183876/OneDrive - UNSW/Documents/GitHub/25_09_02_Bomen_bifacial_gain_2021/Data")
        else:
            self.wind_data_path = Path(wind_data_path)
        
        # Station column mapping (0-based indexing)
        self.station_columns = {
            'CP01': {'air_temp': 0, 'ghi': 2, 'poa': 3},
            'CP02': {'air_temp': 4, 'ghi': 6, 'poa': 7}, 
            'CP03': {'air_temp': 8, 'ghi': 10, 'poa': 11}
        }
        
        # Processing methods
        self.available_methods = ['robust_median', 'average', 'CP01', 'CP02', 'CP03']
        
        # Processing statistics
        self.processing_stats = {
            'outliers_removed': {},
            'negative_values_corrected': {},
            'total_records': 0,
            'valid_records': 0
        }

    def load_weather_data(self, year: int) -> pd.DataFrame:
        """
        Load weather data from pickle file for specified year.
        
        Args:
            year: Year to load data for (2020-2023)
            
        Returns:
            pandas.DataFrame: Raw weather data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If data loading fails
        """
        weather_file = self.base_path / f"{year}_weather_df.pkl"
        
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather data file not found: {weather_file}")
        
        logger.info(f"Loading weather data from: {weather_file}")
        
        try:
            with open(weather_file, 'rb') as file:
                data = pickle.load(file)
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Loaded data is not a pandas DataFrame")
                
            logger.info(f"Successfully loaded {len(data)} records")
            self.processing_stats['total_records'] = len(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load weather data: {str(e)}")
            raise ValueError(f"Error loading weather data: {str(e)}")

    def load_wind_speed_data(self, year: int) -> pd.DataFrame:
        """
        Load wind speed data from CSV file for specified year.
        
        Args:
            year: Year to load wind speed data for
            
        Returns:
            pandas.DataFrame: Wind speed data with datetime index and WS10M column
            
        Raises:
            FileNotFoundError: If the wind speed data file doesn't exist
            ValueError: If data loading fails
        """
        # Construct wind speed file path based on year
        wind_file = self.wind_data_path / f"Bomen_Hourly_{year}0101_{year}1231_WS_LST.csv"
        
        if not wind_file.exists():
            raise FileNotFoundError(f"Wind speed data file not found: {wind_file}")
        
        logger.info(f"Loading wind speed data from: {wind_file}")
        
        try:
            # Load CSV file
            wind_data = pd.read_csv(wind_file)
            
            # Validate expected columns
            expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'WS10M']
            missing_cols = set(expected_cols) - set(wind_data.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in wind speed data: {missing_cols}")
            
            # Create datetime index from YEAR, MO, DY, HR columns
            wind_data['datetime'] = pd.to_datetime(
                wind_data[['YEAR', 'MO', 'DY', 'HR']].rename(columns={
                    'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'
                })
            )
            
            # Set datetime as index and keep only wind speed column
            wind_data = wind_data.set_index('datetime')[['WS10M']]
            
            # Basic data quality checks
            initial_count = len(wind_data)
            wind_data = wind_data.dropna()  # Remove any NaN values
            wind_data = wind_data[wind_data['WS10M'] >= 0]  # Remove negative wind speeds
            final_count = len(wind_data)
            
            if final_count < initial_count:
                logger.warning(f"Removed {initial_count - final_count} invalid wind speed records")
            
            logger.info(f"Successfully loaded {len(wind_data)} wind speed records")
            
            return wind_data
            
        except Exception as e:
            logger.error(f"Failed to load wind speed data: {str(e)}")
            raise ValueError(f"Error loading wind speed data: {str(e)}")

    def merge_wind_speed_data(self, weather_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge hourly wind speed data with 5-minute weather data.
        
        Uses forward-fill method to assign hourly wind speeds to all 5-minute intervals
        within each hour, preserving the 5-minute temporal resolution.
        
        Args:
            weather_df: Weather dataframe with 5-minute resolution
            wind_df: Wind speed dataframe with hourly resolution
            
        Returns:
            pandas.DataFrame: Weather dataframe with added wind speed column
        """
        logger.info("Merging wind speed data with weather data...")
        
        # Make a copy to avoid modifying the original dataframe
        merged_df = weather_df.copy()
        
        # Resample hourly wind data to 5-minute intervals using forward-fill
        # This assigns the hourly wind speed value to all 5-minute intervals within that hour
        wind_5min = wind_df.reindex(weather_df.index, method='ffill')
        
        # Add wind speed column to the weather dataframe
        merged_df['Wind Speed'] = wind_5min['WS10M']
        
        # Count how many records have valid wind speed data
        valid_wind_count = merged_df['Wind Speed'].notna().sum()
        total_records = len(merged_df)
        
        logger.info(f"Wind speed data coverage: {valid_wind_count}/{total_records} records "
                   f"({valid_wind_count/total_records*100:.1f}%)")
        
        # Handle missing wind speed data by forward-filling within reasonable limits
        if merged_df['Wind Speed'].isna().any():
            missing_before = merged_df['Wind Speed'].isna().sum()
            # Forward-fill with limit of 12 intervals (1 hour gap tolerance)
            merged_df['Wind Speed'] = merged_df['Wind Speed'].fillna(method='ffill', limit=12)
            
            missing_after = merged_df['Wind Speed'].isna().sum()
            if missing_after < missing_before:
                filled_count = missing_before - missing_after
                logger.info(f"Forward-filled {filled_count} missing wind speed values")
            
            if missing_after > 0:
                logger.warning(f"{missing_after} wind speed values remain missing after gap-filling")
        
        return merged_df

    def clean_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning by removing NaN values and formatting timestamps.
        
        Args:
            df: Raw weather dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe with proper datetime index
        """
        logger.info("Performing initial data cleaning...")
        
        # Check for NaN values
        initial_count = len(df)
        nan_count = df.isna().any(axis=1).sum()
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} rows with NaN values, removing them")
            df = df.dropna()
        
        # Convert timestamp format
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')
            except:
                logger.warning("Failed to parse timestamps with default format, trying automatic parsing")
                df.index = pd.to_datetime(df.index)
        
        # Add time component columns
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        
        final_count = len(df)
        self.processing_stats['valid_records'] = final_count
        
        logger.info(f"Data cleaning complete: {initial_count} → {final_count} records")
        
        return df

    def robust_median_processing(self, df: pd.DataFrame, parameter: str, 
                                columns: List[int]) -> pd.Series:
        """
        Process weather parameter using robust median with MAD outlier detection.
        
        This method replicates the exact logic from the original notebook:
        1. Extract data from three stations
        2. Calculate median and MAD (Median Absolute Deviation)  
        3. Remove outliers beyond 1.5 × MAD threshold
        4. Calculate average of remaining valid values
        
        Args:
            df: Input dataframe
            parameter: Parameter name ('air_temp', 'ghi', 'poa')
            columns: List of column indices for the three stations
            
        Returns:
            pandas.Series: Processed parameter values
        """
        logger.debug(f"Processing {parameter} using robust median method")
        
        # Extract data from three stations
        station_data = pd.DataFrame({
            f'{parameter}_1': df.iloc[:, columns[0]],
            f'{parameter}_2': df.iloc[:, columns[1]], 
            f'{parameter}_3': df.iloc[:, columns[2]]
        })
        
        # Calculate median across stations
        median_values = station_data.median(axis=1)
        
        # Calculate Median Absolute Deviation (MAD)
        mad_values = (station_data.sub(median_values, axis=0)).abs().median(axis=1)
        
        # Define outlier threshold (1.5 × MAD)
        threshold = 1.5 * mad_values
        
        # Remove outliers by setting them to NaN
        outliers_removed = 0
        for col in station_data.columns:
            outlier_mask = (station_data[col] - median_values).abs() > threshold
            outliers_before = outlier_mask.sum()
            station_data.loc[outlier_mask, col] = np.nan
            outliers_removed += outliers_before
        
        # Calculate mean of remaining valid values
        processed_values = station_data.mean(axis=1)
        
        # Handle physical constraints for all parameters
        if parameter == 'air_temp':
            # Air temperature: clip to reasonable bounds (≥ -50°C for extreme cold)
            invalid_count = (processed_values < -50).sum()
            processed_values = processed_values.clip(lower=-50)
            self.processing_stats['negative_values_corrected'][parameter] = invalid_count
            if invalid_count > 0:
                logger.debug(f"Corrected {invalid_count} extreme {parameter} values (< -50°C)")
        elif parameter in ['ghi', 'poa']:
            # Irradiance parameters: clip to ≥ 0 (physical constraint)
            negative_count = (processed_values < 0).sum()
            processed_values = processed_values.clip(lower=0)
            self.processing_stats['negative_values_corrected'][parameter] = negative_count
            if negative_count > 0:
                logger.debug(f"Corrected {negative_count} negative {parameter} values to zero")
        
        self.processing_stats['outliers_removed'][parameter] = outliers_removed
        logger.debug(f"Removed {outliers_removed} outliers from {parameter}")
        
        return processed_values

    def simple_average_processing(self, df: pd.DataFrame, parameter: str, 
                                 columns: List[int]) -> pd.Series:
        """
        Process weather parameter using simple averaging across all stations.
        
        Args:
            df: Input dataframe
            parameter: Parameter name ('air_temp', 'ghi', 'poa') 
            columns: List of column indices for the three stations
            
        Returns:
            pandas.Series: Processed parameter values
        """
        logger.debug(f"Processing {parameter} using simple average method")
        
        # Extract data from three stations
        station_data = pd.DataFrame({
            f'{parameter}_1': df.iloc[:, columns[0]],
            f'{parameter}_2': df.iloc[:, columns[1]],
            f'{parameter}_3': df.iloc[:, columns[2]]
        })
        
        # Calculate simple average
        processed_values = station_data.mean(axis=1)
        
        # Handle physical constraints for all parameters
        if parameter == 'air_temp':
            # Air temperature: clip to reasonable bounds (≥ -50°C for extreme cold)
            invalid_count = (processed_values < -50).sum()
            processed_values = processed_values.clip(lower=-50)
            self.processing_stats['negative_values_corrected'][parameter] = invalid_count
            if invalid_count > 0:
                logger.debug(f"Corrected {invalid_count} extreme {parameter} values (< -50°C)")
        elif parameter in ['ghi', 'poa']:
            # Irradiance parameters: clip to ≥ 0 (physical constraint)
            negative_count = (processed_values < 0).sum()
            processed_values = processed_values.clip(lower=0)
            self.processing_stats['negative_values_corrected'][parameter] = negative_count
            if negative_count > 0:
                logger.debug(f"Corrected {negative_count} negative {parameter} values to zero")
        
        return processed_values

    def single_station_processing(self, df: pd.DataFrame, parameter: str, 
                                 station: str) -> pd.Series:
        """
        Process weather parameter from a single station.
        
        Args:
            df: Input dataframe
            parameter: Parameter name ('air_temp', 'ghi', 'poa')
            station: Station name ('CP01', 'CP02', 'CP03')
            
        Returns:
            pandas.Series: Processed parameter values
        """
        logger.debug(f"Processing {parameter} from station {station}")
        
        column_index = self.station_columns[station][parameter]
        processed_values = df.iloc[:, column_index]
        
        # Handle physical constraints for all parameters
        if parameter == 'air_temp':
            # Air temperature: clip to reasonable bounds (≥ -50°C for extreme cold)
            invalid_count = (processed_values < -50).sum()
            processed_values = processed_values.clip(lower=-50)
            self.processing_stats['negative_values_corrected'][parameter] = invalid_count
            if invalid_count > 0:
                logger.debug(f"Corrected {invalid_count} extreme {parameter} values (< -50°C)")
        elif parameter in ['ghi', 'poa']:
            # Irradiance parameters: clip to ≥ 0 (physical constraint)
            negative_count = (processed_values < 0).sum()
            processed_values = processed_values.clip(lower=0)
            self.processing_stats['negative_values_corrected'][parameter] = negative_count
            if negative_count > 0:
                logger.debug(f"Corrected {negative_count} negative {parameter} values to zero")
        
        return processed_values

    def process_weather_data(self, year: int, method: str = 'robust_median') -> pd.DataFrame:
        """
        Main processing function that coordinates data loading and processing.
        
        Args:
            year: Year to process (2020-2023)
            method: Processing method ('robust_median', 'average', 'CP01', 'CP02', 'CP03')
            
        Returns:
            pandas.DataFrame: Processed weather data ready for export
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not supported. Available: {self.available_methods}")
        
        logger.info(f"Starting weather data processing for {year} using {method} method")
        
        # Load and clean data
        raw_data = self.load_weather_data(year)
        df = self.clean_initial_data(raw_data)
        
        # Process each parameter based on selected method
        if method == 'robust_median':
            # Process using MAD-based robust median (original notebook method)
            air_temp = self.robust_median_processing(df, 'air_temp', [0, 4, 8])
            ghi = self.robust_median_processing(df, 'ghi', [2, 6, 10]) 
            poa = self.robust_median_processing(df, 'poa', [3, 7, 11])
            
        elif method == 'average':
            # Process using simple averaging
            air_temp = self.simple_average_processing(df, 'air_temp', [0, 4, 8])
            ghi = self.simple_average_processing(df, 'ghi', [2, 6, 10])
            poa = self.simple_average_processing(df, 'poa', [3, 7, 11])
            
        elif method in ['CP01', 'CP02', 'CP03']:
            # Process using single station
            air_temp = self.single_station_processing(df, 'air_temp', method)
            ghi = self.single_station_processing(df, 'ghi', method)
            poa = self.single_station_processing(df, 'poa', method)
        
        # Add processed values to dataframe with method-specific column names
        if method in ['CP01', 'CP02', 'CP03']:
            # Single station naming
            df[f'{method} Air Temp'] = air_temp
            df[f'{method} GHI'] = ghi
            df[f'{method} POA'] = poa
        else:
            # Averaged methods naming (robust_median, average)
            df['Avg Air Temp'] = air_temp
            df['Avg GHI'] = ghi
            df['Avg POA'] = poa
        
        # Load and merge wind speed data
        try:
            wind_data = self.load_wind_speed_data(year)
            df = self.merge_wind_speed_data(df, wind_data)
            logger.info("Wind speed data successfully integrated")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Wind speed data not available: {str(e)}")
            # Add NaN wind speed column if data is not available
            df['Wind Speed'] = np.nan
            logger.info("Added empty wind speed column (data not available)")
        
        logger.info(f"Processing complete using {method} method")
        
        return df

    def export_processed_data(self, df: pd.DataFrame, output_path: str, 
                            year: int, method: str) -> None:
        """
        Export processed data to CSV with metadata header.
        
        Args:
            df: Processed dataframe
            output_path: Output file path
            year: Year processed
            method: Processing method used
        """
        logger.info(f"Exporting processed data to: {output_path}")
        
        # Create metadata header as comments
        metadata_lines = [
            f"# Bomen Solar Farm Weather Data - {year}",
            f"# Processing Method: {method}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total Records: {self.processing_stats['total_records']}",
            f"# Valid Records: {self.processing_stats['valid_records']}",
        ]
        
        # Add processing statistics
        if self.processing_stats['outliers_removed']:
            metadata_lines.append("# Outliers Removed:")
            for param, count in self.processing_stats['outliers_removed'].items():
                metadata_lines.append(f"#   {param}: {count}")
        
        if self.processing_stats['negative_values_corrected']:
            metadata_lines.append("# Negative Values Corrected:")
            for param, count in self.processing_stats['negative_values_corrected'].items():
                metadata_lines.append(f"#   {param}: {count}")
        
        # Add wind speed data information
        if 'Wind Speed' in df.columns:
            wind_valid_count = df['Wind Speed'].notna().sum()
            wind_total_count = len(df)
            wind_coverage = (wind_valid_count / wind_total_count) * 100
            
            metadata_lines.append("# Wind Speed Data:")
            metadata_lines.append(f"#   Source: Bomen_Hourly_{year}0101_{year}1231_WS_LST.csv")
            metadata_lines.append(f"#   Resolution: Hourly data resampled to 5-minute intervals")
            metadata_lines.append(f"#   Coverage: {wind_valid_count:,}/{wind_total_count:,} records ({wind_coverage:.1f}%)")
            if wind_coverage < 100:
                wind_missing_count = wind_total_count - wind_valid_count
                metadata_lines.append(f"#   Missing: {wind_missing_count:,} records")
        
        # Write metadata header
        with open(output_path, 'w') as f:
            for line in metadata_lines:
                f.write(line + '\n')
        
        # Append data (header=True for first append)
        df.to_csv(output_path, mode='a', index=True)
        
        logger.info(f"Successfully exported {len(df)} records")

    def print_processing_summary(self, method: str, year: int) -> None:
        """Print a summary of the processing results."""
        print("\n" + "="*60)
        print(f"WEATHER DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Year: {year}")
        print(f"Method: {method}")
        print(f"Total Records: {self.processing_stats['total_records']:,}")
        print(f"Valid Records: {self.processing_stats['valid_records']:,}")
        
        if self.processing_stats['outliers_removed']:
            print("\nOutliers Removed:")
            for param, count in self.processing_stats['outliers_removed'].items():
                percentage = (count / self.processing_stats['valid_records']) * 100
                print(f"  {param.upper()}: {count:,} ({percentage:.2f}%)")
        
        if self.processing_stats['negative_values_corrected']:
            print("\nNegative Values Corrected:")
            for param, count in self.processing_stats['negative_values_corrected'].items():
                percentage = (count / self.processing_stats['valid_records']) * 100
                print(f"  {param.upper()}: {count:,} ({percentage:.2f}%)")
        
        print("="*60)


def interactive_mode():
    """Run the script in interactive mode with user prompts."""
    print("="*60)
    print("BOMEN SOLAR FARM - WEATHER DATA PROCESSOR")
    print("="*60)
    print("Available years: 2020, 2021, 2022, 2023")
    print("Available methods:")
    print("  1. robust_median - MAD-based outlier detection (recommended)")
    print("  2. average - Simple averaging across all stations") 
    print("  3. CP01 - Process only CP01 station data")
    print("  4. CP02 - Process only CP02 station data")
    print("  5. CP03 - Process only CP03 station data")
    print("-"*60)
    
    # Get year
    while True:
        try:
            year = int(input("Enter year (2020-2023): "))
            if year in [2020, 2021, 2022, 2023]:
                break
            else:
                print("Please enter a valid year (2020-2023)")
        except ValueError:
            print("Please enter a valid integer year")
    
    # Get method
    method_map = {
        '1': 'robust_median',
        '2': 'average', 
        '3': 'CP01',
        '4': 'CP02',
        '5': 'CP03'
    }
    
    while True:
        choice = input("Select processing method (1-5): ").strip()
        if choice in method_map:
            method = method_map[choice]
            break
        else:
            print("Please enter a valid choice (1-5)")
    
    # Get output filename
    default_output = f"{year}_weather_processed_{method}.csv"
    output = input(f"Output filename (default: {default_output}): ").strip()
    if not output:
        output = default_output
    
    return year, method, output


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process weather data from Bomen Solar Farm monitoring stations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python weather_data_processor.py --year 2021 --method robust_median
  python weather_data_processor.py --year 2022 --method average --output custom.csv
  python weather_data_processor.py --interactive
  python weather_data_processor.py --help
        """
    )
    
    parser.add_argument('--year', type=int, choices=[2020, 2021, 2022, 2023],
                       help='Year to process (2020-2023)')
    parser.add_argument('--method', choices=['robust_median', 'average', 'CP01', 'CP02', 'CP03'],
                       help='Processing method')
    parser.add_argument('--output', type=str,
                       help='Output CSV filename')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode with prompts')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Interactive mode
        if args.interactive or (not args.year and not args.method):
            year, method, output_file = interactive_mode()
        else:
            # Command line mode
            if not args.year or not args.method:
                parser.error("--year and --method are required unless using --interactive")
            
            year = args.year
            method = args.method
            output_file = args.output or f"{year}_weather_processed_{method}.csv"
        
        # Initialize processor and run
        processor = WeatherDataProcessor()
        processed_data = processor.process_weather_data(year, method)
        
        # Export results to Results directory
        if not os.path.isabs(output_file):
            # If filename only (no path), use Results directory
            output_path = processor.output_path / output_file
        else:
            # If absolute path provided, use as-is
            output_path = Path(output_file)
        
        # Ensure Results directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        processor.export_processed_data(processed_data, str(output_path), year, method)
        
        # Print summary
        processor.print_processing_summary(method, year)
        
        print(f"\nProcessed data exported to: {output_path}")
        print("Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()