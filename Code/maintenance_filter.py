#!/usr/bin/env python3
"""
Maintenance-Free Day Extraction Script

This script analyzes a PV system maintenance logbook (Excel file) and generates
a list of all days in a given year that do not overlap with any maintenance events.

Features:
- Automatic date format detection with multiple parsing strategies
- Flexible year input via command line
- Comprehensive maintenance period filtering (inclusive of start/end dates)
- Robust error handling and validation
- Clean output in standardized YYYY-MM-DD format
- Project structure compliant (works from Code directory)

Usage (run from Code directory):
    python maintenance_filter.py --year 2021
    python maintenance_filter.py --year 2021 --excel_file "../Data/Faults 1.xlsx" --output "../Results/maintenance_free_days_2021.txt"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import re
from pathlib import Path
import os
from dateutil import parser as dateutil_parser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectStructureManager:
    """Manages project directory structure and path resolution."""
    
    def __init__(self):
        """Initialize project structure manager."""
        # Get the directory where this script is located (should be Code/)
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent
        self.data_dir = self.project_root / "Data"
        self.results_dir = self.project_root / "Results"
        
        logger.info(f"Script directory: {self.script_dir}")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def validate_and_setup_directories(self):
        """Validate project structure and create missing directories."""
        logger.info("Validating project directory structure...")
        
        # Check if Data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create Results directory if it doesn't exist
        if not self.results_dir.exists():
            logger.info(f"Creating Results directory: {self.results_dir}")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate write permissions for Results directory
        if not os.access(self.results_dir, os.W_OK):
            raise PermissionError(f"No write permission for Results directory: {self.results_dir}")
        
        logger.info("Directory structure validation completed successfully")
    
    def resolve_data_file(self, relative_path):
        """Resolve data file path relative to project structure."""
        if Path(relative_path).is_absolute():
            return Path(relative_path)
        else:
            # Resolve relative to project root, not script directory
            resolved_path = self.project_root / relative_path
            logger.info(f"Resolved data file path: {resolved_path}")
            return resolved_path
    
    def resolve_output_file(self, relative_path):
        """Resolve output file path relative to project structure."""
        if Path(relative_path).is_absolute():
            return Path(relative_path)
        else:
            # Resolve relative to project root, not script directory
            resolved_path = self.project_root / relative_path
            logger.info(f"Resolved output file path: {resolved_path}")
            return resolved_path


class DateFormatDetector:
    """Advanced date format detection and parsing class."""
    
    def __init__(self):
        self.common_formats = [
            '%Y-%m-%d',     # 2021-03-15
            '%d/%m/%Y',     # 15/03/2021
            '%m/%d/%Y',     # 03/15/2021
            '%d-%m-%Y',     # 15-03-2021
            '%m-%d-%Y',     # 03-15-2021
            '%Y/%m/%d',     # 2021/03/15
            '%d.%m.%Y',     # 15.03.2021
            '%Y.%m.%d',     # 2021.03.15
        ]
    
    def detect_and_parse_dates(self, date_series):
        """
        Detect date format and parse dates from a pandas Series.
        
        Args:
            date_series (pd.Series): Series containing date strings
            
        Returns:
            pd.Series: Series with parsed datetime objects
        """
        logger.info(f"Analyzing {len(date_series)} date entries for format detection...")
        
        # Remove null values for analysis
        non_null_dates = date_series.dropna()
        if len(non_null_dates) == 0:
            raise ValueError("No valid date entries found in the series")
        
        # Strategy 1: Try pandas built-in parser with infer_datetime_format
        try:
            parsed_dates = pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce')
            success_rate = parsed_dates.notna().sum() / len(non_null_dates)
            
            if success_rate > 0.8:  # 80% success rate threshold
                logger.info(f"Pandas auto-detection successful (success rate: {success_rate:.2%})")
                return parsed_dates
        except Exception as e:
            logger.warning(f"Pandas auto-detection failed: {e}")
        
        # Strategy 2: Try dateutil parser (more flexible)
        try:
            def parse_with_dateutil(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                try:
                    return dateutil_parser.parse(str(date_str), dayfirst=True)  # Assume day-first for ambiguous dates
                except:
                    return pd.NaT
            
            parsed_dates = date_series.apply(parse_with_dateutil)
            success_rate = parsed_dates.notna().sum() / len(non_null_dates)
            
            if success_rate > 0.8:
                logger.info(f"Dateutil parsing successful (success rate: {success_rate:.2%})")
                return parsed_dates
        except Exception as e:
            logger.warning(f"Dateutil parsing failed: {e}")
        
        # Strategy 3: Try common formats manually
        for fmt in self.common_formats:
            try:
                parsed_dates = pd.to_datetime(date_series, format=fmt, errors='coerce')
                success_rate = parsed_dates.notna().sum() / len(non_null_dates)
                
                if success_rate > 0.8:
                    logger.info(f"Manual format detection successful with format '{fmt}' (success rate: {success_rate:.2%})")
                    return parsed_dates
            except:
                continue
        
        # Strategy 4: Regex-based custom parsing for complex formats
        sample_dates = non_null_dates.head(5).astype(str).tolist()
        logger.info(f"Sample dates for manual inspection: {sample_dates}")
        
        raise ValueError(f"Unable to detect date format. Sample dates: {sample_dates}")


class MaintenanceFilter:
    """Main class for filtering maintenance-free days from PV system logbook."""
    
    def __init__(self, excel_file_path, project_manager):
        """
        Initialize the maintenance filter.
        
        Args:
            excel_file_path (str or Path): Path to the Excel maintenance logbook
            project_manager (ProjectStructureManager): Project structure manager instance
        """
        self.project_manager = project_manager
        self.excel_file_path = self.project_manager.resolve_data_file(excel_file_path)
        self.date_detector = DateFormatDetector()
        self.maintenance_events = None
        
        if not self.excel_file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file_path}")
    
    def load_maintenance_data(self):
        """Load and parse maintenance data from Excel or CSV file."""
        logger.info(f"Loading maintenance data from: {self.excel_file_path}")
        
        try:
            # Determine file type and read accordingly
            file_extension = self.excel_file_path.suffix.lower()
            
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(self.excel_file_path)
                logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
            elif file_extension == '.csv':
                df = pd.read_csv(self.excel_file_path)
                logger.info(f"Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .xlsx, .xls, .csv")
            
            logger.info(f"Columns: {list(df.columns)}")
            
            # Find date columns (case-insensitive search)
            start_date_col = None
            end_date_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'start' in col_lower and 'date' in col_lower:
                    start_date_col = col
                elif 'end' in col_lower and 'date' in col_lower:
                    end_date_col = col
            
            if start_date_col is None or end_date_col is None:
                logger.warning("Could not automatically detect date columns. Available columns:")
                for i, col in enumerate(df.columns):
                    logger.warning(f"  [{i}]: {col}")
                raise ValueError("Could not find 'Start Date' and 'End Date' columns. Please check column names.")
            
            logger.info(f"Detected date columns: Start='{start_date_col}', End='{end_date_col}'")
            
            # Parse dates using automatic detection
            logger.info("Parsing start dates...")
            start_dates = self.date_detector.detect_and_parse_dates(df[start_date_col])
            
            logger.info("Parsing end dates...")
            end_dates = self.date_detector.detect_and_parse_dates(df[end_date_col])
            
            # Create maintenance events dataframe
            self.maintenance_events = pd.DataFrame({
                'start_date': start_dates,
                'end_date': end_dates,
                'original_row': df.index
            })
            
            # Remove events with invalid dates
            valid_events = self.maintenance_events.dropna(subset=['start_date', 'end_date'])
            invalid_count = len(self.maintenance_events) - len(valid_events)
            
            if invalid_count > 0:
                logger.warning(f"Removed {invalid_count} events with invalid dates")
            
            self.maintenance_events = valid_events.copy()
            logger.info(f"Successfully parsed {len(self.maintenance_events)} maintenance events")
            
            # Display sample events
            if len(self.maintenance_events) > 0:
                logger.info("Sample maintenance events:")
                for i, (_, event) in enumerate(self.maintenance_events.head(3).iterrows()):
                    start_str = event['start_date'].strftime('%Y-%m-%d')
                    end_str = event['end_date'].strftime('%Y-%m-%d')
                    duration = (event['end_date'] - event['start_date']).days + 1
                    logger.info(f"  Event {i+1}: {start_str} to {end_str} ({duration} days)")
            
        except Exception as e:
            logger.error(f"Error loading maintenance data: {e}")
            raise
    
    def generate_year_calendar(self, year):
        """
        Generate all days in a given year.
        
        Args:
            year (int): Target year
            
        Returns:
            list: List of datetime objects for each day in the year
        """
        logger.info(f"Generating calendar for year {year}")
        
        start_date = datetime(year, 1, 1)
        
        # Handle leap years correctly
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_year = 366
            logger.info(f"Year {year} is a leap year (366 days)")
        else:
            days_in_year = 365
            logger.info(f"Year {year} is a regular year (365 days)")
        
        year_days = [start_date + timedelta(days=i) for i in range(days_in_year)]
        return year_days
    
    def filter_maintenance_days(self, year):
        """
        Filter out all days that overlap with maintenance events.
        
        Args:
            year (int): Target year
            
        Returns:
            list: List of datetime objects for maintenance-free days
        """
        if self.maintenance_events is None:
            raise ValueError("Maintenance data not loaded. Call load_maintenance_data() first.")
        
        # Generate all days in the year
        all_days = self.generate_year_calendar(year)
        all_days_set = set(day.date() for day in all_days)
        
        logger.info(f"Starting with {len(all_days_set)} days in year {year}")
        
        # Collect all maintenance days
        maintenance_days = set()
        year_start = datetime(year, 1, 1).date()
        year_end = datetime(year, 12, 31).date()
        
        for _, event in self.maintenance_events.iterrows():
            event_start = event['start_date'].date()
            event_end = event['end_date'].date()
            
            # Only consider events that overlap with the target year
            if event_end >= year_start and event_start <= year_end:
                # Limit the event to the bounds of the target year
                overlap_start = max(event_start, year_start)
                overlap_end = min(event_end, year_end)
                
                # Generate all days in the maintenance period
                current_day = overlap_start
                while current_day <= overlap_end:
                    maintenance_days.add(current_day)
                    current_day += timedelta(days=1)
        
        logger.info(f"Found {len(maintenance_days)} maintenance days in year {year}")
        
        # Remove maintenance days from all days
        maintenance_free_days = all_days_set - maintenance_days
        
        # Convert back to sorted list of datetime objects
        maintenance_free_days = sorted([datetime.combine(day, datetime.min.time()) 
                                      for day in maintenance_free_days])
        
        logger.info(f"Remaining maintenance-free days: {len(maintenance_free_days)}")
        return maintenance_free_days
    
    def export_to_file(self, maintenance_free_days, output_file):
        """
        Export maintenance-free days to a text file.
        
        Args:
            maintenance_free_days (list): List of datetime objects
            output_file (str or Path): Output file path
        """
        output_path = self.project_manager.resolve_output_file(output_file)
        logger.info(f"Exporting {len(maintenance_free_days)} days to: {output_path}")
        
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for day in maintenance_free_days:
                    f.write(day.strftime('%Y-%m-%d') + '\n')
            
            logger.info(f"Successfully exported to {output_path}")
            logger.info(f"File contains {len(maintenance_free_days)} maintenance-free days")
            
        except Exception as e:
            logger.error(f"Error writing to file: {e}")
            raise


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Extract maintenance-free days from PV system logbook')
    parser.add_argument('--year', type=int, required=True,
                       help='Target year to analyze (e.g., 2021)')
    parser.add_argument('--excel_file', type=str, 
                       default='Data/Faults 1.xlsx',
                       help='Path to Excel maintenance logbook (default: Data/Faults 1.xlsx, relative to project root)')
    parser.add_argument('--output', type=str,
                       help='Output text file path (default: Results/maintenance_free_days_{year}.txt)')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output is None:
        args.output = f'Results/maintenance_free_days_{args.year}.txt'
    
    try:
        # Initialize project structure manager
        logger.info("=== Maintenance-Free Day Extraction Started ===")
        project_manager = ProjectStructureManager()
        project_manager.validate_and_setup_directories()
        
        logger.info(f"Target year: {args.year}")
        logger.info(f"Excel file: {args.excel_file}")
        logger.info(f"Output file: {args.output}")
        
        # Initialize maintenance filter
        maintenance_filter = MaintenanceFilter(args.excel_file, project_manager)
        
        # Load and process data
        maintenance_filter.load_maintenance_data()
        maintenance_free_days = maintenance_filter.filter_maintenance_days(args.year)
        maintenance_filter.export_to_file(maintenance_free_days, args.output)
        
        logger.info("=== Processing Completed Successfully ===")
        print(f"\nSUCCESS: Generated {len(maintenance_free_days)} maintenance-free days for year {args.year}")
        print(f"Output saved to: {project_manager.resolve_output_file(args.output)}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()