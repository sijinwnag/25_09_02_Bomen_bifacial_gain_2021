#!/usr/bin/env python3
'''
Maintenance Day Extraction Script

This script analyzes a PV system maintenance logbook (Excel/CSV) and generates
a list of all days in a given year that overlap with maintenance events.

It mirrors the project structure, parsing, and robustness of maintenance_filter.py
but exports only the days WITH maintenance.

Usage (run from Code directory):
    python maintenance_days_filter.py --year 2021
    python maintenance_days_filter.py --year 2021 --excel_file '../Data/Faults 1.xlsx' --output '../Results/maintenance_days_2021.txt'
'''

import argparse
import logging
import sys
from datetime import datetime, timedelta

# Reuse project/path/date parsing utilities from the existing script
from maintenance_filter import ProjectStructureManager, MaintenanceFilter


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaintenanceDaysFilter(MaintenanceFilter):
    '''Filter that returns days WITH maintenance for a given year.'''

    def get_maintenance_days(self, year):
        '''
        Collect all days that overlap with maintenance events for the target year.

        Args:
            year (int): Target year

        Returns:
            list[datetime]: Sorted list of datetime objects representing maintenance days
        '''
        if self.maintenance_events is None:
            raise ValueError('Maintenance data not loaded. Call load_maintenance_data() first.')

        # Generate all days in the year (for bounds and validation)
        _ = self.generate_year_calendar(year)

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

                # Generate all days in the maintenance period (inclusive)
                current_day = overlap_start
                while current_day <= overlap_end:
                    maintenance_days.add(current_day)
                    current_day += timedelta(days=1)

        logger.info(f'Found {len(maintenance_days)} maintenance days in year {year}')

        # Convert back to sorted list of datetime objects
        maintenance_days = sorted(
            [datetime.combine(day, datetime.min.time()) for day in maintenance_days]
        )

        return maintenance_days

    def export_to_file(self, maintenance_days, output_file):
        '''Export maintenance days to a text file in YYYY-MM-DD format.'''
        output_path = self.project_manager.resolve_output_file(output_file)
        logger.info(f'Exporting {len(maintenance_days)} maintenance days to: {output_path}')

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for day in maintenance_days:
                    f.write(day.strftime('%Y-%m-%d') + '\n')
            logger.info(f'Successfully exported to {output_path}')
        except Exception as e:
            logger.error(f'Error writing to file: {e}')
            raise


def main():
    '''CLI entry point for exporting maintenance days.'''
    parser = argparse.ArgumentParser(description='Extract maintenance days (with maintenance) from PV system logbook')
    parser.add_argument('--year', type=int, required=True, help='Target year to analyze (e.g., 2021)')
    parser.add_argument('--excel_file', type=str, default='Data/Faults 1.xlsx',
                        help='Path to maintenance logbook (default: Data/Faults 1.xlsx, relative to project root)')
    parser.add_argument('--output', type=str,
                        help='Output text file path (default: Results/maintenance_days_{year}.txt)')

    args = parser.parse_args()

    # Set default output file if not provided
    if args.output is None:
        args.output = f'Results/maintenance_days_{args.year}.txt'

    try:
        logger.info('=== Maintenance Day Extraction Started ===')
        project_manager = ProjectStructureManager()
        project_manager.validate_and_setup_directories()

        logger.info(f'Target year: {args.year}')
        logger.info(f'Excel file: {args.excel_file}')
        logger.info(f'Output file: {args.output}')

        # Initialize and run filter
        mfilter = MaintenanceDaysFilter(args.excel_file, project_manager)
        mfilter.load_maintenance_data()
        maintenance_days = mfilter.get_maintenance_days(args.year)
        mfilter.export_to_file(maintenance_days, args.output)

        logger.info('=== Processing Completed Successfully ===')
        print(f"\nSUCCESS: Generated {len(maintenance_days)} maintenance days for year {args.year}")
        print(f'Output saved to: {project_manager.resolve_output_file(args.output)}')

    except Exception as e:
        logger.error(f'Processing failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()

