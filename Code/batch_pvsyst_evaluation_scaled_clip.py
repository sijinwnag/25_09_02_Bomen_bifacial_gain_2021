#!/usr/bin/env python3
"""
Batch PVsyst evaluation for individual inverters with scale-then-clip workflow.

This script extends the evaluation logic in batch_pvsyst_evaluation_inv.py by
optimising the inverter scaling factor before applying clipping, ensuring the
resulting scaled-and-clipped PVsyst simulations achieve negligible MBE against
measured inverter power data.
"""

import logging
import re
from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_evaluation_scaled_clip.log"),
        logging.StreamHandler()
    ]
)


class PVsystScaledClipBatchEvaluator:
    """Batch processor that evaluates PVsyst per-inverter simulations using a scale-then-clip workflow."""

    def __init__(self, project_root=None, inverter=None, optimization_folder=None,
                 apply_clipping=None, clipping_threshold=None):
        if project_root is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent

        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "Data"
        self.results_dir = self.project_root / "Results"

        if inverter is None or optimization_folder is None:
            self.inverter, self.optimization_folder = self.select_inverter_and_folder()
        else:
            self.inverter = inverter
            self.optimization_folder = optimization_folder

        if apply_clipping is None or clipping_threshold is None:
            self.apply_clipping, self.clipping_threshold = self.configure_power_clipping()
        else:
            self.apply_clipping = apply_clipping
            self.clipping_threshold = clipping_threshold

        if self.optimization_folder:
            self.pvsyst_dir = Path(self.optimization_folder)
        else:
            self.pvsyst_dir = self.data_dir / "PVsyst" / "per_inv" / self.inverter

        self.results_dir.mkdir(exist_ok=True)

        self.target_mbe_tolerance = 1e-13
        self.max_iterations = 100

        self._setup_logging()

        logging.info(f"Project root: {self.project_root}")
        logging.info(f"Selected inverter: {self.inverter}")
        logging.info(f"Optimization folder: {self.optimization_folder}")
        logging.info(f"PVsyst directory: {self.pvsyst_dir}")
        logging.info(f"Power clipping enabled: {self.apply_clipping}")
        if self.apply_clipping:
            logging.info(f"Clipping threshold (MW): {self.clipping_threshold}")

    def _setup_logging(self):
        """Setup inverter-specific logging handlers."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        safe_inverter = self.inverter.replace("-", "_") if self.inverter else "unknown"
        log_file = f"batch_evaluation_scaled_clip_{safe_inverter}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )

    def configure_power_clipping(self):
        """Interactively configure power clipping behaviour."""
        print("\n" + "=" * 60)
        print("POWER CLIPPING CONFIGURATION")
        print("=" * 60)
        print("Power clipping limits simulation power values to prevent unrealistic peaks.")
        print("Scaling is performed first, followed by clipping before energy aggregation.")

        while True:
            try:
                clipping_choice = input("\nApply power clipping to simulation data? (y/n): ").strip().lower()

                if clipping_choice in {"y", "yes"}:
                    apply_clipping = True
                    break
                if clipping_choice in {"n", "no"}:
                    print("Power clipping disabled.")
                    return False, None

                print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                raise SystemExit(1)

        clipping_threshold = None
        while True:
            try:
                method_choice = input(
                    "\nChoose clipping threshold method:\n"
                    "  1. Automatic (lookup from CSV)\n"
                    "  2. Manual (enter value)\n"
                    "Enter choice (1/2): "
                ).strip()

                if method_choice == "1":
                    print("Attempting automatic lookup from inverter clipping analysis CSV...")
                    clipping_threshold = self.lookup_clipping_limit_from_csv()

                    if clipping_threshold is not None:
                        print(f"Using clipping limit: {clipping_threshold} MW")
                        break

                    print("Automatic lookup failed. Falling back to manual input.")
                    method_choice = "2"

                if method_choice == "2":
                    threshold_input = input("Enter clipping threshold in MW (default 2.392): ").strip()

                    if not threshold_input:
                        clipping_threshold = 2.392
                        break

                    clipping_threshold = float(threshold_input)
                    if clipping_threshold <= 0:
                        print("Clipping threshold must be a positive number. Please try again.")
                        continue
                    break

                if method_choice not in {"1", "2"}:
                    print("Please enter '1' for automatic or '2' for manual.")
            except ValueError:
                print("Please enter a valid number for the clipping threshold.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                raise SystemExit(1)

        print(f"Power clipping enabled with threshold: {clipping_threshold} MW")
        return apply_clipping, clipping_threshold

    def lookup_clipping_limit_from_csv(self):
        """Lookup clipping limit for selected inverter from CSV file."""
        try:
            csv_inverter_name = f"INV_{self.inverter}"
            csv_file = self.results_dir / "inverter_clipping_analysis_2021.csv"

            if not csv_file.exists():
                logging.warning(f"Clipping analysis CSV not found: {csv_file}")
                return None

            df = pd.read_csv(csv_file)

            if "inverter_name" not in df.columns or "clipping_limit" not in df.columns:
                logging.warning("Required columns not present in clipping analysis CSV.")
                return None

            matching_row = df[df["inverter_name"] == csv_inverter_name]

            if matching_row.empty:
                logging.warning(f"Inverter {csv_inverter_name} not found in clipping analysis CSV.")
                return None

            clipping_limit = matching_row.iloc[0]["clipping_limit"]

            if clipping_limit > 100:
                clipping_threshold = clipping_limit / 1000.0
                logging.info(f"Converted clipping limit from {clipping_limit} kW to {clipping_threshold} MW")
            else:
                clipping_threshold = clipping_limit
                logging.info(f"Using clipping limit: {clipping_threshold} MW")

            return clipping_threshold

        except Exception as exc:
            logging.error(f"Error looking up clipping limit from CSV: {exc}")
            return None

    def validate_optimization_folder(self, folder_path):
        """Validate optimisation folder exists and contains CSV files."""
        folder_path = Path(folder_path)

        if not folder_path.exists():
            logging.error(f"Optimization folder does not exist: {folder_path}")
            return False

        if not folder_path.is_dir():
            logging.error(f"Path is not a directory: {folder_path}")
            return False

        csv_files = list(folder_path.glob("*.CSV")) + list(folder_path.glob("*.csv"))
        if not csv_files:
            logging.error(f"No CSV files found in optimization folder: {folder_path}")
            return False

        logging.info(f"Found {len(csv_files)} CSV files in optimization folder: {folder_path}")
        return True

    def extract_parameters_from_filename(self, filename):
        """Extract Uc, Uv values and model name from PVsyst filename."""
        try:
            pattern = r"Bomen solar farm 2021 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\w+) model\.csv"

            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                uc_str, uv_str, model_name = match.groups()
                uc_value = float(uc_str) if "." in uc_str else int(uc_str)
                uv_value = float(uv_str) if "." in uv_str else int(uv_str)
                logging.debug(f"Extracted from {filename}: Uc={uc_value}, Uv={uv_value}, Model={model_name}")
                return uc_value, uv_value, model_name

            logging.warning(f"Could not extract parameters from filename: {filename}")
            return None, None, None

        except Exception as exc:
            logging.error(f"Error extracting parameters from filename {filename}: {exc}")
            return None, None, None

    def detect_transpositional_model(self, results):
        """Detect transpositional model type from folder path or results."""
        try:
            if hasattr(self, "optimization_folder") and self.optimization_folder:
                folder_name = Path(self.optimization_folder).name.lower()
                if folder_name in ["perez", "hay", "isotropic"]:
                    model_name = folder_name.capitalize()
                    logging.info(f"Model type detected from folder path: {model_name}")
                    return model_name

            model_names = [r.get("model_name") for r in results if r and r.get("model_name")]
            if model_names:
                model_counts = Counter(model_names)
                most_common_model = model_counts.most_common(1)[0][0]

                if len(model_counts) > 1:
                    logging.info(f"Multiple models detected: {dict(model_counts)}. Using most common: {most_common_model}")
                else:
                    logging.info(f"Model type detected from filenames: {most_common_model}")

                return most_common_model

            logging.warning("No model type could be detected from folder path or filenames.")
            return "Unknown"

        except Exception as exc:
            logging.error(f"Error detecting transpositional model: {exc}")
            return "Unknown"

    def select_inverter_and_folder(self):
        """Prompt user to select an inverter and optimization folder from available options."""
        electrical_data_file = self.data_dir / "full_inv_pow_5min.pkl"

        if not electrical_data_file.exists():
            raise FileNotFoundError(f"Electrical data file not found: {electrical_data_file}")

        try:
            df = pd.read_pickle(electrical_data_file)
            available_inverters = [col for col in df.columns if col != "Timestamp"]
            available_inverters.sort()

            print("Available inverters in electrical data:")
            for index, inverter in enumerate(available_inverters, 1):
                print(f"  {index:2d}. {inverter}")

            per_inv_dir = self.data_dir / "PVsyst" / "per_inv"
            if per_inv_dir.exists():
                pvsyst_inverters = [d.name for d in per_inv_dir.iterdir() if d.is_dir()]
                pvsyst_inverters.sort()

                print("\nInverters with PVsyst simulation files:")
                for inverter in pvsyst_inverters:
                    marker = "[both]" if inverter in available_inverters else "[sim-only]"
                    print(f"  {marker} {inverter}")

                valid_inverters = [inv for inv in available_inverters if inv in pvsyst_inverters]

                if not valid_inverters:
                    raise ValueError("No inverters found with both electrical data and PVsyst simulation files")

                print(f"\nValid inverters (both datasets available): {len(valid_inverters)}")
                for inverter in valid_inverters:
                    print(f"  - {inverter}")
            else:
                valid_inverters = available_inverters
                print("\nPVsyst per_inv directory not found. Using all available inverters from electrical data.")

            selected_inverter = None
            while True:
                try:
                    selection = input("\nSelect inverter to analyze (enter inverter ID, e.g., '2-1'): ").strip()

                    if selection in valid_inverters:
                        print(f"Selected inverter: {selection}")
                        selected_inverter = selection
                        break

                    print(f"Invalid selection. Please choose from: {', '.join(valid_inverters)}")

                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    raise SystemExit(1)

            print("\n" + "=" * 60)
            print("OPTIMIZATION FOLDER SELECTION")
            print("=" * 60)

            available_subfolders = []
            default_inv_dir = per_inv_dir / selected_inverter if per_inv_dir.exists() else None

            if default_inv_dir and default_inv_dir.exists():
                subfolders = [d for d in default_inv_dir.iterdir() if d.is_dir()]
                if subfolders:
                    print(f"\nAvailable optimization folders for inverter {selected_inverter}:")
                    for i, subfolder in enumerate(subfolders, 1):
                        csv_count = len(list(subfolder.glob("*.CSV")) + list(subfolder.glob("*.csv")))
                        print(f"  {i}. {subfolder.name} ({csv_count} CSV files)")
                        available_subfolders.append(subfolder)

                    print(f"  {len(subfolders) + 1}. Custom folder (enter full path)")

            selected_folder = None
            while True:
                try:
                    if available_subfolders:
                        choice = input(f"\nSelect optimization folder (1-{len(available_subfolders) + 1}) or enter full path: ").strip()

                        try:
                            choice_num = int(choice)
                            if 1 <= choice_num <= len(available_subfolders):
                                selected_folder = str(available_subfolders[choice_num - 1])
                                break
                            if choice_num == len(available_subfolders) + 1:
                                custom_path = input("Enter full path to optimization folder: ").strip().strip('\"\'')
                                if self.validate_optimization_folder(custom_path):
                                    selected_folder = custom_path
                                    break
                                print("Please try again with a valid folder path.")
                                continue
                            print(f"Invalid selection. Please choose 1-{len(available_subfolders) + 1} or enter a full path.")
                            continue
                        except ValueError:
                            choice_path = choice.strip('\"\'')
                            if self.validate_optimization_folder(choice_path):
                                selected_folder = choice_path
                                break
                            print("Please try again with a valid folder path.")
                            continue
                    else:
                        custom_path = input("Enter full path to optimization folder: ").strip().strip('\"\'')
                        if self.validate_optimization_folder(custom_path):
                            selected_folder = custom_path
                            break
                        print("Please try again with a valid folder path.")
                        continue

                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    raise SystemExit(1)

            print(f"Selected optimization folder: {selected_folder}")
            return selected_inverter, selected_folder

        except Exception as exc:
            logging.error(f"Error during inverter and folder selection: {exc}")
            raise

    def load_maintenance_free_days(self):
        """Load maintenance-free days from text file."""
        maintenance_file = self.results_dir / "remaining_dates_2021.txt"

        if not maintenance_file.exists():
            logging.warning(f"Maintenance file not found: {maintenance_file}")
            return None

        try:
            with maintenance_file.open("r", encoding="utf-8") as file_handle:
                maintenance_free_days = [line.strip() for line in file_handle if line.strip()]

            maintenance_free_dates = pd.to_datetime(maintenance_free_days)
            logging.info(f"Loaded {len(maintenance_free_dates)} maintenance-free days")
            return maintenance_free_dates

        except Exception as exc:
            logging.error(f"Error loading maintenance-free days: {exc}")
            return None

    def load_measured_power_data(self):
        """Load measured electrical power data for the selected inverter from the pickle file."""
        pickle_file = self.data_dir / "full_inv_pow_5min.pkl"

        if not pickle_file.exists():
            logging.error(f"Required electrical data file not found: {pickle_file}")
            return None

        try:
            logging.info(f"Loading measured electrical data from: {pickle_file}")
            df = pd.read_pickle(pickle_file)

            logging.info(f"Raw data shape: {df.shape}")
            logging.info(f"Available inverters: {[col for col in df.columns if col != 'Timestamp']}")

            if self.inverter not in df.columns:
                logging.error(f"Selected inverter '{self.inverter}' not found in electrical data")
                logging.error(f"Available inverters: {[col for col in df.columns if col != 'Timestamp']}")
                return None

            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df.set_index("Timestamp", inplace=True)
                logging.info(f"Set Timestamp as index. Date range: {df.index.min()} to {df.index.max()}")
            else:
                logging.error("'Timestamp' column not found in electrical data")
                return None

            inverter_df = df[[self.inverter]].copy()
            inverter_df = inverter_df / 1000.0
            inverter_df.columns = ["Power"]

            logging.info(
                f"Filtered to inverter {self.inverter}. Power range: "
                f"{inverter_df['Power'].min():.3f} to {inverter_df['Power'].max():.3f} MW"
            )
            logging.info(f"Successfully loaded {len(inverter_df)} measured data points for inverter {self.inverter}")

            return inverter_df

        except Exception as exc:
            logging.error(f"Error loading electrical data file: {exc}")
            return None

    def load_pvsyst_csv(self, file_path):
        """Load and parse a single PVsyst CSV file."""
        try:
            logging.info(f"Loading PVsyst file: {file_path.name}")

            df = pd.read_csv(
                file_path,
                delimiter=";",
                skiprows=list(range(10)) + [11],
                header=0,
                encoding="latin-1",
                low_memory=False,
                na_values=["", " ", "nan", "NaN"]
            )

            date_col = df.columns[0]

            df["timestamp"] = pd.to_datetime(
                df[date_col],
                format="%d/%m/%y %H:%M",
                errors="coerce"
            )

            if df["timestamp"].isna().any():
                df["timestamp"] = pd.to_datetime(
                    df[date_col],
                    dayfirst=True,
                    errors="coerce"
                )

            df = df.drop(columns=[date_col])
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

            if "EArray" not in df.columns:
                logging.error(f"EArray column not found in {file_path.name}")
                return None

            if pd.api.types.is_object_dtype(df["EArray"]):
                df["EArray"] = pd.to_numeric(df["EArray"].str.replace(",", "."), errors="coerce")

            df.set_index("timestamp", inplace=True)

            logging.info(f"Successfully loaded {len(df)} data points from {file_path.name}")
            return df

        except Exception as exc:
            logging.error(f"Error loading {file_path.name}: {exc}")
            return None

    def compute_measured_daily_energy(self, measured_df):
        """Convert measured 5-minute power data to daily energy totals."""
        measured_copy = measured_df.copy()
        measured_copy["Energy_MWh"] = measured_copy["Power"] * (5 / 60)
        daily_actual_energy = measured_copy["Energy_MWh"].resample("D").sum()

        logging.info(f"Resampled measured data to daily totals: {len(daily_actual_energy)} days")
        logging.info(
            f"Measured daily energy range: {daily_actual_energy.min():.2f} to {daily_actual_energy.max():.2f} MWh/day"
        )
        return daily_actual_energy

    def _scale_and_clip_series(self, simulation_df, scale_factor):
        """Scale and optionally clip the simulation series."""
        scaled_series = simulation_df["EArray"] * scale_factor
        if self.apply_clipping and self.clipping_threshold is not None:
            clipped_series = scaled_series.clip(upper=self.clipping_threshold)
        else:
            clipped_series = scaled_series
        return scaled_series, clipped_series

    def compute_daily_simulation_energy(self, simulation_df, scale_factor):
        """Compute daily energy totals for a given scaling factor."""
        _, clipped_series = self._scale_and_clip_series(simulation_df, scale_factor)
        return clipped_series.resample("D").sum()

    def build_metrics_dataframe(self, daily_actual, daily_simulated, maintenance_dates):
        """Create aligned metrics dataframe for actual vs simulated energy."""
        metrics_df = pd.DataFrame({
            "Actual": daily_actual,
            "Simulated": daily_simulated
        })

        initial_rows = len(metrics_df)
        metrics_df = metrics_df.dropna()
        logging.debug(
            f"After aligning daily totals: {len(metrics_df)} rows "
            f"(removed {initial_rows - len(metrics_df)} NaN rows)"
        )

        if maintenance_dates is not None:
            allowed_dates = maintenance_dates.normalize()
            keep_mask = metrics_df.index.normalize().isin(allowed_dates)
            removed = len(metrics_df) - keep_mask.sum()
            metrics_df = metrics_df[keep_mask]
            logging.debug(
                f"After maintenance-day filter: {len(metrics_df)} rows "
                f"(removed {removed} rows)"
            )

        if metrics_df.empty:
            logging.warning("No data points remain after filtering.")

        return metrics_df

    def calculate_mbe(self, metrics_df):
        """Calculate Mean Bias Error given aligned metrics dataframe."""
        if metrics_df.empty:
            return float("inf")
        return float(np.mean(metrics_df["Simulated"] - metrics_df["Actual"]))

    def find_optimal_scaling_factor(self, daily_actual_energy, simulation_df, maintenance_dates,
                                    min_factor=0.5, max_factor=2.0):
        """Binary search to find scaling factor that minimises MBE after scaling then clipping."""
        iterations = 0
        best_factor = None
        best_mbe = float("inf")
        best_metrics_df = None

        while iterations < self.max_iterations:
            iterations += 1
            mid_factor = (min_factor + max_factor) / 2.0

            daily_simulated_energy = self.compute_daily_simulation_energy(simulation_df, mid_factor)
            metrics_df = self.build_metrics_dataframe(daily_actual_energy, daily_simulated_energy, maintenance_dates)

            if metrics_df.empty:
                logging.warning("No overlapping data points for current scaling factor; stopping optimisation.")
                break

            mbe = self.calculate_mbe(metrics_df)

            if abs(mbe) < abs(best_mbe):
                best_factor = mid_factor
                best_mbe = mbe
                best_metrics_df = metrics_df

            if abs(mbe) < self.target_mbe_tolerance:
                logging.info(f"Scaling optimisation converged after {iterations} iterations with MBE {mbe:.6e}.")
                return mid_factor, mbe, iterations, metrics_df

            if mbe > 0:
                max_factor = mid_factor
            else:
                min_factor = mid_factor

            if max_factor - min_factor < 1e-15:
                logging.info("Reached numerical precision limit during scaling optimisation.")
                break

        return best_factor, best_mbe, iterations, best_metrics_df

    def calculate_evaluation_metrics(self, metrics_df):
        """Calculate RMSE, MBE, and CRMSE from aligned metrics dataframe."""
        if metrics_df is None or metrics_df.empty:
            return None, None, None

        rmse = float(np.sqrt(mean_squared_error(metrics_df["Actual"], metrics_df["Simulated"])))
        actual_centered = metrics_df["Actual"] - metrics_df["Actual"].mean()
        simulated_centered = metrics_df["Simulated"] - metrics_df["Simulated"].mean()
        crmse = float(np.sqrt(np.mean((actual_centered - simulated_centered) ** 2)))
        mbe = float(np.mean(metrics_df["Simulated"] - metrics_df["Actual"]))

        return rmse, mbe, crmse

    def process_single_file(self, file_path, daily_actual_energy, maintenance_dates):
        """Process a single PVsyst file and return evaluation metrics."""
        try:
            simulation_df = self.load_pvsyst_csv(file_path)
            if simulation_df is None:
                return None

            optimal_factor, final_mbe, iterations, metrics_df = self.find_optimal_scaling_factor(
                daily_actual_energy, simulation_df, maintenance_dates
            )

            if optimal_factor is None or metrics_df is None or metrics_df.empty:
                logging.error(f"Could not determine optimal scaling factor for {file_path.name}")
                return None

            scaled_series, clipped_series = self._scale_and_clip_series(simulation_df, optimal_factor)
            clipped_count = int((scaled_series > clipped_series).sum())

            daily_simulated_energy = clipped_series.resample("D").sum()
            metrics_df = self.build_metrics_dataframe(daily_actual_energy, daily_simulated_energy, maintenance_dates)

            rmse, mbe, crmse = self.calculate_evaluation_metrics(metrics_df)

            uc_value, uv_value, model_name = self.extract_parameters_from_filename(file_path.name)

            logging.info(
                f"Results for {file_path.name}: Inverter={self.inverter}, Model={model_name}, "
                f"Uc={uc_value}, Uv={uv_value}, RMSE={rmse:.3f}, MBE={mbe:.6e}, "
                f"CRMSE={crmse:.3f}, Scale={optimal_factor:.6f}, ClippedHours={clipped_count}"
            )

            if self.apply_clipping:
                logging.info(
                    f"Clipping summary for {file_path.name}: threshold={self.clipping_threshold} MW, "
                    f"max_before={scaled_series.max():.3f} MW, max_after={clipped_series.max():.3f} MW"
                )

            return {
                "file directory": str(file_path),
                "inverter_id": self.inverter,
                "model_name": model_name,
                "Uc": uc_value,
                "Uv": uv_value,
                "RMSE": rmse,
                "MBE": mbe,
                "CRMSE": crmse,
                "The optimised scale factor": optimal_factor,
                "data_points": len(metrics_df),
                "iterations": iterations,
                "clipped_hours": clipped_count
            }

        except Exception as exc:
            logging.error(f"Error processing {file_path.name}: {exc}")
            return None

    def run_batch_evaluation(self):
        """Run batch evaluation on all PVsyst files in the selected folder."""
        logging.info("Starting batch evaluation with scale -> clip -> evaluation workflow.")

        measured_data = self.load_measured_power_data()
        if measured_data is None:
            logging.error("Measured electrical data could not be loaded. Aborting batch evaluation.")
            return None

        daily_actual_energy = self.compute_measured_daily_energy(measured_data)
        maintenance_dates = self.load_maintenance_free_days()

        pvsyst_files = list(self.pvsyst_dir.glob("*.CSV"))
        if not pvsyst_files:
            logging.error(f"No CSV files found in {self.pvsyst_dir}")
            return None

        logging.info(f"Found {len(pvsyst_files)} PVsyst CSV files to process.")

        results = []
        for index, file_path in enumerate(pvsyst_files, 1):
            logging.info(f"Processing file {index}/{len(pvsyst_files)}: {file_path.name}")
            result = self.process_single_file(file_path, daily_actual_energy, maintenance_dates)
            if result:
                results.append(result)

        if not results:
            logging.error("No results obtained from any files.")
            return None

        results_df = pd.DataFrame(results)
        model_type = self.detect_transpositional_model(results)
        inverter_safe = self.inverter.replace("-", "_")
        output_file = self.results_dir / f"pvsyst_batch_evaluation_scaled_clip_{inverter_safe}_{model_type}.csv"

        output_columns = [
            "file directory",
            "inverter_id",
            "model_name",
            "Uc",
            "Uv",
            "RMSE",
            "MBE",
            "CRMSE",
            "The optimised scale factor"
        ]
        results_df[output_columns].to_csv(output_file, index=False)

        logging.info(f"Results saved to: {output_file}")
        logging.info(f"Transpositional model type: {model_type}")
        logging.info("=" * 50)
        logging.info("BATCH EVALUATION SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Files processed successfully: {len(results)}")
        logging.info(f"Average RMSE: {results_df['RMSE'].mean():.3f} +/- {results_df['RMSE'].std():.3f}")
        logging.info(f"Average CRMSE: {results_df['CRMSE'].mean():.3f} +/- {results_df['CRMSE'].std():.3f}")
        logging.info(
            f"Average scaling factor: {results_df['The optimised scale factor'].mean():.6f} +/- "
            f"{results_df['The optimised scale factor'].std():.6f}"
        )
        logging.info(f"Average data points per file: {results_df['data_points'].mean():.1f}")

        if self.apply_clipping:
            logging.info(f"Average clipped hours per file: {results_df['clipped_hours'].mean():.1f}")

        return results_df, str(output_file)


def main():
    """Script entry point."""
    print("PVsyst Batch Evaluation (Scale -> Clip -> Evaluate)")
    print("=" * 60)

    try:
        evaluator = PVsystScaledClipBatchEvaluator()
        evaluation_result = evaluator.run_batch_evaluation()

        if evaluation_result is not None:
            _, actual_filename = evaluation_result
            print("\nBatch evaluation completed successfully!")
            print(f"Results saved to: {actual_filename}")
        else:
            print("\nBatch evaluation failed!")
    except Exception as exc:
        logging.error(f"Fatal error in main: {exc}")
        print(f"\nFatal error: {exc}")


if __name__ == "__main__":
    main()
