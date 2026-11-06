
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import yfinance as yf


# ==================================================
# 1. CONFIGURATION
# ==================================================

START_DATE = "2004-09-23"

# Yahoo Finance end date is exclusive.
# Therefore, November 5 includes November 4.
END_DATE_EXCLUSIVE = "2026-11-05"

FRED_END_DATE = (
    pd.Timestamp(END_DATE_EXCLUSIVE) - pd.Timedelta(days=1)
).strftime("%Y-%m-%d")

TIMEZONE = "America/New_York"

# Automatically locate the directory containing this script.
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "financial_dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
LOG_DIR = DATASET_DIR / "logs"

# Create all required directories automatically.
for directory in [RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==================================================
# 3. YAHOO FINANCE: ETF PRICES
# ==================================================

TICKERS = ["SPY", "IEF", "LQD", "VNQ"]

prices = {}

logger.info("Starting Yahoo Finance downloads...")

for ticker in TICKERS:

    logger.info(f"Downloading {ticker}...")

    try:
        data = yf.Ticker(ticker).history(
            start=START_DATE,
            end=END_DATE_EXCLUSIVE,
            interval="1d",
            auto_adjust=False,
            actions=True
        )

        if data.empty:
            raise ValueError(f"No historical data returned for {ticker}")

        if "Adj Close" not in data.columns:
            raise ValueError(
                f"Adjusted closing prices unavailable for {ticker}"
            )

        # Save original downloaded data.
        data.to_csv(RAW_DIR / f"{ticker}_raw.csv")

        # Normalize Yahoo timestamps to US Eastern dates.
        index = pd.DatetimeIndex(data.index)

        if index.tz is not None:
            index = index.tz_convert(TIMEZONE).tz_localize(None)

        data.index = index.normalize()
        data.index.name = "Date"

        prices[ticker] = pd.to_numeric(
            data["Adj Close"], errors="coerce"
        ).rename(ticker)

        logger.info(
            f"{ticker}: {len(data)} observations downloaded."
        )

    except Exception:
        logger.exception(f"Failed to download {ticker}.")
        raise


prices_df = pd.concat(prices.values(), axis=1).sort_index()


# ==================================================
# 4. FRED: MACROECONOMIC INDICATORS
# ==================================================

FRED_SERIES = {
    "VIX": "VIXCLS",
    "BAA10Y": "BAA10Y",
    "DGS3MO": "DGS3MO",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
    "T10Y2Y": "T10Y2Y"
}

macro_data = {}

logger.info("Starting FRED downloads...")

for name, series_id in FRED_SERIES.items():

    logger.info(f"Downloading {name} ({series_id})...")

    try:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}"
            f"&cosd={START_DATE}"
            f"&coed={FRED_END_DATE}"
        )

        data = pd.read_csv(
            url,
            na_values=[".", "NA", "N/A"]
        )

        if data.empty or len(data.columns) < 2:
            raise ValueError(f"Invalid FRED response for {series_id}")

        # Identify date and value columns.
        date_column = data.columns[0]

        value_column = (
            series_id if series_id in data.columns
            else data.columns[1]
        )

        data[date_column] = pd.to_datetime(
            data[date_column], errors="coerce"
        )

        data[value_column] = pd.to_numeric(
            data[value_column], errors="coerce"
        )

        data = data.dropna(subset=[date_column])

        # Preserve the original downloaded CSV.
        data.to_csv(RAW_DIR / f"{name}_raw.csv", index=False)

        # FRED observations are date-based, not intraday timestamps.
        series = data.set_index(date_column)[value_column]

        series.index = pd.DatetimeIndex(series.index).normalize()
        series.index.name = "Date"

        if series.dropna().empty:
            raise ValueError(f"No valid observations for {series_id}")

        if series.index.has_duplicates:
            raise ValueError(f"Duplicate dates in {series_id}")

        macro_data[name] = series.rename(name)

        logger.info(
            f"{name}: {series.notna().sum()} valid observations."
        )

    except Exception:
        logger.exception(f"Failed to download {name}.")
        raise


macro_df = pd.concat(
    macro_data.values(), axis=1
).sort_index()


# ==================================================
# 5. DATA VALIDATION
# ==================================================

logger.info("Validating downloaded data...")

if prices_df.index.has_duplicates:
    raise ValueError("Duplicate dates found in ETF data.")

if macro_df.index.has_duplicates:
    raise ValueError("Duplicate dates found in FRED data.")

# Report coverage for each individual variable.
all_individual_data = pd.concat(
    [prices_df, macro_df], axis=1
)

coverage = []

for column in all_individual_data.columns:

    series = all_individual_data[column]

    coverage.append({
        "Variable": column,
        "First observation": series.first_valid_index(),
        "Last observation": series.last_valid_index(),
        "Valid observations": series.notna().sum(),
        "Missing observations": series.isna().sum()
    })

coverage_df = pd.DataFrame(coverage)


# ==================================================
# 6. MERGE DATASETS
# ==================================================

logger.info("Merging all datasets...")

# Keep only common observation dates.
all_data = prices_df.join(
    macro_df,
    how="inner"
).sort_index()

# Record missing values before removing incomplete rows.
missing_before_cleaning = all_data.isna().sum()

# Keep only complete observations.
final_data = all_data.dropna().copy()

if final_data.empty:
    raise ValueError(
        "Final dataset is empty. Check series coverage and date alignment."
    )

if final_data.index.has_duplicates:
    raise ValueError("Duplicate dates in final dataset.")

# Restrict the requested date range.
final_data = final_data.loc[
    (final_data.index >= pd.Timestamp(START_DATE)) &
    (final_data.index < pd.Timestamp(END_DATE_EXCLUSIVE))
]


# ==================================================
# 7. EXPORT FINAL DATASET
# ==================================================

output_file = PROCESSED_DIR / "final_dataset.csv"

final_data.to_csv(output_file, index_label="Date")

logger.info(f"Final dataset saved to: {output_file}")


# ==================================================
# 8. DATA QUALITY REPORT
# ==================================================

report_file = PROCESSED_DIR / "data_quality_report.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("FINANCIAL DATASET - QUALITY REPORT\n")
    report.write("=" * 50 + "\n\n")

    report.write(f"Generation date: {datetime.now()}\n")
    report.write(f"Start date requested: {START_DATE}\n")
    report.write(f"End date requested: {FRED_END_DATE}\n")
    report.write(f"Reference timezone: {TIMEZONE}\n")
    report.write("Frequency: Daily\n")
    report.write("Geographical scope: United States\n\n")

    report.write("INDIVIDUAL VARIABLE COVERAGE\n")
    report.write("-" * 50 + "\n")

    report.write(coverage_df.to_string(index=False))

    report.write("\n\nFINAL DATASET\n")
    report.write("-" * 50 + "\n")

    report.write(f"First observation: {final_data.index.min()}\n")
    report.write(f"Last observation: {final_data.index.max()}\n")
    report.write(f"Total rows: {len(final_data)}\n")
    report.write(f"Total columns: {len(final_data.columns)}\n")

    report.write("\nMISSING VALUES BEFORE CLEANING\n")
    report.write("-" * 50 + "\n")
    report.write(missing_before_cleaning.to_string())

    report.write("\n\nMISSING VALUES AFTER CLEANING\n")
    report.write("-" * 50 + "\n")
    report.write(final_data.isna().sum().to_string())

    report.write("\n\nVARIABLES\n")
    report.write("-" * 50 + "\n")
    report.write(", ".join(final_data.columns))

    report.write("\n\nDATASET STATUS\n")
    report.write("-" * 50 + "\n")

    if final_data.index.max() < pd.Timestamp(FRED_END_DATE):
        report.write(
            "WARNING: Final observation is earlier than requested endpoint.\n"
        )
        report.write(
            "Check individual series coverage, especially BAA10Y.\n"
        )
    else:
        report.write("Requested endpoint reached.\n")

    report.write(
        "\nNote: Common dates do not guarantee simultaneous publication "
        "of all indicators.\n"
    )

logger.info(f"Quality report saved to: {report_file}")


# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 55)
print("FINANCIAL DATASET DOWNLOAD COMPLETED")
print("=" * 55)

print(f"\nFirst date: {final_data.index.min()}")
print(f"Last date:  {final_data.index.max()}")

print(f"\nRows:    {len(final_data)}")
print(f"Columns: {len(final_data.columns)}")

print("\nIndividual coverage:")
print(coverage_df.to_string(index=False))

print("\nFinal missing values:")
print(final_data.isna().sum())

print("\nGenerated files:")
print(f"Raw data:       {RAW_DIR}")
print(f"Processed data: {PROCESSED_DIR}")
print(f"Logs:           {LOG_DIR}")

print("\nDataset generation finished.")
