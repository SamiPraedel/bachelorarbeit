# csv_logger.py
# ------------------------------------------------------------
# This file handles CSV logging for the SSL experiments.

import csv

# Import configurations from experiment_config.py
try:
    import experiment_config as config
except ImportError:
    print("Error: experiment_config.py not found. Please ensure it's in the same directory.")
    exit()

CSV_HEADERS = [
    "Dataset", "Model", "SSL_Meth", "Lab_Frac", 
    "T_Acc", "S_Acc", "Sil_Firing_True"
]
COLUMN_WIDTHS = [10, 12, 12, 12, 10, 10, 15, 15, 18]

def format_for_csv_row(data_list, widths):
    return [str(item).ljust(widths[i]) for i, item in enumerate(data_list)]

def initialize_csv():
    with open(config.CSV_FILE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(CSV_HEADERS, COLUMN_WIDTHS))

def append_to_csv(data_row):
    with open(config.CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(data_row, COLUMN_WIDTHS))