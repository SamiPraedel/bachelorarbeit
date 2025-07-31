# csv_logger.py
# ------------------------------------------------------------
# This file handles CSV logging for the SSL experiments.

import csv

# ------------------------------------------------------------
# This file handles CSV logging for the SSL experiments.

CSV_FILE_PATH = "ssl_results.csv"   # default output file


CSV_HEADERS = [
    "Dataset", "Model", "SSL_Meth", "Lab_Frac", "Acc",
]
COLUMN_WIDTHS = [10, 16, 14, 12, 10]

def format_for_csv_row(data_list, widths):
    return [str(item).ljust(widths[i]) for i, item in enumerate(data_list)]

def initialize_csv():
    with open(CSV_FILE_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(CSV_HEADERS, COLUMN_WIDTHS))

def append_to_csv(data_row):
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(format_for_csv_row(data_row, COLUMN_WIDTHS))