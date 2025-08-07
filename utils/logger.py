import csv
import os

class CSVLogger:
    def __init__(self, filepath, header=None):
        self.filepath = filepath
        self.header_written = False
        if header:
            self.write_header(header)

    def write_header(self, header):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            self.header_written = True

    def log_row(self, row):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
