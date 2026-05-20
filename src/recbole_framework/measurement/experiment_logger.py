from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class ExperimentLogger:
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def log_result(self, result: dict) -> None:
        result["logged_at"] = datetime.now().isoformat(timespec="seconds")

        df = pd.DataFrame([result])

        if self.output_file.exists():
            existing_df = pd.read_csv(self.output_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(self.output_file, index=False)

    @staticmethod
    def serialize_config(config_updates: dict) -> str:
        return json.dumps(config_updates, sort_keys=True, default=str)