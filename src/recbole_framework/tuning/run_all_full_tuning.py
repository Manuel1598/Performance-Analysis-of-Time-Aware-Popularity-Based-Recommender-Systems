import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path) -> None:
    print(f"\n===== STARTING {script_path.name} =====")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
    )

    if result.returncode != 0:
        print(f"Script failed: {script_path}")
        print(f"Return code: {result.returncode}")
    else:
        print(f"Finished successfully: {script_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    scripts = [
        project_root / "src" / "recbole_framework" / "tuning" / "tune_session_models_full.py",
        project_root / "src" / "recbole_framework" / "tuning" / "tune_topn_models_full.py",
    ]

    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    main()