"""Train a frozen-corpus policy and publish progress to the local replay dashboard."""
from pathlib import Path
import sys

# Direct script execution must use this checkout even without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quoridor_lab.supervised import main


if __name__ == "__main__":
    main()
