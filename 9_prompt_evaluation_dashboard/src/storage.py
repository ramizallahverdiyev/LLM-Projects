import json
from pathlib import Path

EVALUATIONS_FILE = Path(__file__).parent.parent / "data" / "evaluations.json"

def save_evaluation(evaluation_data):
    """Saves a new evaluation to the evaluations file."""
    evaluations = get_evaluations()
    evaluations.append(evaluation_data)
    with open(EVALUATIONS_FILE, "w") as f:
        json.dump(evaluations, f, indent=4)

def get_evaluations():
    """Reads and returns all evaluations from the evaluations file."""
    if not EVALUATIONS_FILE.exists():
        return []
    with open(EVALUATIONS_FILE, "r") as f:
        return json.load(f)
