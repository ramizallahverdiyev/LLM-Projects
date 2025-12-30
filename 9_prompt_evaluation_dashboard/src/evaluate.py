import json
from pathlib import Path

def evaluate_output(output, reference_answers):
    """
    Evaluates if any of the reference answers are present in the model's output.
    This is a simple substring check.
    """
    cleaned_output = output.strip().lower()
    
    for ans in reference_answers:
        if ans.strip().lower() in cleaned_output:
            return {"accuracy": 1.0}

    return {"accuracy": 0.0}

def get_reference_answers(prompt_id):
    """
    Retrieves the reference answers for a given prompt ID from the SQuAD dataset.
    """
    SQUAD_FILE = Path(__file__).parent.parent / "data" / "squad-train-v2.0.json"
    with open(SQUAD_FILE, "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] == prompt_id:
                    # Return only non-empty answers
                    return [ans["text"] for ans in qa.get("answers", []) if ans.get("text")]
    return []

if __name__ == '__main__':
    # Example usage:
    prompt_id_to_test = "56be892d3aeaaa14008c908c" # Where did Beyonce get her name from?
    reference_answers = get_reference_answers(prompt_id_to_test)
    
    model_output_verbose = "Beyoncé's name is a tribute to her mother's maiden name."
    model_output_exact = "her mother's maiden name"
    model_output_incorrect = "from a song"

    eval_verbose = evaluate_output(model_output_verbose, reference_answers)
    eval_exact = evaluate_output(model_output_exact, reference_answers)
    eval_incorrect = evaluate_output(model_output_incorrect, reference_answers)

    print(f"Reference Answers: {reference_answers}")
    print(f"Model Output (Verbose): {model_output_verbose}")
    print(f"Evaluation (Verbose): {eval_verbose}")
    print("-" * 20)
    print(f"Model Output (Exact): {model_output_exact}")
    print(f"Evaluation (Exact): {eval_exact}")
    print("-" * 20)
    print(f"Model Output (Incorrect): {model_output_incorrect}")
    print(f"Evaluation (Incorrect): {eval_incorrect}")
