import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import json
from pathlib import Path
import os

from src.storage import get_evaluations, save_evaluation
from src.run_model import run_llama_model
from src.evaluate import evaluate_output, get_reference_answers

# --- Clear history on startup ---
EVALUATIONS_FILE = Path(__file__).parent.parent / "data" / "evaluations.json"
if EVALUATIONS_FILE.exists():
    os.remove(EVALUATIONS_FILE)

# --- Performance Improvement: Limit the number of prompts loaded ---
MAX_PROMPTS = 500  # Limit to the first 500 prompts for faster loading

SQUAD_FILE = Path(__file__).parent.parent / "data" / "squad-train-v2.0.json"
with open(SQUAD_FILE, "r", encoding="utf-8") as f:
    squad_data = json.load(f)

prompt_options = []
prompt_count = 0
for article in squad_data["data"]:
    if prompt_count >= MAX_PROMPTS:
        break
    for paragraph in article["paragraphs"]:
        if prompt_count >= MAX_PROMPTS:
            break
        for qa in paragraph["qas"]:
            if not qa["is_impossible"]:
                prompt_options.append({"label": qa["question"], "value": qa["id"]})
                prompt_count += 1
                if prompt_count >= MAX_PROMPTS:
                    break

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Prompt Evaluation Dashboard"), width=12)),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="prompt-dropdown", options=prompt_options, placeholder="Select a prompt"),
            dbc.Button("Evaluate", id="evaluate-button", n_clicks=0, className="mt-2"),
        ], width=6),
    ]),
    dbc.Row(dbc.Col(id="alert-container", width=12, className="mt-4")),
    dbc.Row(dbc.Col(html.H2("Evaluation Results"), width=12, className="mt-4")),
    dbc.Row(dbc.Col(id="results-table", width=12)),
])

@callback(
    [Output("results-table", "children"), Output("alert-container", "children")],
    Input("evaluate-button", "n_clicks"),
    State("prompt-dropdown", "value"),
)
def run_evaluation_and_update_table(n_clicks, prompt_id):
    ctx = dash.callback_context
    if not ctx.triggered or n_clicks == 0:
        return create_results_table(), []

    if not prompt_id:
        return dash.no_update, dash.no_update

    prompt_text = next((opt["label"] for opt in prompt_options if opt["value"] == prompt_id), None)
    if not prompt_text:
        return dash.no_update, dash.no_update

    reference_answers = get_reference_answers(prompt_id)
    if not reference_answers:
        alert = dbc.Alert(f"No reference answers found for prompt ID: {prompt_id}", color="warning", dismissable=True)
        return dash.no_update, alert

    model_output = run_llama_model(prompt_text)
    
    if model_output.startswith("Error:"):
        alert = dbc.Alert(model_output, color="danger", dismissable=True)
        return create_results_table(), alert

    evaluation = evaluate_output(model_output, reference_answers)

    evaluation_data = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "model_output": model_output,
        "reference_answers": reference_answers,
        "evaluation": evaluation,
    }
    save_evaluation(evaluation_data)

    return create_results_table(), []

def create_results_table():
    evaluations = get_evaluations()
    if not evaluations:
        return dbc.Alert("No evaluations yet.", color="info")

    df = pd.DataFrame(evaluations)
    df['reference_answers'] = df['reference_answers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['evaluation'] = df['evaluation'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, responsive=True)

if __name__ == "__main__":
    app.run(debug=True)
