import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import json
from pathlib import Path
import os
import random

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

# Shuffle the articles to get a random sample of prompts
random.shuffle(squad_data["data"])

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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    dcc.Store(id='evaluations-store', data=[]),
    dbc.Row(dbc.Col(html.H1("Prompt Evaluation Dashboard", className="text-center my-4"), width=12)),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prompt Selection", className="card-title"),
                    dcc.Dropdown(id="prompt-dropdown", options=prompt_options, placeholder="Select a prompt", className="prompt-dropdown-styling"),
                    html.Div(
                        [
                            dbc.Button("Evaluate", id="evaluate-button", n_clicks=0, color="primary", className="mt-3"),
                            html.Div(
                                dcc.Loading(
                                    id="loading",
                                    type="circle",
                                    color="white",
                                    children=html.Div(id="loading-output"),
                                ),
                                style={"margin-left": "15px"},
                            ),
                        ],
                        style={"display": "flex", "align-items": "center"},
                    ),
                ])
            ]),
        ], width=12),
    ]),

    dbc.Row(dbc.Col(id="alert-container", width=12, className="mt-4")),

    dbc.Row([
        dbc.Col([
            html.H2("Evaluation Results", className="text-center my-4"),
            dbc.Card(id="results-card", body=True, className="mt-4"),
        ], width=12),
    ]),
], fluid=True)


@callback(
    [Output("evaluations-store", "data"), Output("alert-container", "children"), Output("loading-output", "children")],
    Input("evaluate-button", "n_clicks"),
    [State("prompt-dropdown", "value"), State("evaluations-store", "data")],
)
def run_evaluation_and_update_store(n_clicks, prompt_id, evaluations_data):
    ctx = dash.callback_context
    if not ctx.triggered or n_clicks == 0:
        return dash.no_update, [], ""

    if not prompt_id:
        return dash.no_update, [], ""

    prompt_text = next((opt["label"] for opt in prompt_options if opt["value"] == prompt_id), None)
    if not prompt_text:
        return dash.no_update, [], ""

    reference_answers = get_reference_answers(prompt_id)
    if not reference_answers:
        alert = dbc.Alert(f"No reference answers found for prompt ID: {prompt_id}", color="warning", dismissable=True)
        return dash.no_update, alert, ""

    model_output = run_llama_model(prompt_text)
    
    if model_output.startswith("Error:"):
        alert = dbc.Alert(model_output, color="danger", dismissable=True)
        return dash.no_update, alert, ""

    evaluation = evaluate_output(model_output, reference_answers)

    evaluation_data = {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "model_output": model_output,
        "reference_answers": reference_answers,
        "evaluation": evaluation,
    }
    
    save_evaluation(evaluation_data)
    
    evaluations_data.append(evaluation_data)

    return evaluations_data, [], ""

@callback(
    Output("results-card", "children"),
    Input("evaluations-store", "data"),
)
def update_results_table(evaluations):
    if not evaluations:
        return dbc.Alert("No evaluations yet.", color="info")

    df = pd.DataFrame(evaluations)

    # Expand the 'evaluation' column
    if 'evaluation' in df.columns:
        eval_df = df['evaluation'].apply(pd.Series)
        df = pd.concat([df.drop(['evaluation'], axis=1), eval_df], axis=1)

    # Prettify reference answers
    if 'reference_answers' in df.columns:
        df['reference_answers'] = df['reference_answers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Select and reorder columns
    columns_to_display = [
        "prompt_text", "model_output", "reference_answers", "accuracy"
    ]
    df_display = df[[col for col in columns_to_display if col in df.columns]]

    return dbc.Table.from_dataframe(df_display, striped=True, bordered=True, hover=True, responsive=True)


if __name__ == "__main__":
    app.run(debug=True)