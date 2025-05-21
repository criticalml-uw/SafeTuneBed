import json

# Fallback score keys to check in order of priority
SCORE_KEYS = ["score", "average_score", "ASR", "harmfulness", "Harmfulness Score", "accuracy", "Average Harmfulness"]

def extract_score(metrics: dict) -> str:
    if not isinstance(metrics, dict):
        return "–"
    for key in SCORE_KEYS:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                return f"{val:.2f}"
            return str(val)
    return "–"

def json_to_latex_table(json_path: str, out_tex: str):
    with open(json_path, "r") as f:
        results = json.load(f)

    models = list(results.keys())
    benchmarks = list(next(iter(results.values())).keys())

    # Start LaTeX table
    latex = "\\begin{tabular}{l" + "c" * len(benchmarks) + "}\n"
    latex += "\\toprule\n"
    latex += "Model & " + " & ".join(benchmarks) + " \\\\\n"
    latex += "\\midrule\n"

    for model in models:
        row = [model]
        for bench in benchmarks:
            metrics = results[model].get(bench, {})
            row.append(extract_score(metrics))
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"

    with open(out_tex, "w") as f:
        f.write(latex)

    print(f"LaTeX table written to: {out_tex}")