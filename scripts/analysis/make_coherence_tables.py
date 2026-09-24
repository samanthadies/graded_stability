"""
Converts the CCK coherence analysis into SI LaTeX tables reporting exact coherence
rates and violations of the individual coherence constraints by model and domain.

Examples:
    python -m scripts.analysis.make_coherence_tables --probe sawmil --overwrite
    python -m scripts.analysis.make_coherence_tables --probe svm --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


DATASETS = ("cities_loc", "med_indications", "defs")
DATASET_NAMES = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}
ESTIMATORS = ("direct", "joint")
ESTIMATOR_NAMES = {
    "direct": "Direct",
    "joint": "Joint",
}
PROBE_NAMES = {
    "sawmil": "sAwMIL",
    "svm": "SVM",
    "mean_difference": "Mass Mean",
}

CONSTRAINTS = (
    ("C_T_le_P_T", r"$C_T \le P_T$"),
    ("C_F_le_P_F", r"$C_F \le P_F$"),
    ("x_F_le_C_N", r"$x_F \le C_N$"),
    ("C_N_le_P_N_plus_x_F", r"$C_N \le P_N+x_F$"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SI LaTeX tables for exact CCK coherence and "
            "individual constraint violations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/analysis/coherence/model_summary.parquet"),
        help="Coherence model_summary.parquet.",
    )
    parser.add_argument(
        "--model_list",
        type=Path,
        default=Path("configs/model_list.yaml"),
        help="Canonical model ordering.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/analysis/coherence"),
    )
    parser.add_argument(
        "--probe",
        choices=tuple(PROBE_NAMES),
        default="sawmil",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; "
            f"available={frame.columns.tolist()}"
        )


def load_model_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]

    if isinstance(raw, dict):
        models = [str(key) for key in raw]
    elif isinstance(raw, list):
        models = []
        for item in raw:
            if isinstance(item, str):
                models.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(f"Unsupported model-list entry: {item!r}")

            value = next(
                (
                    item[key]
                    for key in ("name", "config", "model_name", "key")
                    if key in item
                ),
                None,
            )

            if value is None and len(item) == 1:
                value = next(iter(item))

            if value is None:
                raise ValueError(f"Cannot infer model key from {item!r}")

            models.append(str(value))
    else:
        raise ValueError(f"Unsupported model-list format in {path}")

    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError(f"No models found in {path}")

    return models


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def display_model(model: str) -> str:
    model = str(model)
    instruction = model.startswith("_")
    if instruction:
        model = model[1:]
    label = latex_escape(model)
    if instruction:
        label += " (i)"
    return label


def model_family(model: str) -> str:
    key = str(model).lstrip("_").lower()
    if key.startswith("llama"):
        return "llama"
    if key.startswith("gemma"):
        return "gemma"
    if key.startswith("mistral"):
        return "mistral"
    if key.startswith("qwen"):
        return "qwen"
    return key.split("-", 1)[0]


def format_exact_fraction(value: float) -> str:
    value = float(value)
    pct = 100.0 * value

    if value == 0.0:
        return r"0.000\%"
    if 0.0 < pct < 0.001:
        return r"$<0.001\%$"
    return f"{pct:.3f}" + r"\%"


def format_violation_cell(
    *,
    fraction: float,
    n_violating: int,
    mean_magnitude: float,
) -> str:

    fraction = float(fraction)
    n_violating = int(n_violating)
    mean_magnitude = float(mean_magnitude)

    pct = 100.0 * fraction

    if n_violating == 0:
        return r"0.0\% (--)"
    if 0.0 < pct < 0.1:
        freq = r"$<0.1\%$"
    else:
        freq = f"{pct:.1f}" + r"\%"

    return f"{freq} ({mean_magnitude:.3f})"


def selected_columns() -> list[str]:
    columns = [
        "model",
        "dataset",
        "probe",
        "estimator",
        "n_pairs",
        "n_exactly_coherent",
        "fraction_exactly_coherent",
    ]

    for key, _ in CONSTRAINTS:
        columns.extend(
            [
                f"n_violating_{key}",
                f"fraction_violating_{key}",
                f"mean_violation_{key}_given_violation",
            ]
        )

    return columns


def validate_table_population(
    frame: pd.DataFrame,
    *,
    model_order: list[str],
    probe: str,
) -> None:
    expected_models = set(model_order)
    observed_models = set(frame["model"].astype(str))

    missing_models = sorted(expected_models - observed_models)
    extra_models = sorted(observed_models - expected_models)

    if missing_models or extra_models:
        raise ValueError(
            "Model population differs from configs/model_list.yaml. "
            f"missing={missing_models}, extra={extra_models}"
        )

    expected_keys = {
        (model, dataset, estimator)
        for model in model_order
        for dataset in DATASETS
        for estimator in ESTIMATORS
    }
    observed_keys = set(
        zip(
            frame["model"].astype(str),
            frame["dataset"].astype(str),
            frame["estimator"].astype(str),
        )
    )

    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)

    if missing or extra:
        raise ValueError(
            f"Expected exactly one {probe} row for every "
            "model x dataset x estimator. "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    duplicates = frame.duplicated(["model", "dataset", "estimator"])
    if duplicates.any():
        bad = frame.loc[
            duplicates,
            ["model", "dataset", "estimator"],
        ]
        raise ValueError(
            "Duplicate table rows:\n" + bad.to_string(index=False)
        )


def make_table(
    frame: pd.DataFrame,
    *,
    dataset: str,
    probe: str,
    model_order: list[str],
) -> str:
    dataset_name = DATASET_NAMES[dataset]
    probe_name = PROBE_NAMES[probe]

    panel = frame.loc[frame["dataset"].eq(dataset)].copy()
    lookup = panel.set_index(["model", "estimator"], drop=False)

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        (
            r"\caption{\textbf{Exact CCK coherence and constraint violations "
            + f"for {dataset_name} using {probe_name}."
            + r"}}"
        ),
        f"\\label{{tab:si:exact_coherence_{dataset}_{probe}}}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (
            r" & & & "
            r"\multicolumn{4}{c}{Constraint violation: frequency (mean excess)} \\"
        ),
        r"\cmidrule(lr){4-7}",
        (
            r"Model & Estimator & Exact & "
            + " & ".join(label for _, label in CONSTRAINTS)
            + r" \\"
        ),
        r"\midrule",
    ]

    previous_family: str | None = None

    for model_index, model in enumerate(model_order):
        family = model_family(model)

        if (
            previous_family is not None
            and family != previous_family
        ):
            lines.append(r"\addlinespace[2.5pt]")

        for estimator_index, estimator in enumerate(ESTIMATORS):
            row = lookup.loc[(model, estimator)]

            model_cell = (
                display_model(model)
                if estimator_index == 0
                else ""
            )

            exact = format_exact_fraction(
                row["fraction_exactly_coherent"]
            )

            constraint_cells: list[str] = []
            for key, _ in CONSTRAINTS:
                constraint_cells.append(
                    format_violation_cell(
                        fraction=row[f"fraction_violating_{key}"],
                        n_violating=row[f"n_violating_{key}"],
                        mean_magnitude=row[
                            f"mean_violation_{key}_given_violation"
                        ],
                    )
                )

            lines.append(
                " & ".join(
                    [
                        model_cell,
                        ESTIMATOR_NAMES[estimator],
                        exact,
                        *constraint_cells,
                    ]
                )
                + r" \\"
            )

        next_model = (
            model_order[model_index + 1]
            if model_index + 1 < len(model_order)
            else None
        )
        if (
            next_model is not None
            and model_family(next_model) == family
        ):
            lines.append(r"\addlinespace[1.0pt]")

        previous_family = family

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\begin{minipage}{0.99\linewidth}\footnotesize "
                r"\textit{Note.} Exact is the percentage of matched $(P,x)$ "
                r"pairs satisfying all four CCK constraints simultaneously "
                r"using tolerance $10^{-8}$. For each individual constraint, "
                r"entries report the percentage of pairs violating the "
                r"inequality, followed in parentheses by the mean signed "
                r"constraint excess among violating pairs. A dash indicates "
                r"that no pairs violated that constraint. Direct and Joint "
                r"are evaluated on identical $(P,x)$ populations."
                r"\end{minipage}"
            ),
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def write_text(
    path: Path,
    text: str,
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    frame = pd.read_parquet(args.input)

    require_columns(
        frame,
        selected_columns(),
        source=str(args.input),
    )

    frame = frame.loc[
        frame["probe"].astype(str).eq(args.probe)
        & frame["dataset"].astype(str).isin(DATASETS)
        & frame["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    if frame.empty:
        raise RuntimeError(
            f"No rows found for probe={args.probe!r} in {args.input}."
        )

    model_order = load_model_list(args.model_list)

    validate_table_population(
        frame,
        model_order=model_order,
        probe=args.probe,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_paths: list[Path] = []
    table_texts: list[str] = []

    for dataset in DATASETS:
        text = make_table(
            frame,
            dataset=dataset,
            probe=args.probe,
            model_order=model_order,
        )

        path = (
            args.output_dir
            / f"table_si_exact_coherence_{dataset}_{args.probe}.tex"
        )

        write_text(
            path,
            text,
            overwrite=args.overwrite,
        )
        table_paths.append(path)
        table_texts.append(text)

    combined_path = (
        args.output_dir
        / f"exact_coherence_tables_{args.probe}.tex"
    )
    write_text(
        combined_path,
        "\n".join(table_texts),
        overwrite=args.overwrite,
    )

    print("=" * 100)
    print("EXACT CCK COHERENCE TABLES")
    print("=" * 100)
    print(f"Input:  {args.input}")
    print(f"Probe:  {args.probe} ({PROBE_NAMES[args.probe]})")
    print(f"Models: {len(model_order)}")
    print(f"Rows:   {len(frame)}")
    print()
    for path in table_paths:
        print(f"Wrote: {path}")
    print(f"Wrote: {combined_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
