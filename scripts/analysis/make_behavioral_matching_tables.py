from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


DATASETS = (
    "cities_loc",
    "med_indications",
    "defs",
)

DATASET_NAMES = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}

ESTIMATORS = (
    "conditional",
    "joint",
)

ESTIMATOR_NAMES = {
    "conditional": "Direct",
    "joint": "Joint",
}

PROBE_NAMES = {
    "sawmil": "sAwMIL",
    "svm": "SVM",
    "mean_difference": "Mass Mean",
}

DEFAULT_SAMPLE = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SI LaTeX tables summarizing behavioral matched-pair "
            "sample sizes and matching/separation quality."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/analysis/behavioral_resilience/"
            "matched_results_all.parquet"
        ),
        help="Aggregated matched behavioral results.",
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
        default=Path("outputs/analysis/behavioral_matching"),
    )

    parser.add_argument(
        "--probe",
        choices=tuple(PROBE_NAMES),
        default="sawmil",
    )

    parser.add_argument(
        "--sample",
        type=str,
        default=DEFAULT_SAMPLE,
        help=(
            "Behavioral-analysis sample to summarize. "
            "Default: round0_agreement."
        ),
    )

    parser.add_argument(
        "--atomic_decimals",
        type=int,
        default=5,
        help="Decimal places for |Delta pi_T| mean and SD.",
    )

    parser.add_argument(
        "--gamma_decimals",
        type=int,
        default=3,
        help="Decimal places for |Delta gamma| mean and SD.",
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
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{source}: missing required columns {missing}; "
            f"available={frame.columns.tolist()}"
        )


def load_model_list(
    path: Path,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        isinstance(raw, dict)
        and "models" in raw
    ):
        raw = raw["models"]

    if isinstance(raw, dict):
        models = [
            str(key)
            for key in raw
        ]

    elif isinstance(raw, list):
        models = []

        for item in raw:
            if isinstance(
                item,
                str,
            ):
                models.append(
                    item
                )
                continue

            if not isinstance(
                item,
                dict,
            ):
                raise ValueError(
                    f"Unsupported model-list entry: {item!r}"
                )

            value = next(
                (
                    item[key]
                    for key in (
                        "name",
                        "config",
                        "model_name",
                        "key",
                    )
                    if key in item
                ),
                None,
            )

            if (
                value is None
                and len(item) == 1
            ):
                value = next(
                    iter(item)
                )

            if value is None:
                raise ValueError(
                    f"Cannot infer model key from {item!r}"
                )

            models.append(
                str(value)
            )

    else:
        raise ValueError(
            f"Unsupported model-list format in {path}"
        )

    models = list(
        dict.fromkeys(
            models
        )
    )

    if not models:
        raise ValueError(
            f"No models found in {path}"
        )

    return models


def latex_escape(
    text: str,
) -> str:
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

    return "".join(
        replacements.get(
            char,
            char,
        )
        for char in str(text)
    )


def display_model(
    model: str,
) -> str:
    model = str(
        model
    )

    instruction = model.startswith(
        "_"
    )

    if instruction:
        model = model[
            1:
        ]

    label = latex_escape(
        model
    )

    if instruction:
        label += " (i)"

    return label


def model_family(
    model: str,
) -> str:
    key = str(
        model
    ).lstrip(
        "_"
    ).lower()

    if key.startswith(
        "llama"
    ):
        return "llama"

    if key.startswith(
        "gemma"
    ):
        return "gemma"

    if key.startswith(
        "mistral"
    ):
        return "mistral"

    if key.startswith(
        "qwen"
    ):
        return "qwen"

    return key.split(
        "-",
        1,
    )[0]


def prepare_selected_rows(
    frame: pd.DataFrame,
    *,
    probe: str,
    sample: str,
) -> pd.DataFrame:
    required = [
        "model_name",
        "dataset",
        "probe",
        "sample",
        "estimator",
        "match_id",
        "atomic_gap",
        "gamma_gap",
    ]

    require_columns(
        frame,
        required,
        source="matched behavioral results",
    )

    out = frame.loc[
        frame[
            "probe"
        ].astype(str).eq(
            probe
        )
        & frame[
            "sample"
        ].astype(str).eq(
            sample
        )
        & frame[
            "dataset"
        ].astype(str).isin(
            DATASETS
        )
        & frame[
            "estimator"
        ].astype(str).isin(
            ESTIMATORS
        )
    ].copy()

    if out.empty:
        raise RuntimeError(
            "No rows remain after filtering to "
            f"probe={probe!r}, sample={sample!r}, "
            f"estimators={ESTIMATORS}."
        )

    out[
        "atomic_gap"
    ] = pd.to_numeric(
        out[
            "atomic_gap"
        ],
        errors="coerce",
    )

    out[
        "gamma_gap"
    ] = pd.to_numeric(
        out[
            "gamma_gap"
        ],
        errors="coerce",
    )

    bad_atomic = (
        ~np.isfinite(
            out[
                "atomic_gap"
            ].to_numpy(
                dtype=float
            )
        )
    )

    bad_gamma = (
        ~np.isfinite(
            out[
                "gamma_gap"
            ].to_numpy(
                dtype=float
            )
        )
    )

    if bad_atomic.any():
        raise ValueError(
            f"Found {int(bad_atomic.sum())} non-finite atomic_gap values "
            "in the selected matched-pair sample."
        )

    if bad_gamma.any():
        raise ValueError(
            f"Found {int(bad_gamma.sum())} non-finite gamma_gap values "
            "in the selected matched-pair sample."
        )

    if (
        out[
            "atomic_gap"
        ] < 0
    ).any():
        raise ValueError(
            "atomic_gap must be non-negative."
        )

    if (
        out[
            "gamma_gap"
        ] < 0
    ).any():
        raise ValueError(
            "gamma_gap must be non-negative."
        )

    duplicate_keys = [
        "model_name",
        "dataset",
        "probe",
        "sample",
        "estimator",
        "match_id",
    ]

    duplicates = out.duplicated(
        duplicate_keys,
        keep=False,
    )

    if duplicates.any():
        bad = out.loc[
            duplicates,
            duplicate_keys,
        ].sort_values(
            duplicate_keys
        )

        raise ValueError(
            "Duplicate matched-pair rows found in the selected sample:\n"
            + bad.head(
                20
            ).to_string(
                index=False
            )
        )

    return out


def summarize_matching(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        frame
        .groupby(
            [
                "model_name",
                "dataset",
                "probe",
                "sample",
                "estimator",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            n_pairs=(
                "match_id",
                "nunique",
            ),
            mean_atomic_gap=(
                "atomic_gap",
                "mean",
            ),
            std_atomic_gap=(
                "atomic_gap",
                "std",
            ),
            mean_gamma_gap=(
                "gamma_gap",
                "mean",
            ),
            std_gamma_gap=(
                "gamma_gap",
                "std",
            ),
        )
    )

    summary[
        "n_pairs"
    ] = summary[
        "n_pairs"
    ].astype(
        int
    )

    return summary


def complete_and_validate_table_population(
    summary: pd.DataFrame,
    *,
    model_order: list[str],
    probe: str,
    sample: str,
) -> pd.DataFrame:

    key_cols = [
        "model_name",
        "dataset",
        "estimator",
    ]

    duplicates = summary.duplicated(
        key_cols,
        keep=False,
    )
    if duplicates.any():
        bad = summary.loc[
            duplicates,
            key_cols,
        ].sort_values(
            key_cols
        )
        raise ValueError(
            "Duplicate summary rows:\n"
            + bad.to_string(index=False)
        )

    expected_keys = {
        (
            model,
            dataset,
            estimator,
        )
        for model in model_order
        for dataset in DATASETS
        for estimator in ESTIMATORS
    }

    observed_keys = set(
        zip(
            summary["model_name"].astype(str),
            summary["dataset"].astype(str),
            summary["estimator"].astype(str),
        )
    )

    extra = sorted(
        observed_keys
        - expected_keys
    )
    if extra:
        raise ValueError(
            "Found unexpected model/dataset/estimator combinations: "
            f"{extra[:10]}"
        )

    missing = sorted(
        expected_keys
        - observed_keys
    )

    if missing:
        print(
            "NOTE: selected matched-pair sample has zero matched pairs for "
            f"{len(missing)} model x dataset x estimator cells."
        )
        for model, dataset, estimator in missing:
            print(
                "  zero pairs: "
                f"{model} / {dataset} / {ESTIMATOR_NAMES[estimator]}"
            )

    grid = pd.MultiIndex.from_product(
        [
            model_order,
            DATASETS,
            ESTIMATORS,
        ],
        names=key_cols,
    ).to_frame(
        index=False
    )

    completed = grid.merge(
        summary,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    completed["probe"] = completed[
        "probe"
    ].fillna(
        probe
    )

    completed["sample"] = completed[
        "sample"
    ].fillna(
        sample
    )

    completed["n_pairs"] = (
        completed["n_pairs"]
        .fillna(0)
        .astype(int)
    )

    # Means/SDs remain NaN for zero-pair cells. That is intentional:
    # there is no empirical gap distribution to summarize.
    return completed


def format_mean_sd(
    mean: float,
    sd: float,
    *,
    decimals: int,
) -> str:
    if not np.isfinite(
        float(mean)
    ):
        return "--"

    if not np.isfinite(
        float(sd)
    ):
        return (
            f"{float(mean):.{decimals}f} "
            "(--)"
        )

    return (
        f"{float(mean):.{decimals}f} "
        f"({float(sd):.{decimals}f})"
    )


def make_table(
    summary: pd.DataFrame,
    *,
    dataset: str,
    probe: str,
    sample: str,
    model_order: list[str],
    atomic_decimals: int,
    gamma_decimals: int,
) -> str:
    dataset_name = DATASET_NAMES[
        dataset
    ]

    probe_name = PROBE_NAMES[
        probe
    ]

    panel = summary.loc[
        summary[
            "dataset"
        ].astype(str).eq(
            dataset
        )
    ].copy()

    lookup = panel.set_index(
        [
            "model_name",
            "estimator",
        ],
        drop=False,
    )

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        (
            r"\caption{\textbf{Behavioral matching quality for "
            + f"{dataset_name}."
            + r"} "
            + f"For each LLM, $n$ is the number of behaviorally evaluated {probe_name}-matched "
            + r"proposition pairs. "
            + r"$|\Delta \pi_T|$ is the absolute difference in atomic belief "
            + r"probability between the two propositions in each pair, and "
            + r"$|\Delta \gamma|$ is their absolute difference in graded "
            + r"stability. Difference entries report mean (SD).}"
        ),
        (
            f"\\label{{tab:si:behavioral_matching_{dataset}_{probe}}}"
        ),
        r"\begin{tabular}{lrrr rrr}",
        r"\toprule",
        (
            r" & \multicolumn{3}{c}{Direct} "
            r"& \multicolumn{3}{c}{Joint} \\"
        ),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        (
            r"Model "
            r"& $n$ "
            r"& $|\Delta \pi_T|$ "
            r"& $|\Delta \gamma|$ "
            r"& $n$ "
            r"& $|\Delta \pi_T|$ "
            r"& $|\Delta \gamma|$ \\"
        ),
        r"\midrule",
    ]

    previous_family: str | None = None

    for model_index, model in enumerate(
        model_order
    ):
        family = model_family(
            model
        )

        if (
            previous_family is not None
            and family != previous_family
        ):
            lines.append(
                r"\addlinespace[2.5pt]"
            )

        direct = lookup.loc[
            (
                model,
                "conditional",
            )
        ]

        joint = lookup.loc[
            (
                model,
                "joint",
            )
        ]

        direct_atomic = format_mean_sd(
            direct[
                "mean_atomic_gap"
            ],
            direct[
                "std_atomic_gap"
            ],
            decimals=atomic_decimals,
        )

        direct_gamma = format_mean_sd(
            direct[
                "mean_gamma_gap"
            ],
            direct[
                "std_gamma_gap"
            ],
            decimals=gamma_decimals,
        )

        joint_atomic = format_mean_sd(
            joint[
                "mean_atomic_gap"
            ],
            joint[
                "std_atomic_gap"
            ],
            decimals=atomic_decimals,
        )

        joint_gamma = format_mean_sd(
            joint[
                "mean_gamma_gap"
            ],
            joint[
                "std_gamma_gap"
            ],
            decimals=gamma_decimals,
        )

        lines.append(
            " & ".join(
                [
                    display_model(
                        model
                    ),
                    str(
                        int(
                            direct[
                                "n_pairs"
                            ]
                        )
                    ),
                    direct_atomic,
                    direct_gamma,
                    str(
                        int(
                            joint[
                                "n_pairs"
                            ]
                        )
                    ),
                    joint_atomic,
                    joint_gamma,
                ]
            )
            + r" \\"
        )

        next_model = (
            model_order[
                model_index + 1
            ]
            if (
                model_index + 1
                < len(
                    model_order
                )
            )
            else None
        )

        if (
            next_model is not None
            and model_family(
                next_model
            ) == family
        ):
            lines.append(
                r"\addlinespace[1.0pt]"
            )

        previous_family = family

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            (
                r"\begin{minipage}{0.99\linewidth}\footnotesize "
                r"\textit{Note.} Tables summarize the "
                + latex_escape(
                    sample.replace(
                        "_",
                        " ",
                    )
                )
                + r" sample. Matching is performed on atomic belief "
                r"probability within challenge-sequence blocks; graded "
                r"stability is used afterward to orient each pair. "
                r"Direct denotes the Direct Conditional stability "
                r"operationalization and Joint denotes Joint-to-Conditional "
                r"stability. Standard deviations are sample SDs across "
                r"matched proposition pairs. The default table uses the full "
                r"behaviorally evaluated matched-pair sample; the downstream "
                r"primary behavioral analysis additionally restricts to pairs "
                r"whose two round-0 responses agree with the atomic probe. "
                r"When no matched pairs enter the selected sample, $n=0$ and "
                r"the gap summaries are shown as dashes."
                r"\end{minipage}"
            ),
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(
        lines
    )


def write_text(
    path: Path,
    text: str,
    *,
    overwrite: bool,
) -> None:
    if (
        path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"{path} exists; pass --overwrite."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def write_summary_outputs(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    probe: str,
    sample: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    stem = (
        f"matching_summary_"
        f"{probe}_"
        f"{sample}"
    )

    parquet_path = (
        output_dir
        / f"{stem}.parquet"
    )

    csv_path = (
        output_dir
        / f"{stem}.csv"
    )

    for path in (
        parquet_path,
        csv_path,
    ):
        if (
            path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"{path} exists; pass --overwrite."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_parquet(
        parquet_path,
        index=False,
    )

    summary.to_csv(
        csv_path,
        index=False,
    )

    return (
        parquet_path,
        csv_path,
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            args.input
        )

    if args.atomic_decimals < 0:
        raise ValueError(
            "--atomic_decimals must be non-negative."
        )

    if args.gamma_decimals < 0:
        raise ValueError(
            "--gamma_decimals must be non-negative."
        )

    frame = pd.read_parquet(
        args.input
    )

    selected = prepare_selected_rows(
        frame,
        probe=args.probe,
        sample=args.sample,
    )

    summary = summarize_matching(
        selected
    )

    model_order = load_model_list(
        args.model_list
    )

    summary = complete_and_validate_table_population(
        summary,
        model_order=model_order,
        probe=args.probe,
        sample=args.sample,
    )

    model_rank = {
        model: index
        for index, model in enumerate(
            model_order
        )
    }

    dataset_rank = {
        dataset: index
        for index, dataset in enumerate(
            DATASETS
        )
    }

    estimator_rank = {
        estimator: index
        for index, estimator in enumerate(
            ESTIMATORS
        )
    }

    summary[
        "_model_rank"
    ] = summary[
        "model_name"
    ].map(
        model_rank
    )

    summary[
        "_dataset_rank"
    ] = summary[
        "dataset"
    ].map(
        dataset_rank
    )

    summary[
        "_estimator_rank"
    ] = summary[
        "estimator"
    ].map(
        estimator_rank
    )

    summary = (
        summary
        .sort_values(
            [
                "_dataset_rank",
                "_model_rank",
                "_estimator_rank",
            ]
        )
        .drop(
            columns=[
                "_model_rank",
                "_dataset_rank",
                "_estimator_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path, csv_path = (
        write_summary_outputs(
            summary,
            output_dir=args.output_dir,
            probe=args.probe,
            sample=args.sample,
            overwrite=args.overwrite,
        )
    )

    table_paths: list[
        Path
    ] = []

    table_texts: list[
        str
    ] = []

    for dataset in DATASETS:
        text = make_table(
            summary,
            dataset=dataset,
            probe=args.probe,
            sample=args.sample,
            model_order=model_order,
            atomic_decimals=args.atomic_decimals,
            gamma_decimals=args.gamma_decimals,
        )

        path = (
            args.output_dir
            / (
                "table_si_behavioral_matching_"
                f"{dataset}_"
                f"{args.probe}.tex"
            )
        )

        write_text(
            path,
            text,
            overwrite=args.overwrite,
        )

        table_paths.append(
            path
        )

        table_texts.append(
            text
        )

    combined_path = (
        args.output_dir
        / (
            "behavioral_matching_tables_"
            f"{args.probe}.tex"
        )
    )

    write_text(
        combined_path,
        "\n".join(
            table_texts
        ),
        overwrite=args.overwrite,
    )

    print(
        "=" * 100
    )

    print(
        "BEHAVIORAL MATCHING TABLES"
    )

    print(
        "=" * 100
    )

    print(
        f"Input:   {args.input}"
    )

    print(
        f"Probe:   {args.probe} "
        f"({PROBE_NAMES[args.probe]})"
    )

    print(
        f"Sample:  {args.sample}"
    )

    print(
        f"Models:  {len(model_order)}"
    )

    print(
        f"Rows:    {len(summary)} "
        "(expected "
        f"{len(model_order) * len(DATASETS) * len(ESTIMATORS)})"
    )

    print()

    print(
        f"Wrote: {parquet_path}"
    )

    print(
        f"Wrote: {csv_path}"
    )

    for path in table_paths:
        print(
            f"Wrote: {path}"
        )

    print(
        f"Wrote: {combined_path}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
