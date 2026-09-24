"""
Recomputes manuscript-facing numerical summaries directly from the current analysis
outputs and records their source files, providing a consistency audit.

Examples:
    python -m scripts.analysis.audit_manuscript_numbers
    python -m scripts.analysis.audit_manuscript_numbers --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PRIMARY_PROBE = "sawmil"
PRIMARY_ESTIMATOR = "direct"

DATASET_ORDER = ["cities_loc", "med_indications", "defs"]
DATASET_LABELS = {
    "cities_loc": "City Locations",
    "med_indications": "Medical Indications",
    "defs": "Word Definitions",
}

PARAM_B = {
    "gemma-7b": 8.54,
    "gemma-2-9b": 9.24,
    "gemma-2-27b": 27.23,
    "llama-3.2-3b": 3.21,
    "llama-3.1-8b": 8.03,
    "llama-3.1-70b": 70.55,
    "mistral-7b": 7.25,
    "mistral-12b": 12.25,
    "mistral-3.1-24b": 23.57,
    "qwen-2.5-7b": 7.62,
    "qwen-2.5-14b": 14.80,
    "qwen-2.5-72b": 72.70,
}

FAMILY_ORDER = ["llama", "gemma", "mistral", "qwen"]

SOURCE_MAP = {
    "Figure 2": {
        "analysis_script": "scripts/analysis/analyze_stability_vs_credence.py",
        "plot_script": "scripts/plotting/plot_fig2.py",
        "figure": "outputs/figures/figure_2_bars.pdf",
        "results": [
            "outputs/analysis/stability_vs_credence/model_summary.parquet",
            "outputs/analysis/stability_vs_credence/cross_model_agreement.parquet",
            "outputs/analysis/stability_vs_credence/permutation_summary.parquet",
        ],
    },
    "Figure 3": {
        "analysis_script": "scripts/analysis/analyze_coherence.py",
        "plot_script": "scripts/plotting/plot_fig3.py",
        "figure": "outputs/figures/figure_3.pdf",
        "results": [
            "outputs/analysis/coherence/model_summary.parquet",
        ],
    },
    "Figure 4": {
        "analysis_script": "scripts/analysis/analyze_stability_variation.py",
        "plot_script": "scripts/plotting/plot_fig4.py",
        "figure_current_manuscript": "outputs/figures/figure_4_relative_scale.pdf",
        "figure_parameter_version": "outputs/figures/figure_4_parameter_scale.pdf",
        "results": [
            "outputs/analysis/stability_variation/model_summary.parquet",
            "outputs/analysis/stability_variation/scale_pair_deltas.parquet",
            "outputs/analysis/stability_variation/scale_trends.parquet",
        ],
    },
    "Figure 5": {
        "analysis_script": "scripts/analysis/analyze_behavioral_resilience.py",
        "plot_script": "scripts/plotting/plot_fig5.py",
        "figure": "outputs/figures/figure_5.pdf",
        "results": [
            "outputs/analysis/behavioral_resilience/matched_unit_summary.parquet",
        ],
    },
    "SI Direct–Joint agreement": {
        "analysis_script": "scripts/analysis/analyze_operationalization_agreement.py",
        "plot_script": "scripts/plotting/plot_operationalization_agreement_si.py",
        "figure": "outputs/figures/figure_si_operationalization_agreement.pdf",
        "results": [
            "outputs/analysis/operationalization_agreement/model_summary.parquet",
        ],
    },
    "SI base vs instruction": {
        "analysis_script": "scripts/analysis/analyze_stability_variation.py",
        "plot_script": "scripts/plotting/plot_base_v_instruct_si.py",
        "figure": "outputs/figures/figure_si_base_v_instruct.pdf",
        "results": [
            "outputs/analysis/stability_variation/instruction_pair_summary.parquet",
            "outputs/analysis/stability_variation/instruction_statement_deltas.parquet",
        ],
    },
}


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, s: str = "") -> None:
        self.lines.append(str(s))
        print(s)

    def header(self, title: str) -> None:
        self.say()
        self.say("=" * 88)
        self.say(title)
        self.say("=" * 88)

    def subheader(self, title: str) -> None:
        self.say()
        self.say("-" * 88)
        self.say(title)
        self.say("-" * 88)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if pd.isna(x) if not isinstance(x, (dict, list, tuple)) else False:
        return None
    return x


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def norm_text(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def pick_col(
    df: pd.DataFrame,
    aliases: Iterable[str],
    *,
    label: str,
    required: bool = True,
    numeric: bool | None = None,
) -> str | None:

    aliases = list(aliases)
    normalized = {norm_col(c): c for c in df.columns}

    # Exact normalized aliases.
    for a in aliases:
        na = norm_col(a)
        if na in normalized:
            c = normalized[na]
            if numeric is True and not pd.api.types.is_numeric_dtype(df[c]):
                continue
            return c

    # Conservative token matching.
    matches: list[str] = []
    alias_tokens = [
        [tok for tok in norm_col(a).split("_") if tok not in {"of", "the"}]
        for a in aliases
    ]
    for c in df.columns:
        nc = norm_col(c)
        if numeric is True and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        for toks in alias_tokens:
            if toks and all(tok in nc for tok in toks):
                matches.append(c)
                break

    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]

    if not required:
        return None

    raise KeyError(
        f"Could not uniquely identify {label}.\n"
        f"Aliases tried: {aliases}\n"
        f"Candidate matches: {matches}\n"
        f"Available columns: {list(df.columns)}"
    )


def canonical_dataset(x: Any) -> str:
    s = norm_text(x)
    mapping = {
        "cities_loc": "cities_loc",
        "city_locations": "cities_loc",
        "city_location": "cities_loc",
        "cities": "cities_loc",
        "medical_indications": "med_indications",
        "med_indications": "med_indications",
        "medical": "med_indications",
        "word_definitions": "defs",
        "word_definition": "defs",
        "definitions": "defs",
        "defs": "defs",
    }
    if s in mapping:
        return mapping[s]
    if "cities_loc" in s or "city_location" in s:
        return "cities_loc"
    if "med_indications" in s or "medical_indications" in s:
        return "med_indications"
    if "word_definitions" in s or s.startswith("defs") or "_defs" in s:
        return "defs"
    return s


def is_instruction_model_name(x: Any) -> bool:

    return str(x).strip().startswith("_")


def canonical_model(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.lstrip("_")
    s = re.sub(r"\s*\(b\)\s*$", "", s)
    s = s.replace("base_", "")
    return s


def model_family(model: str) -> str:
    m = canonical_model(model)
    for fam in FAMILY_ORDER:
        if m.startswith(fam):
            return fam
    raise ValueError(f"Cannot infer model family from {model!r}")


@dataclass
class Loaded:
    path: Path
    df: pd.DataFrame
    sha256: str


class Audit:
    def __init__(self, root: Path, strict: bool, reporter: Reporter):
        self.root = root
        self.strict = strict
        self.r = reporter
        self.loaded: dict[str, Loaded] = {}
        self.selected_columns: dict[str, dict[str, str]] = {}
        self.results: dict[str, Any] = {}
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.r.say(f"WARNING: {msg}")
        if self.strict:
            raise RuntimeError(msg)

    def load(self, rel: str) -> pd.DataFrame:
        if rel in self.loaded:
            return self.loaded[rel].df.copy()
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Required audit input not found:\n  {path}\n"
                "Run the corresponding analysis first or verify --root."
            )
        df = pd.read_parquet(path)
        self.loaded[rel] = Loaded(path=path, df=df.copy(), sha256=sha256(path))
        return df

    def select(self, file_key: str, logical: str, col: str) -> None:
        self.selected_columns.setdefault(file_key, {})[logical] = col

    def basic_dims(
        self,
        df: pd.DataFrame,
        file_key: str,
        *,
        model_required: bool = True,
        dataset_required: bool = True,
    ) -> tuple[str | None, str | None]:
        model = pick_col(
            df, ["model", "llm", "model_name", "short_name"],
            label=f"{file_key}: model", required=model_required
        )
        dataset = pick_col(
            df, ["dataset", "domain", "dataset_name"],
            label=f"{file_key}: dataset", required=dataset_required
        )
        if model:
            self.select(file_key, "model", model)
        if dataset:
            self.select(file_key, "dataset", dataset)
        return model, dataset

    def filter_primary(
        self,
        df: pd.DataFrame,
        file_key: str,
        *,
        instruction_only: bool = True,
        filter_probe: bool = True,
        filter_estimator: bool = True,
        model_required: bool = True,
        dataset_required: bool = True,
    ) -> pd.DataFrame:
        out = df.copy()
        model_col, dataset_col = self.basic_dims(
            out, file_key,
            model_required=model_required,
            dataset_required=dataset_required,
        )

        if dataset_col is not None:
            out = out.assign(_dataset=out[dataset_col].map(canonical_dataset))

        if filter_probe:
            probe_col = pick_col(
                out, ["probe", "probe_name", "probe_type"],
                label=f"{file_key}: probe", required=False
            )
            if probe_col:
                self.select(file_key, "probe", probe_col)
                vals = out[probe_col].astype(str).str.lower()
                mask = vals.str.contains("sawmil", regex=False)
                out = out.loc[mask].copy()

        if filter_estimator:
            est_col = pick_col(
                out,
                ["estimator", "conditional_estimator", "operationalization", "source_type"],
                label=f"{file_key}: estimator",
                required=False,
            )
            if est_col:
                self.select(file_key, "estimator", est_col)
                vals = out[est_col].astype(str).str.lower()
                mask = vals.str.contains("direct", regex=False) & ~vals.str.contains("joint", regex=False)
                out = out.loc[mask].copy()

        if instruction_only:

            is_base_col = pick_col(
                out,
                ["is_base", "base_model", "is_pretrained_base"],
                label=f"{file_key}: base flag",
                required=False,
            )
            tuning_col = pick_col(
                out,
                ["tuning", "model_type", "variant", "model_variant"],
                label=f"{file_key}: tuning field",
                required=False,
            )
            is_instruct_col = pick_col(
                out,
                ["is_instruction", "is_instruction_tuned", "instruction_tuned"],
                label=f"{file_key}: instruction flag",
                required=False,
            )

            if is_instruct_col is not None:
                self.select(file_key, "instruction_flag", is_instruct_col)
                flag = out[is_instruct_col]
                if pd.api.types.is_bool_dtype(flag):
                    out = out.loc[flag].copy()
                else:
                    out = out.loc[
                        flag.astype(str).str.lower().isin(
                            {"true", "1", "yes", "instruction", "instruct", "instruction_tuned"}
                        )
                    ].copy()
            elif is_base_col is not None:
                self.select(file_key, "base_flag", is_base_col)
                flag = out[is_base_col]
                if pd.api.types.is_bool_dtype(flag):
                    out = out.loc[~flag].copy()
                else:
                    out = out.loc[
                        ~flag.astype(str).str.lower().isin({"true", "1", "yes", "base"})
                    ].copy()
            elif tuning_col is not None:
                self.select(file_key, "tuning", tuning_col)
                flag = out[tuning_col].astype(str).str.lower()
                if flag.str.contains("instruct").any():
                    out = out.loc[flag.str.contains("instruct")].copy()
                elif flag.str.contains("base").any():
                    out = out.loc[~flag.str.contains("base")].copy()

            if model_col is not None:
                out = out.loc[out[model_col].map(is_instruction_model_name)].copy()
                self.select(file_key, "instruction_model_name_guard", model_col)
            else:
                model_a_col = pick_col(
                    out,
                    ["model_a", "llm_a", "model_1", "model1"],
                    label=f"{file_key}: first model in pair",
                    required=False,
                )
                model_b_col = pick_col(
                    out,
                    ["model_b", "llm_b", "model_2", "model2"],
                    label=f"{file_key}: second model in pair",
                    required=False,
                )
                if model_a_col is not None and model_b_col is not None:
                    self.select(file_key, "model_a", model_a_col)
                    self.select(file_key, "model_b", model_b_col)
                    out = out.loc[
                        out[model_a_col].map(is_instruction_model_name)
                        & out[model_b_col].map(is_instruction_model_name)
                    ].copy()
                    self.select(file_key, "instruction_pair_name_guard", f"{model_a_col}, {model_b_col}")
                elif self.strict:
                    raise RuntimeError(
                        f"{file_key}: instruction_only=True, but no single-model column "
                        "or pair of model columns was available to enforce the leading-underscore "
                        "instruction-model convention."
                    )

        if model_col is not None:
            out = out.assign(_model=out[model_col].map(canonical_model))

        return out

    def check_model_domain_grid(
        self,
        df: pd.DataFrame,
        label: str,
        expected_models: int = 12,
        expected_domains: int = 3,
        require_one_row_per_pair: bool = True,
    ) -> None:
        if "_model" not in df or "_dataset" not in df:
            return
        n_models = df["_model"].nunique()
        n_domains = df["_dataset"].nunique()
        pairs = df[["_model", "_dataset"]].drop_duplicates()
        expected_pairs = expected_models * expected_domains

        problems = []
        if n_models != expected_models:
            problems.append(f"{n_models} unique models (expected {expected_models})")
        if n_domains != expected_domains:
            problems.append(f"{n_domains} domains (expected {expected_domains})")
        if len(pairs) != expected_pairs:
            problems.append(f"{len(pairs)} unique model-domain pairs (expected {expected_pairs})")
        if require_one_row_per_pair and len(df) != expected_pairs:
            dup = (
                df.groupby(["_model", "_dataset"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            dup = dup[dup > 1]
            problems.append(
                f"{len(df)} rows after filtering (expected exactly {expected_pairs}); "
                f"duplicated model-domain cells: {dup.head(12).to_dict()}"
            )

        if problems:
            self.warn(f"{label}: " + "; ".join(problems))


def audit_fig2(a: Audit) -> None:
    a.r.header("FIGURE 2 — Stability vs. individual belief probability")
    file_model = "outputs/analysis/stability_vs_credence/model_summary.parquet"
    file_cross = "outputs/analysis/stability_vs_credence/cross_model_agreement.parquet"
    file_perm = "outputs/analysis/stability_vs_credence/permutation_summary.parquet"

    ms = a.filter_primary(a.load(file_model), file_model)
    a.check_model_domain_grid(ms, "Figure 2 model summary")

    a.r.say(
        f"Filtered model_summary rows: {len(ms)}; "
        f"models={ms['_model'].nunique() if '_model' in ms else 'NA'}; "
        f"domains={ms['_dataset'].nunique() if '_dataset' in ms else 'NA'}"
    )
    if model_col_dbg := a.selected_columns.get(file_model, {}).get("model"):
        n_prefixed = int(ms[model_col_dbg].astype(str).str.startswith("_").sum())
        a.r.say(
            f"Instruction-name check: {n_prefixed}/{len(ms)} surviving rows "
            'use the required leading "_" convention.'
        )
    if model_col_dbg is not None:
        a.r.say(
            "Raw model values used: "
            + ", ".join(sorted(ms[model_col_dbg].astype(str).unique().tolist()))
        )
    probe_col_dbg = a.selected_columns.get(file_model, {}).get("probe")
    if probe_col_dbg is not None:
        a.r.say(
            "Probe values used: "
            + ", ".join(sorted(ms[probe_col_dbg].astype(str).unique().tolist()))
        )
    est_col_dbg = a.selected_columns.get(file_model, {}).get("estimator")
    if est_col_dbg is not None:
        a.r.say(
            "Estimator values used: "
            + ", ".join(sorted(ms[est_col_dbg].astype(str).unique().tolist()))
        )

    r2_col = pick_col(
        ms,
        ["cv_r_squared", "cv_r2", "heldout_r_squared", "held_out_r_squared", "r_squared", "r2"],
        label="Figure 2 held-out R^2",
        numeric=True,
    )
    a.select(file_model, "heldout_r2", r2_col)

    r2 = ms.groupby("_dataset")[r2_col].median()
    a.results["fig2_median_r2"] = r2.to_dict()

    a.r.say("Median held-out R^2 across the 12 instruction-tuned models:")
    for d in DATASET_ORDER:
        x = float(r2.loc[d])
        a.r.say(f"  {DATASET_LABELS[d]}: {x:.6f}  (manuscript-ready: {x:.2f})")

    cross = a.filter_primary(a.load(file_cross), file_cross, model_required=False)
    rho_col = pick_col(
        cross,
        [
            "residual_gamma_spearman_rho",
            "residual_spearman_rho",
            "spearman_rho",
            "rho",
        ],
        label="Figure 2 residual Spearman rho",
        numeric=True,
    )
    a.select(file_cross, "residual_rho", rho_col)
    valid = cross.loc[cross[rho_col].notna()].copy()
    rho_median = valid.groupby("_dataset")[rho_col].median()
    pos_n = int((valid[rho_col] > 0).sum())
    total_n = int(valid[rho_col].notna().sum())
    pos_pct = 100.0 * pos_n / total_n if total_n else np.nan

    a.results["fig2_residual_rho_median"] = rho_median.to_dict()
    a.results["fig2_positive_residual"] = {
        "n_positive": pos_n, "n_total": total_n, "percent": pos_pct
    }

    a.r.say()
    a.r.say("Median cross-model residual Spearman rho:")
    for d in DATASET_ORDER:
        x = float(rho_median.loc[d])
        a.r.say(f"  {DATASET_LABELS[d]}: {x:.6f}  (manuscript-ready: {x:.2f})")

    a.r.say()
    a.r.say(
        f"Positive residual correlations: {pos_n}/{total_n} = {pos_pct:.6f}% "
        f"(manuscript-ready: {pos_pct:.1f}%)"
    )

    if total_n != 198:
        a.warn(
            f"Figure 2 residual agreement: expected 198 valid pairwise correlations "
            f"(66 model pairs × 3 domains), found {total_n}."
        )

    perm_path = a.root / file_perm
    if perm_path.exists():
        perm = a.filter_primary(
            a.load(file_perm),
            file_perm,
            instruction_only=False,
            model_required=False,
            filter_estimator=True,
            filter_probe=True,
        )

        if "_dataset" not in perm:
            raise RuntimeError(
                f"{file_perm}: no dataset/domain column remains after filtering."
            )

        per_domain_counts = perm.groupby("_dataset", dropna=False).size()
        bad_counts = {
            str(k): int(v)
            for k, v in per_domain_counts.items()
            if k in DATASET_ORDER and int(v) != 1
        }
        missing_domains = [d for d in DATASET_ORDER if d not in set(perm["_dataset"])]

        if bad_counts or missing_domains:
            raise RuntimeError(
                f"{file_perm}: expected exactly one aggregate permutation-summary row "
                f"per manuscript domain after filtering to probe={PRIMARY_PROBE} and "
                f"estimator={PRIMARY_ESTIMATOR}. Counts={per_domain_counts.to_dict()}, "
                f"missing={missing_domains}. This aggregate file does not retain model "
                f"identities, so its instruction/base population must be verified from "
                f"the generating analysis configuration before using its p-values."
            )

        a.r.say(
            "Permutation-summary note: this aggregate file does not retain model names; "
            "probe/estimator/domain uniqueness is verified here, while the 12-model "
            "instruction-tuned population is verified from the model-level Figure 2 "
            "outputs above."
        )

        p_col = pick_col(
            perm,
            ["p_value", "pvalue", "monte_carlo_p_value", "permutation_p_value", "p"],
            label="Figure 2 permutation p-value",
            required=False,
            numeric=True,
        )
        if p_col and "_dataset" in perm:
            a.select(file_perm, "permutation_p_value", p_col)
            pvals = perm.groupby("_dataset")[p_col].first()
            a.results["fig2_permutation_p"] = pvals.to_dict()
            a.r.say()
            a.r.say("Permutation-null p-values:")
            for d in DATASET_ORDER:
                if d in pvals.index:
                    a.r.say(f"  {DATASET_LABELS[d]}: {float(pvals.loc[d]):.6g}")

def audit_fig3(a: Audit) -> None:
    a.r.header("FIGURE 3 — Approximate CCK coherence")
    file_model = "outputs/analysis/coherence/model_summary.parquet"

    df = a.filter_primary(a.load(file_model), file_model)
    a.check_model_domain_grid(df, "Figure 3 coherence model summary")
    a.r.say(
        f"Filtered coherence rows: {len(df)}; "
        f"models={df['_model'].nunique() if '_model' in df else 'NA'}; "
        f"domains={df['_dataset'].nunique() if '_dataset' in df else 'NA'}"
    )
    model_col_dbg = a.selected_columns.get(file_model, {}).get("model")
    if model_col_dbg is not None:
        n_prefixed = int(df[model_col_dbg].astype(str).str.startswith("_").sum())
        a.r.say(
            f"Instruction-name check: {n_prefixed}/{len(df)} surviving rows "
            'use the required leading "_" convention.'
        )

    d_col = pick_col(
        df,
        [
            "median_d_cck",
            "d_cck_median",
            "median_dcck",
            "dcck_median",
            "median_distance_to_coherence",
            "median_projection_rmse",
            "median_rms_adjustment",
        ],
        label="Figure 3 model-level median d_CCK",
        numeric=True,
    )
    a.select(file_model, "model_median_dcck", d_col)

    med = df.groupby("_dataset")[d_col].median()
    a.results["fig3_median_model_median_dcck"] = med.to_dict()

    a.r.say("Median across models of each model's median d_CCK:")
    for d in DATASET_ORDER:
        x = float(med.loc[d])
        a.r.say(f"  {DATASET_LABELS[d]}: {x:.6f}  (manuscript-ready: {x:.3f})")

    exact_col = pick_col(
        df,
        [
            "exact_coherence_rate",
            "exact_rate",
            "fraction_exact_coherent",
            "proportion_exact_coherent",
            "pct_exact_coherent",
            "percent_exact_coherent",
            "fraction_exactly_coherent",
        ],
        label="exact-coherence rate",
        required=False,
        numeric=True,
    )
    if exact_col:
        a.select(file_model, "exact_coherence_rate", exact_col)
        vals = df[exact_col].dropna().astype(float)
        if len(vals):
            # Convert fractions to percentages; preserve an already-percent scale.
            max_raw = float(vals.max())
            max_pct = max_raw * 100.0 if max_raw <= 1.0 else max_raw
            a.results["fig3_max_exact_coherence_percent"] = max_pct
            a.r.say(
                f"Maximum model-domain exact-coherence rate: {max_pct:.6f}% "
                f"(rounds to {max_pct:.2f}%)"
            )

def audit_fig4(a: Audit, outdir: Path) -> None:
    a.r.header("FIGURE 4 — Domain and model-scale characterization")
    file_model = "outputs/analysis/stability_variation/model_summary.parquet"

    df = a.filter_primary(a.load(file_model), file_model)
    a.check_model_domain_grid(df, "Figure 4 stability-variation model summary")

    mean_col = pick_col(
        df,
        [
            "mean_gamma",
            "gamma_mean",
            "mean_stability",
            "mean_graded_stability",
            "model_mean_gamma",
        ],
        label="Figure 4 model-level mean gamma",
        numeric=True,
    )
    median_col = pick_col(
        df,
        [
            "median_gamma",
            "gamma_median",
            "median_stability",
            "median_graded_stability",
            "model_median_gamma",
        ],
        label="Figure 4 model-level median gamma",
        numeric=True,
    )
    a.select(file_model, "model_mean_gamma", mean_col)
    a.select(file_model, "model_median_gamma", median_col)

    median_of_means = df.groupby("_dataset")[mean_col].median()
    median_of_medians = df.groupby("_dataset")[median_col].median()
    a.results["fig4_median_of_model_means"] = median_of_means.to_dict()
    a.results["fig4_median_of_model_medians"] = median_of_medians.to_dict()

    a.r.say("DOMAIN SUMMARY — both aggregations are shown explicitly:")
    for d in DATASET_ORDER:
        mom = float(median_of_means.loc[d])
        momed = float(median_of_medians.loc[d])
        a.r.say(
            f"  {DATASET_LABELS[d]}:\n"
            f"    median(model MEAN gamma)   = {mom:.6f}  (manuscript-ready: {mom:.2f})\n"
            f"    median(model MEDIAN gamma) = {momed:.6f}  (manuscript-ready: {momed:.2f})"
        )

    piv = df.pivot_table(index="_model", columns="_dataset", values=median_col, aggfunc="first")
    missing = [d for d in DATASET_ORDER if d not in piv.columns]
    if missing:
        raise RuntimeError(f"Figure 4 domain-winner audit missing domains: {missing}")

    city = piv["cities_loc"]
    other_max = piv[["med_indications", "defs"]].max(axis=1)
    city_strict = city > other_max
    city_tied = city == other_max
    n_city_strict = int(city_strict.sum())
    tied_models = piv.index[city_tied].tolist()
    exceptions = piv.index[~city_strict].tolist()

    a.results["fig4_city_highest"] = {
        "strict_n": n_city_strict,
        "n_models": int(len(piv)),
        "ties": tied_models,
        "not_strictly_highest": exceptions,
    }
    a.r.say()
    a.r.say(
        f"City Locations strictly highest model-level MEDIAN stability: "
        f"{n_city_strict}/{len(piv)}"
    )
    a.r.say(f"Models where City is not strictly highest: {exceptions}")
    if tied_models:
        a.r.say(f"Models with a City tie for highest: {tied_models}")

    # Scale analysis from the same model-level MEAN gamma used by the Methods.
    scale_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        model = row["_model"]
        if model not in PARAM_B:
            raise KeyError(
                f"Figure 4: model {model!r} is not in PARAM_B. "
                "Update PARAM_B so parameter-count analyses are explicit."
            )
        figure_rows.append({
            "dataset": row["_dataset"],
            "dataset_label": DATASET_LABELS.get(row["_dataset"], row["_dataset"]),
            "model": model,
            "family": model_family(model),
            "parameters_b": PARAM_B[model],
            "mean_gamma": float(row[mean_col]),
            "median_gamma": float(row[median_col]),
        })

    fig4_table = pd.DataFrame(figure_rows).sort_values(
        ["dataset", "family", "parameters_b"]
    )
    fig4_csv = outdir / "figure4_parameter_count_source.csv"
    fig4_table.to_csv(fig4_csv, index=False)

    for d in DATASET_ORDER:
        dd = fig4_table.loc[fig4_table["dataset"] == d]
        for fam in FAMILY_ORDER:
            g = dd.loc[dd["family"] == fam].sort_values("parameters_b")
            if len(g) != 3:
                a.warn(
                    f"Figure 4 scale audit: expected 3 instruction-tuned {fam} models "
                    f"for {d}, found {len(g)}."
                )
                continue
            vals = g["mean_gamma"].to_numpy(float)
            params = g["parameters_b"].to_numpy(float)
            models = g["model"].tolist()
            delta = float(vals[-1] - vals[0])
            strictly_monotonic = bool(np.all(np.diff(vals) > 0))
            nondecreasing = bool(np.all(np.diff(vals) >= 0))
            scale_rows.append({
                "dataset": d,
                "family": fam,
                "small_model": models[0],
                "small_params_b": float(params[0]),
                "small_mean_gamma": float(vals[0]),
                "medium_model": models[1],
                "medium_params_b": float(params[1]),
                "medium_mean_gamma": float(vals[1]),
                "large_model": models[2],
                "large_params_b": float(params[2]),
                "large_mean_gamma": float(vals[2]),
                "large_minus_small": delta,
                "large_gt_small": bool(delta > 0),
                "strictly_monotonic": strictly_monotonic,
                "nondecreasing": nondecreasing,
            })

    scale = pd.DataFrame(scale_rows)
    a.results["fig4_scale_rows"] = scale.to_dict(orient="records")

    a.r.say()
    a.r.say("SCALE / PARAMETER-COUNT SUMMARY (using model MEAN gamma):")
    for d in DATASET_ORDER:
        g = scale.loc[scale["dataset"] == d]
        n_gt = int(g["large_gt_small"].sum())
        n_mono = int(g["strictly_monotonic"].sum())
        n_nondecr = int(g["nondecreasing"].sum())
        med_delta = float(g["large_minus_small"].median())

        a.results.setdefault("fig4_scale_summary", {})[d] = {
            "large_gt_small_n": n_gt,
            "n_families": int(len(g)),
            "median_large_minus_small": med_delta,
            "strictly_monotonic_n": n_mono,
            "nondecreasing_n": n_nondecr,
        }

        a.r.say(f"  {DATASET_LABELS[d]}:")
        a.r.say(f"    Large > Small: {n_gt}/{len(g)}")
        a.r.say(
            f"    median(Large - Small): {med_delta:.6f} "
            f"(manuscript-ready: {med_delta:.3f})"
        )
        a.r.say(f"    strictly increasing Small→Medium→Large: {n_mono}/{len(g)}")
        if n_nondecr != n_mono:
            a.r.say(f"    nondecreasing (allows ties): {n_nondecr}/{len(g)}")

    a.r.say()
    a.r.say(
        f"Saved exact parameter-count plotting/audit table to: "
        f"{fig4_csv.relative_to(a.root) if fig4_csv.is_relative_to(a.root) else fig4_csv}"
    )


def build_behavior_model_effects(a: Audit) -> tuple[pd.DataFrame, str, str]:

    source_file = "outputs/analysis/behavioral_resilience/matched_unit_summary.parquet"
    df = a.load(source_file).copy()

    required = [
        "model_name",
        "dataset",
        "probe",
        "sample",
        "estimator",
        "mean_movement_difference",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{source_file}: missing required Figure 5 columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    a.select(source_file, "model", "model_name")
    a.select(source_file, "dataset", "dataset")
    a.select(source_file, "probe", "probe")
    a.select(source_file, "sample", "sample")
    a.select(source_file, "estimator", "estimator")
    a.select(source_file, "model_mean_delta_m", "mean_movement_difference")

    expected_raw_models = {f"_{m}" for m in PARAM_B}

    out = df.loc[
        df["probe"].astype(str).eq(PRIMARY_PROBE)
        & df["sample"].astype(str).eq("round0_agreement")
        & df["dataset"].astype(str).isin(DATASET_ORDER)
        & df["estimator"].astype(str).eq("conditional")
        & df["model_name"].astype(str).isin(expected_raw_models)
    ].copy()

    out["mean_movement_difference"] = pd.to_numeric(
        out["mean_movement_difference"], errors="coerce"
    )
    out = out.loc[
        np.isfinite(out["mean_movement_difference"].to_numpy(dtype=float))
    ].copy()

    key_cols = ["model_name", "dataset", "probe", "sample", "estimator"]
    duplicates = out.duplicated(key_cols, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "Figure 5 audit found duplicate rows for the plotting key:\n"
            + out.loc[duplicates, key_cols].sort_values(key_cols).to_string(index=False)
        )

    out["_model"] = out["model_name"].map(canonical_model)
    out["_dataset"] = out["dataset"].map(canonical_dataset)

    observed_raw_models = set(out["model_name"].astype(str))
    missing_models = sorted(expected_raw_models - observed_raw_models)
    unexpected_models = sorted(observed_raw_models - expected_raw_models)
    if missing_models or unexpected_models:
        raise RuntimeError(
            "Figure 5 model population does not match plot_fig5.py. "
            f"Missing={missing_models}; unexpected={unexpected_models}"
        )

    if not out["model_name"].astype(str).str.startswith("_").all():
        bad = sorted(
            out.loc[
                ~out["model_name"].astype(str).str.startswith("_"),
                "model_name",
            ].astype(str).unique()
        )
        raise RuntimeError(
            f"Figure 5 contains non-instruction model names (missing leading '_'): {bad}"
        )

    a.r.say(
        "Figure 5 filter (copied from plot_fig5.py): "
        "probe=sawmil; sample=round0_agreement; estimator=conditional "
        "(repository name for Direct Conditional); "
        "12 leading-underscore instruction-tuned models."
    )
    a.r.say(
        f"Filtered matched_unit_summary rows: {len(out)}; "
        f"models={out['_model'].nunique()}; domains={out['_dataset'].nunique()}"
    )
    a.r.say(
        f'Instruction-name check: '
        f'{int(out["model_name"].astype(str).str.startswith("_").sum())}/{len(out)} '
        'surviving rows use the required leading "_" convention.'
    )

    return out, "mean_movement_difference", source_file


def audit_fig5(a: Audit) -> None:
    a.r.header("FIGURE 5 — Behavioral resilience")
    df, effect_col, source_file = build_behavior_model_effects(a)
    a.check_model_domain_grid(df, "Figure 5 model effects")

    effects = df[["_model", "_dataset", effect_col]].copy()
    effects = effects.dropna(subset=[effect_col])

    rows = {}
    total_pos = 0
    total = 0
    a.r.say(f"Model-level effect source: {source_file}")
    a.r.say(f"Effect column used: {effect_col}")
    a.r.say()
    a.r.say("Positive model-level mean Delta M and median effect by domain:")

    for d in DATASET_ORDER:
        vals = effects.loc[effects["_dataset"] == d, effect_col].astype(float)
        n_pos = int((vals > 0).sum())
        n_zero = int((vals == 0).sum())
        n = int(vals.notna().sum())
        med = float(vals.median())
        total_pos += n_pos
        total += n
        rows[d] = {
            "n_positive": n_pos,
            "n_zero": n_zero,
            "n_total": n,
            "median_model_mean_delta_m": med,
        }
        a.r.say(
            f"  {DATASET_LABELS[d]}: positive={n_pos}/{n}, zero={n_zero}; "
            f"median(model mean Delta M)={med:.6f} "
            f"(manuscript-ready: {med:.3f})"
        )

    pct = 100.0 * total_pos / total if total else np.nan
    a.results["fig5_behavioral"] = {
        "source_file": source_file,
        "probe": PRIMARY_PROBE,
        "sample": "round0_agreement",
        "estimator_repository_name": "conditional",
        "estimator_manuscript_name": "Direct Conditional",
        "by_domain": rows,
        "overall_positive_n": total_pos,
        "overall_n": total,
        "overall_positive_percent": pct,
    }

    a.r.say()
    a.r.say(
        f"Overall positive model-domain effects: {total_pos}/{total} = {pct:.6f}% "
        f"(manuscript-ready: {pct:.1f}%)"
    )

    if total != 36:
        a.warn(f"Figure 5: expected 36 model-domain effects, found {total}.")

    # Print the exact 36 values plotted in Figure 5.
    piv = effects.pivot(index="_model", columns="_dataset", values=effect_col)
    ordered_models = sorted(
        piv.index,
        key=lambda m: (FAMILY_ORDER.index(model_family(m)), PARAM_B.get(m, 999)),
    )
    piv = piv.reindex(ordered_models)
    piv = piv.reindex(columns=DATASET_ORDER)
    a.r.say()
    a.r.say("All model-level mean Delta M values used for Figure 5:")
    a.r.say(
        piv.rename(columns=DATASET_LABELS).to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

def audit_direct_joint(a: Audit) -> None:
    a.r.header("SI — Direct–Joint operationalization agreement")
    file_model = "outputs/analysis/operationalization_agreement/model_summary.parquet"

    df = a.filter_primary(
        a.load(file_model),
        file_model,
        instruction_only=True,
        filter_probe=True,
        filter_estimator=False,
    )
    a.check_model_domain_grid(df, "Direct–Joint agreement")

    rho_col = "raw_spearman_rho"
    if rho_col not in df.columns:
        raise RuntimeError(
            f"{file_model}: expected {rho_col!r} for the manuscript's "
            "Direct–Joint rank-agreement statistic, but it is absent. "
            f"Available columns: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[rho_col]):
        raise RuntimeError(
            f"{file_model}: {rho_col!r} exists but is not numeric."
        )

    a.select(file_model, "direct_joint_raw_spearman_rho", rho_col)
    med = df.groupby("_dataset")[rho_col].median()
    a.results["si_direct_joint_median_rho"] = med.to_dict()

    a.r.say(
        "Agreement column used: raw_spearman_rho "
        "(raw Direct gamma vs. raw Joint gamma rankings)"
    )
    a.r.say("Median Direct–Joint Spearman rho across models:")
    for d in DATASET_ORDER:
        x = float(med.loc[d])
        a.r.say(f"  {DATASET_LABELS[d]}: {x:.6f}  (manuscript-ready: {x:.2f})")


def build_instruction_effects(a: Audit) -> tuple[pd.DataFrame, str, str]:
    pair_summary = "outputs/analysis/stability_variation/instruction_pair_summary.parquet"
    statement_file = "outputs/analysis/stability_variation/instruction_statement_deltas.parquet"

    if (a.root / pair_summary).exists():
        df = a.filter_primary(
            a.load(pair_summary),
            pair_summary,
            instruction_only=False,  # rows already represent base/instruct PAIRS
            filter_probe=True,
            filter_estimator=True,
            model_required=False,
        )
        delta_col = pick_col(
            df,
            [
                "mean_delta_gamma",
                "delta_gamma_mean",
                "mean_instruction_effect",
                "delta_inst",
                "instruction_effect",
                "mean_stability_delta",
            ],
            label="base-vs-instruction model-pair mean Delta gamma",
            required=False,
            numeric=True,
        )
        if delta_col:
            a.select(pair_summary, "mean_instruction_delta_gamma", delta_col)
            return df, delta_col, pair_summary

    df = a.filter_primary(
        a.load(statement_file),
        statement_file,
        instruction_only=False,
        filter_probe=True,
        filter_estimator=True,
        model_required=False,
    )
    delta_col = pick_col(
        df,
        [
            "delta_gamma",
            "instruction_delta_gamma",
            "gamma_difference",
            "stability_delta",
        ],
        label="base-vs-instruction statement-level Delta gamma",
        numeric=True,
    )
    a.select(statement_file, "statement_instruction_delta_gamma", delta_col)

    pair_key = pick_col(
        df,
        [
            "model_pair",
            "pair_id",
            "model",
            "model_name",
            "instruction_model",
            "instruct_model",
        ],
        label="base-vs-instruction model-pair identifier",
        required=True,
    )
    a.select(statement_file, "instruction_pair_key", pair_key)

    if "_dataset" not in df:
        raise RuntimeError("Instruction statement deltas lack a dataset/domain column.")

    grouped = (
        df.groupby([pair_key, "_dataset"], as_index=False)[delta_col]
        .mean()
        .rename(columns={delta_col: "_mean_delta_gamma"})
    )
    grouped["_pair"] = grouped[pair_key].astype(str)
    return grouped, "_mean_delta_gamma", statement_file


def audit_base_instruction(a: Audit) -> None:
    a.r.header("SI — Base vs. instruction-tuned stability")
    df, delta_col, source_file = build_instruction_effects(a)

    vals = df[delta_col].dropna().astype(float)
    n_pos = int((vals > 0).sum())
    n_zero = int((vals == 0).sum())
    n = int(vals.notna().sum())
    pct = 100.0 * n_pos / n if n else np.nan

    a.results["si_instruction_tuning"] = {
        "n_positive": n_pos,
        "n_zero": n_zero,
        "n_total": n,
        "positive_percent": pct,
    }

    a.r.say(f"Effect source: {source_file}")
    a.r.say(f"Delta column used: {delta_col}")
    a.r.say(
        f"Positive base→instruction matched comparisons: {n_pos}/{n} = {pct:.6f}% "
        f"(manuscript-ready: {pct:.1f}%; zeros={n_zero})"
    )
    if n != 36:
        a.warn(f"Base-vs-instruction audit: expected 36 model-domain comparisons, found {n}.")

    if "_dataset" in df:
        a.r.say("By domain:")
        for d in DATASET_ORDER:
            vv = df.loc[df["_dataset"] == d, delta_col].dropna().astype(float)
            if len(vv):
                a.r.say(
                    f"  {DATASET_LABELS[d]}: {(vv > 0).sum()}/{len(vv)} positive; "
                    f"median Delta_inst={vv.median():.6f}"
                )


def audit_matching_captions(a: Audit) -> None:
    a.r.header("SI — Behavioral matching-quality caption numbers")
    file_match = "outputs/analysis/behavioral_matching/matching_summary_sawmil_all.parquet"
    if not (a.root / file_match).exists():
        a.r.say(f"SKIP: {file_match} not found.")
        return

    df = a.filter_primary(
        a.load(file_match),
        file_match,
        instruction_only=True,
        filter_probe=False,  # filename already says sawmil; preserve if probe column absent
        filter_estimator=True,
    )

    prob_col = pick_col(
        df,
        [
            "mean_abs_prob_diff",
            "mean_abs_probability_diff",
            "mean_abs_pi_t_diff",
            "mean_abs_delta_pi_t",
            "mean_abs_credence_diff",
        ],
        label="matching mean absolute belief-probability difference",
        required=False,
        numeric=True,
    )
    gamma_col = pick_col(
        df,
        [
            "mean_abs_gamma_diff",
            "mean_abs_stability_diff",
            "mean_abs_delta_gamma",
        ],
        label="matching mean absolute stability difference",
        required=False,
        numeric=True,
    )

    if prob_col is None or gamma_col is None:
        a.r.say("Could not identify both matching-caption summary columns.")
        a.r.say(f"Available columns: {list(df.columns)}")
        return

    a.select(file_match, "mean_abs_probability_difference", prob_col)
    a.select(file_match, "mean_abs_gamma_difference", gamma_col)

    out = {}
    for d in DATASET_ORDER:
        g = df.loc[df["_dataset"] == d]
        if g.empty:
            continue
        max_prob = float(g[prob_col].max())
        min_gamma = float(g[gamma_col].min())
        max_gamma = float(g[gamma_col].max())
        out[d] = {
            "max_model_mean_abs_probability_gap": max_prob,
            "min_model_mean_abs_gamma_gap": min_gamma,
            "max_model_mean_abs_gamma_gap": max_gamma,
        }
        a.r.say(
            f"  {DATASET_LABELS[d]}: max mean |Delta pi_T|={max_prob:.5f}; "
            f"mean |Delta gamma| range={min_gamma:.3f}–{max_gamma:.3f}"
        )
    a.results["si_matching_caption_numbers"] = out
    a.r.say(
        "NOTE: these summarize the CURRENT matching output only. They do not validate "
        "that the matching implementation matches the Methods description."
    )


def audit_joint_exclusions(a: Audit) -> None:
    a.r.header("SI / Methods — Joint-to-Conditional excluded-pair rate")
    file_cov = "outputs/analysis/conditional_coverage/coverage.parquet"
    if not (a.root / file_cov).exists():
        a.r.say(f"SKIP: {file_cov} not found.")
        return

    df = a.filter_primary(
        a.load(file_cov),
        file_cov,
        instruction_only=True,
        filter_probe=True,
        filter_estimator=False,
    )

    candidate_col = pick_col(
        df,
        ["candidate_pairs", "n_candidate_pairs", "candidate", "n_candidate"],
        label="conditional coverage candidate-pair count",
        required=False,
        numeric=True,
    )
    excluded_col = pick_col(
        df,
        ["joint_excluded", "n_joint_excluded", "excluded_joint_pairs", "joint_excluded_pairs"],
        label="Joint excluded-pair count",
        required=False,
        numeric=True,
    )
    valid_joint_col = pick_col(
        df,
        ["joint_pairs", "n_joint_pairs", "joint_valid", "n_joint_valid", "valid_joint_pairs"],
        label="valid Joint pair count",
        required=False,
        numeric=True,
    )

    if candidate_col is None:
        a.r.say(f"Could not identify candidate-pair count. Columns: {list(df.columns)}")
        return

    if excluded_col is None and valid_joint_col is None:
        a.r.say(f"Could not identify Joint excluded or valid count. Columns: {list(df.columns)}")
        return

    candidates = float(df[candidate_col].sum())
    if excluded_col is not None:
        excluded = float(df[excluded_col].sum())
        a.select(file_cov, "joint_excluded_pairs", excluded_col)
    else:
        excluded = float((df[candidate_col] - df[valid_joint_col]).sum())
        a.select(file_cov, "joint_valid_pairs", valid_joint_col)

    a.select(file_cov, "candidate_pairs", candidate_col)
    pct = 100.0 * excluded / candidates if candidates else np.nan
    a.results["joint_exclusion_rate"] = {
        "candidate_pairs": int(candidates),
        "excluded_pairs": int(excluded),
        "percent": pct,
    }

    a.r.say(f"Candidate pairs: {int(candidates):,}")
    a.r.say(f"Excluded Joint pairs: {int(excluded):,}")
    a.r.say(f"Pooled excluded rate: {pct:.6f}%")
    a.r.say("Current Methods value flagged by the review: 0.042%.")


def print_source_map(a: Audit) -> None:
    a.r.header("SOURCE MAP — analyses, plotting scripts, figures, and result files")
    for name, info in SOURCE_MAP.items():
        a.r.say(name)
        for key, val in info.items():
            if isinstance(val, list):
                a.r.say(f"  {key}:")
                for x in val:
                    exists = (a.root / x).exists()
                    a.r.say(f"    {'[x]' if exists else '[ ]'} {x}")
            else:
                exists = (a.root / val).exists()
                a.r.say(f"  {key}: {'[x]' if exists else '[ ]'} {val}")
        a.r.say()


def print_provenance(a: Audit) -> None:
    a.r.header("PROVENANCE — exact files actually read")
    for rel, obj in sorted(a.loaded.items()):
        stat = obj.path.stat()
        a.r.say(f"{rel}")
        a.r.say(f"  rows × cols: {obj.df.shape[0]} × {obj.df.shape[1]}")
        a.r.say(f"  mtime: {stat.st_mtime}")
        a.r.say(f"  sha256: {obj.sha256}")
        if rel in a.selected_columns:
            a.r.say(f"  selected columns: {a.selected_columns[rel]}")


def print_warnings(a: Audit) -> None:
    a.r.header("AUDIT WARNINGS")
    if not a.warnings:
        a.r.say("No audit warnings.")
    else:
        for w in a.warnings:
            a.r.say(f"- {w}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute manuscript-facing numerical summaries from current analysis outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/analysis/manuscript_audit"),
        help="Output directory relative to --root unless absolute.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Turn unexpected row-count/schema warnings into hard errors.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    outdir = args.outdir.expanduser()
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter()
    audit = Audit(root=root, strict=args.strict, reporter=reporter)

    reporter.header("GRADED STABILITY MANUSCRIPT NUMERICAL AUDIT")
    reporter.say(f"Repository root: {root}")
    reporter.say(f"Primary probe: {PRIMARY_PROBE}")
    reporter.say(f"Primary estimator: {PRIMARY_ESTIMATOR}")
    reporter.say("Primary model population: 12 instruction-tuned LLMs")
    reporter.say(
        "IMPORTANT: recomputed values below come from the current parquet outputs, "
        "not from reading values off plotted PDFs."
    )

    print_source_map(audit)

    # Main-paper audit.
    audit_fig2(audit)
    audit_fig3(audit)
    audit_fig4(audit, outdir)
    audit_fig5(audit)

    # SI / related reporting gaps.
    audit_direct_joint(audit)
    audit_base_instruction(audit)
    audit_matching_captions(audit)
    audit_joint_exclusions(audit)

    print_provenance(audit)
    print_warnings(audit)

    report_path = outdir / "manuscript_numbers.txt"
    report_path.write_text(reporter.text(), encoding="utf-8")

    json_path = outdir / "manuscript_numbers.json"
    payload = {
        "primary_analysis": {
            "probe": PRIMARY_PROBE,
            "estimator": PRIMARY_ESTIMATOR,
            "model_population": "instruction-tuned",
            "expected_models": 12,
            "datasets": DATASET_ORDER,
        },
        "recomputed": audit.results,
        "selected_columns": audit.selected_columns,
        "warnings": audit.warnings,
        "source_files": {
            rel: {
                "path": str(obj.path),
                "rows": int(obj.df.shape[0]),
                "columns": int(obj.df.shape[1]),
                "sha256": obj.sha256,
            }
            for rel, obj in audit.loaded.items()
        },
    }
    json_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")

    reporter.say()
    reporter.say(f"Saved report: {report_path}")
    reporter.say(f"Saved JSON:   {json_path}")
    reporter.say(f"Saved Fig. 4 parameter table: {outdir / 'figure4_parameter_count_source.csv'}")


if __name__ == "__main__":
    main()
