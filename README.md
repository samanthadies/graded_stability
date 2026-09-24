# Toward a Graded Measure of Belief Stability in Large Language Models

This repository contains the code accompanying *Toward a Graded Measure of Belief Stability in Large Language Models*. We introduce **graded belief stability**, a statement-level measure of how broadly an LLM's belief in a proposition persists when considered alongside the model's other non-disbelieved propositions.

For a belief $P$, graded stability $\gamma_{\mathcal{M}}(P)$ is the proportion of conditioning propositions $x$ for which the model's estimated conditional probability of $P$ remains above its empirical belief threshold. The repository includes scripts for:

- Training calibrated **trivalent probes** over LLM hidden representations
- Identifying model-specific **belief**, **disbelief**, and **suspended-belief** states
- Constructing belief--non-disbelief pairs $(P,x)$
- Estimating conditional belief with the primary **Direct Conditional** estimator
- Computing the secondary **Joint-to-Conditional** estimator used for robustness analyses
- Computing proposition-level **graded belief stability**
- Evaluating stability beyond individual belief probability and testing **probabilistic coherence**
- Running the multi-round **behavioral challenge** experiment
- Reproducing main-text Figures 2--5 and the supplementary analyses

------------------------------------------------------------------------

### Environment Setup

The experiments were run with **Python 3.11.15**.

To recreate the full Conda environment used for the experiments, run:

```bash
conda env create -f environment.yml
conda activate lockean
```

For a lighter-weight installation using the direct Python dependencies:

```bash
conda create -n lockean python=3.11
conda activate lockean
pip install -r requirements.txt
```

`requirements.txt` contains the direct Python dependencies used by the repository, while `environment.yml` records the fuller Conda environment used for the experiments. An explicit Linux Conda package snapshot is also provided in `environment-explicit.txt` for archival reproducibility.

### Hugging Face Access

The experiments use open-weight models from **Gemma**, **Llama**, **Mistral**, and **Qwen**. Some checkpoints require accepting the corresponding Hugging Face model license and authenticating before the model can be downloaded.

Hugging Face access tokens can be created at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure the environment running the experiments is authenticated before launching model-scoring jobs.

------------------------------------------------------------------------

## Usage and Examples

Run commands from the repository root.

The three dataset keys used throughout the code are:

- `cities_loc` -- City Locations
- `med_indications` -- Medical Indications
- `defs` -- Word Definitions

The primary probe in the paper is `sawmil`; `svm` and `mean_difference` are used for robustness analyses. The primary conditional-probability estimator is **Direct Conditional**, while **Joint-to-Conditional** is used as a secondary operationalization.

**Model naming convention:** instruction-tuned checkpoints use a leading underscore in the repository configuration keys (e.g., `_llama-3.1-8b`), while the corresponding base checkpoint omits the underscore (e.g., `llama-3.1-8b`).

Generated model outputs are not included in the repository. By default, experimental artifacts are written under `outputs/`.

### 1. Select Probe Layers

Each probe is evaluated across transformer layers, and the selected layer minimizes held-out test log loss. The model configuration files in `configs/model/` already contain the selected layers used in the paper, so this step can be skipped when directly reproducing the reported experiments.

To reproduce the layer sweep for one model and dataset, for example `_llama-3.1-8b` on City Locations, run:

```bash
python -m scripts.probes.sweep_layers \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --resume
```

After completing the desired sweeps, summarize the best-performing layer for each model, dataset, and probe with:

```bash
python -m scripts.probes.select_layers --require_complete
```

Layer-sweep results are saved under `outputs/layer_sweep/`, and the selected-layer summary is written to:

```text
outputs/layer_sweep/selected_layers.parquet
```

The selected layers used by the downstream paper pipeline are stored explicitly in each `configs/model/<model>.yaml` file.

### 2. Estimate Atomic Beliefs

For each model and dataset, fit the atomic trivalent probes at their preselected layers and score the held-out statements:

```bash
python -m scripts.beliefs.score_atomic \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --resume
```

By default, the script evaluates all configured probes. To run only the primary `sawmil` probe, add:

```bash
--probes sawmil
```

Atomic outputs are saved under:

```text
outputs/atomic/<model>/<dataset>.parquet
outputs/atomic/<model>/<dataset>.joblib
```

The Parquet file contains calibrated probabilities over `True`, `False`, and `Neither` for the test statements.

### 3. Define Belief Sets and Build $(P,x)$ Pairs

Use the atomic predictions to identify each model's belief set, non-disbelief set, and empirical belief threshold:

```bash
python -m scripts.beliefs.define_sets \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --overwrite
```

Then construct the probe-specific belief--non-disbelief pairs used for conditional estimation:

```bash
python -m scripts.beliefs.build_pairs \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --overwrite
```

The pair tables are saved under:

```text
outputs/pairs/<model>/<dataset>.parquet
```

Self-pairs $P=x$ are excluded by default.

### 4. Estimate Conditional Belief

#### Direct Conditional

The primary estimator represents each pair as a conditional statement of the form *"Given $x$, $P$."* and trains a new trivalent probe to estimate the corresponding conditional belief distribution.

Run:

```bash
python -m scripts.beliefs.score_conditionals \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --resume
```

Outputs are written under:

```text
outputs/conditional/<model>/
```

#### Joint-to-Conditional

The secondary estimator probes the full ordered $3 \times 3$ joint epistemic state of *"$x$ and $P$."* and derives the conditional probability from the resulting joint distribution.

Run:

```bash
python -m scripts.beliefs.score_joint \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --resume
```

Outputs are written under:

```text
outputs/joint/<model>/
```

All main-text analyses use the Direct Conditional estimator. Joint-to-Conditional is used for supplementary robustness analyses.

### 5. Compute Graded Belief Stability

After atomic, pair, and conditional/joint outputs have been generated, compute proposition-level graded stability:

```bash
python -m scripts.beliefs.compute_gamma \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --sources conditional joint \
  --overwrite
```

To compute only the primary Direct Conditional measure, use:

```bash
python -m scripts.beliefs.compute_gamma \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --sources conditional \
  --overwrite
```

Graded-stability outputs are saved under:

```text
outputs/gamma/<model>/<dataset>.parquet
```

Here, `conditional` corresponds to the Direct Conditional estimator and `joint` corresponds to the Joint-to-Conditional estimator.

### 6. Run the Main Stability Analyses

Once graded stability has been computed across the desired models and datasets, run the manuscript-level analyses.

To quantify how much graded stability is explained by individual belief probability and measure cross-model agreement in the remaining residual structure:

```bash
python -m scripts.analysis.analyze_stability_vs_credence \
  --probe sawmil \
  --overwrite
```

To evaluate the distance of the measured probability systems from CCK probabilistic coherence:

```bash
python -m scripts.analysis.analyze_coherence \
  --probe sawmil \
  --overwrite
```

To characterize graded-stability variation across propositions, domains, models, instruction tuning, and model scale:

```bash
python -m scripts.analysis.analyze_stability_variation \
  --probe sawmil \
  --overwrite
```

To compare Direct Conditional and Joint-to-Conditional graded stability:

```bash
python -m scripts.analysis.analyze_operationalization_agreement \
  --probe sawmil \
  --overwrite
```

Analysis outputs are saved under:

```text
outputs/analysis/
```

### 7. Run the Behavioral Challenge Experiment

The behavioral validation compares conversational resilience among beliefs with similar individual belief probability but different graded stability.

First, construct both the general and probability-matched challenge sets:

```bash
python -m scripts.behavior.build_challenge_set \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --mode both \
  --overwrite
```

Then run the multi-round challenge experiment:

```bash
python -m scripts.behavior.run_challenge \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --resume
```

Analyze the completed model--dataset run with:

```bash
python -m scripts.behavior.analyze_challenge \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --overwrite
```

After all desired model--dataset combinations have been analyzed, aggregate the behavioral-resilience results:

```bash
python -m scripts.analysis.analyze_behavioral_resilience \
  --probe sawmil \
  --overwrite
```

Behavioral outputs are organized under:

```text
outputs/behavior/sets/
outputs/behavior/challenge/
outputs/behavior/analysis/
outputs/analysis/behavioral_resilience/
```

The challenge prompts and counterbalancing configuration are defined in `configs/experiments/challenge.yaml`.

### 8. Generate the Main-Text Figures

The data-driven main-text figures can be regenerated after the corresponding analyses have been completed.

**Figure 2 -- Stability vs. individual belief probability**

```bash
python -m scripts.plotting.plot_fig2 \
  --probe sawmil \
  --overwrite
```

**Figure 3 -- Distance from probabilistic coherence**

```bash
python -m scripts.plotting.plot_fig3 \
  --probe sawmil \
  --overwrite
```

**Figure 4 -- Variation in graded belief stability across domains**

```bash
python -m scripts.plotting.plot_fig4 \
  --probe sawmil \
  --overwrite
```

**Figure 5 -- Behavioral resilience**

```bash
python -m scripts.plotting.plot_fig5 \
  --probe sawmil \
  --overwrite
```

Figures are saved under:

```text
outputs/figures/
```

Supplementary plotting scripts follow the same organization:

- `plot_fig*_base.py` -- matched base-model analyses
- `plot_fig*_joint.py` -- Joint-to-Conditional analyses
- `plot_fig*_two_probes.py` -- SVM and Mass Mean probe robustness
- `plot_base_v_instruct_si.py` -- base vs. instruction-tuned comparison
- `plot_joint_ordering_si.py` -- conjunction-order sensitivity
- `plot_layersweep_si.py` -- layer-selection results
- `plot_model_scale_si.py` -- model-scale analysis
- `plot_operationalization_agreement_si.py` -- Direct vs. Joint agreement
- `plot_stability_probability_robustness_si.py` -- spline and residual-agreement robustness

Each plotting script contains a short description and example invocation at the top of the file.

### 9. Additional Supplementary Analyses and Tables

To test the Joint-to-Conditional estimator's sensitivity to conjunction order, first generate the alternate *"$P$ and $x$"* joint outputs:

```bash
python -m scripts.beliefs.score_joint \
  --model_name _llama-3.1-8b \
  --dataset cities_loc \
  --template P_then_x \
  --output_dir outputs/joint_ordering/P_then_x \
  --resume
```

Then aggregate the ordering analysis:

```bash
python -m scripts.analysis.analyze_joint_ordering \
  --probes sawmil \
  --overwrite
```

To generate supplementary behavioral-matching and coherence tables:

```bash
python -m scripts.analysis.make_behavioral_matching_tables \
  --probe sawmil \
  --sample round0_agreement \
  --overwrite

python -m scripts.analysis.make_coherence_tables \
  --probe sawmil \
  --overwrite
```

The manuscript-facing numerical audit can be rerun with:

```bash
python -m scripts.analysis.audit_manuscript_numbers --strict
```

### 10. Running the Full Experiment Matrix with Slurm

The repository includes Slurm generators under `slurms/generators/` for the full **24-model x 3-dataset** experiment matrix: the 12 instruction-tuned models used in the main analyses and their matched base checkpoints.

The generators mirror the pipeline above:

1. `generate_layer_sweep_slurms.py`
2. `generate_atomic_slurms.py`
3. `generate_pairs_slurms.py`
4. `generate_conditional_slurms.py`
5. `generate_joint_slurms.py`
6. `generate_gamma_slurms.py`
7. `generate_behavior_slurms.py`

`generate_joint_ordering_slurms.py` generates the additional conjunction-order sensitivity runs.

These generators were written for the authors' university cluster and include cluster-specific Slurm, module, Conda, and cache settings. **Inspect and adapt the generated-job configuration before using these scripts on another cluster.** The underlying `scripts/` commands above are not tied to Slurm.

------------------------------------------------------------------------

### Repository Structure

```text
graded_stability/
├── configs/                 # Model, probe, statement, and experiment configurations
├── data/                    # City Locations, Medical Indications, and Word Definitions data
├── scripts/
│   ├── probes/              # Layer sweeps and layer selection
│   ├── beliefs/             # Atomic, pair, conditional/joint, and gamma pipeline
│   ├── behavior/            # Behavioral challenge construction, scoring, and analysis
│   ├── analysis/            # Manuscript-level analyses and tables
│   └── plotting/            # Main-text and supplementary figures
├── slurms/
│   └── generators/          # Full experiment-matrix Slurm generators
├── stability/               # Reusable data, model, probe, belief, and analysis utilities
├── environment.yml          # Full Conda environment
├── environment-explicit.txt # Explicit Linux Conda package snapshot
└── requirements.txt         # Direct Python dependencies
```

------------------------------------------------------------------------

### Citations

1. Dies, S., Fitelson, B. & Eliassi-Rad, T. *Toward a Graded Measure of Belief Stability in Large Language Models* (2026). Accompanying manuscript.

2. Savcisens, G. & Eliassi-Rad, T. *Trilemma of Truth in Large Language Models*, **Mechanistic Interpretability Workshop at NeurIPS 2025**, [https://openreview.net/forum?id=z7dLG2ycRf](https://openreview.net/forum?id=z7dLG2ycRf) (2025).

3. Marks, S. & Tegmark, M. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*, **Proceedings of the 1st Conference on Language Modeling (COLM)**, [https://openreview.net/forum?id=aajyHYjjsk](https://openreview.net/forum?id=aajyHYjjsk) (2024).
