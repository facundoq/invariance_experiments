# AI Research Experimentation Agent

This guide outlines the standard operating procedures for conducting experiments within this project, based on the `guide.md` principles and our technical stack.

## 1. Planning & Design
- **Timeline & Milestones:** Define a clear schedule for the first functional version, evaluations, and iterations.
- **MVP First:** Prioritize a Minimum Viable Product (MVP) with a baseline metric before complex optimizations.
- **Datasets:** Favor established, documented, and validated datasets (Kaggle, Hugging Face, etc.).
- **Literature Review:** Conduct systematic searches to understand the state-of-the-art and avoid reinventing the wheel.
- **Experiment Log:** Maintain a detailed log (markdown or notebook) of ideas, hypotheses, and results.

## 2. Technical Stack
- **PyTorch:** Primary framework for model development and training.
- **TensorBoard:** Used for real-time visualization of training curves and model diagnostics.
- **MLflow:** Centralized system for experiment tracking, including:
  - Unique experiment and run identifiers.
  - Hyperparameters and training assets.
  - Metrics (Accuracy, F1, Loss, etc.).
  - Reproducibility (seeds, versioned scripts).

## 3. Configuration Management
- **Python-centric Config (Mandatory):** Define ALL experiment configurations, combinations, and hyperparameters directly in Python code (e.g., as a list of dictionaries or an OmegaConf object).
- **NO Command Line Arguments:** Do NOT use `argparse` or any command-line flags to control experiment logic. This ensures reproducibility and avoids brittle Bash scripts.
- **Environment Variables (`config.env`):**
  - Used ONLY for environment-specific infrastructure settings (e.g., `EXPERIMENT_HOST`).
  - **Important:** `config.env` is ignored by Git to protect sensitive environment-specific data.

## 4. Execution Workflow
### Local Execution
Standard runs are performed on the host specified in `config.env`.
1. **Smoke Test:** Run `uv run python test_models.py` to verify the code and models.
2. **Full Experiment:** Run `uv run python experiment.py` to execute the full experiment matrix.
3. **Visualization:** Run `uv run python dashboard.py` to launch the MLflow dashboard and visualize metrics, parameters, and logged models.

### Remote Execution
When running on a remote server (e.g., `EXPERIMENT_HOST` is not `localhost`):
1. **Sync Files:** The runner must `rsync` the project files to the target server.
   - **Exclude:** `.env`, `config.env`, `.git/`, and any other git-ignored files/folders.
2. **Environment Setup:** Update the remote environment using `uv sync`.
3. **Run:** Execute the experiment script on the remote server.

## 5. Evaluation & Results
- **Systematic Search:** Orderly exploration of hyperparameters (learning rate, batch size, etc.).
- **Ablation Studies:** Deactivate individual components to validate their contribution to the overall performance.
- **Visualizations:** Automate the generation of plots (preferring `plotnine` or `lets-plot`) and save them to `outputs/` for easy inclusion in reports.
- **Intermediate Data:** Save checkpoints and processed data to avoid redundant computations.
