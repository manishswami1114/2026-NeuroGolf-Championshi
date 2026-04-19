# 2026 NeuroGolf Championship Solver

This repository contains the codebase, Jupyter Notebooks, and data analysis pipelines for solving logic puzzles in the 2026 NeuroGolf Championship (similar to ARC-AGI problem sets).

## Repository Structure

Inside the `neurogolf-2026` directory, you will find:

- **Data Files (`task001.json` - `task400.json`)**: JSON representations of the 400 logic puzzles/tasks required for the championship.
- **Jupyter Notebooks (`neurogolf-v*.ipynb`)**: Progression of experimental models and solvers from version 2 up to version 9. `neurogolf-championship-2026-starter-notebook.ipynb` serves as an entry point.
- **Data Analytics (`_task_analysis_full.csv`, `task_rules_analysis.xlsx`)**: Extracted metadata summarizing task complexity, geometries, and underlying rules.
- **Solver Scripts (`solve.py`, `neurogolf-notebook.py`)**: Core algorithmic logic meant for inference and generating final solutions.
- **Analysis Modules (`_full_task_analysis.py`, `_analyze_*.py`)**: Python scripts designed to classify task properties, identifying which are solvable using geometric ONNX operations vs. machine intelligence models.
- **`neurogolf_utils/`**: Shared helper functions used across various notebooks and scripts.
- **`submission*/`**: Prepared directories containing output solutions for the corresponding versions.

## How to Work with the Repository

Follow these instructions to safely build upon this codebase.

### 1. Setup Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# From the repository root
python3 -m venv venv
source venv/bin/activate  # (On Windows use `venv\Scripts\activate`)

# Install common expected libraries
pip install numpy pandas jupyter openpyxl
```

### 2. Exploring the Tasks
To view how tasks are analyzed and categorized, you can run the full task analyzer:

```bash
cd neurogolf-2026
python _full_task_analysis.py
```
This updates analytical artifacts and explores the structure of the JSON formats, distinguishing between parametric and non-parametric rule candidates.

### 3. Iterating on Solvers
The experimental work occurs primarily in the Jupyter notebooks. Start up the Jupyter server to iterate on the pipeline:

```bash
cd neurogolf-2026
jupyter notebook
```
Review `neurogolf-v9.ipynb` for the latest architecture, and utilize `solve.py` for headless or automated testing against local validation data.

### 4. Running Validation / Smoke Tests
Various smoke testing and debugging files are present (`_smoke_v9.py`, `_debug_task001.py`, etc.). Run these to verify your local logic implementations against edge cases.

```bash
python _smoke_v9.py
```

## Contribution & Git Etiquette
- **Avoid tracking temporary or AI files**: A `.gitignore` is set up tracking `.claude`, `.DS_Store`, and `__pycache__`. Ensure that other environment files `.env` or IDE `.vscode`/`.idea` folders remain ignored.
- **Submissions**: If you're building a submission, export your results JSON and place it in the respective `submission_vX/` folder before archiving it.
