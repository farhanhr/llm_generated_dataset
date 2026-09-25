# LLM Synthetic Data Augmentation for Indonesian SMS Spam

This repository contains an end-to-end pipeline for generating, balancing, and evaluating synthetic Indonesian SMS spam data using Large Language Models (LLMs). The primary objective of this project is to address small data sets or class imbalance

## Directory Structure

*   `app/` : Web application module and interface (HTML templates).
*   `data/` : Dataset storage folder.
    *   `raw/` : Original Indonesian SMS spam dataset and train/test splits.
    *   `augmented/` : LLM output logs, synthetic texts, and the balanced merged data.
*   `notebooks/` : Jupyter Notebooks for research result visualization, table generation, and evaluation charts (Mako theme).
*   `src/` : Core module (source code).
    *   `augmenter.py` : Prompting logic and LLM API calls.
    *   `balancer_of_augmented_data.py` : Data processing and balancing.
    *   `classifier.py` : XGBoost training architecture.
    *   `evaluator.py` : NLP metric calculations (NLTK, Sentence-Transformers, ROUGE).
*   `main_*.py` : Main executor scripts to run each pipeline stage separately.

## Usage Guide

**1. Dependency Installation**
Ensure you are using Python 3.8+ and activate your virtual environment. Install all required libraries:
```bash
pip install -r requirements.txt
```

**2. Environment Configuration**
Copy the `.env.example` file to `.env` and insert your API Key or local/cloud LLM URL configuration:
```bash
cp .env.example .env
```

**3. Main Pipeline Execution**
Run the following scripts sequentially based on your experimental needs:
*   **Stage 1:** Run text augmentation using LLMs.
    ```bash
    python main_augment.py
    ```
*   **Stage 2:** Train the XGBoost model and generate classification scores.
    ```bash
    python main_classifier.py
    ```
*   **Stage 3:** Run the intrinsic text quality evaluation.
    ```bash
    python main_evaluator.py
    ```

**4. Result Visualization**
Open `notebooks/notebooksV2.ipynb` to run code to run statistical test, visualize data, and exports as PNG