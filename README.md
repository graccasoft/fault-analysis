# AI-Driven Fault Analysis and Operational Optimization in Power Systems

## Overview
This project investigates how Large Language Models (LLMs) can turn unstructured power system data (shift notes, alarm logs, maintenance reports) into insights. By fine-tuning a transformer-based LLM (e.g., FLAN-T5 or LLaMA-2) with domain data, the system enables:

- Fault classification
- Trend/frequency analysis
- Actionable recommendation generation
- Insight dashboards for operators

## Repository Structure
- src/fault_analysis: Core library (data schema, loaders, preprocessing, training, analysis)
- data: Raw, processed and sample datasets
- configs: Configurations for experiments
- scripts: CLI entrypoints
- app: Streamlit dashboard
- notebooks: Research notebooks

## Quickstart
1) Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Review sample config
```
cat configs/config.example.yaml
```

4) Train a LoRA-tuned model (instruction-style SFT)
```
python scripts/train_lora.py --config configs/config.example.yaml
```

5) Run the dashboard
```
streamlit run app/streamlit_app.py
```

## Data Format
JSONL lines. Two supported schemas:
- instruction: {"id", "timestamp", "source", "instruction", "input", "output", "tags", "metadata"}
- fault: {"id", "timestamp", "source", "text", "fault_type", "labels", "recommendations", "metadata"}

Use `src/fault_analysis/data/loader.py` to validate and load.

## Notes
- Start with smaller models for prototyping (e.g., google/flan-t5-base).
- GPU recommended for training. CPU will be slow.
- Adjust LoRA and training hyperparameters in the config.
