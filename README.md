# Evals Test

Simple DeepEval "Answer Correctness" demo for our agent evaluation work.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U deepeval openai
export OPENAI_API_KEY="sk-..."
python deepeval-test.py
