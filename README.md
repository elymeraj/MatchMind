# MatchMind

MatchMind is an AI-powered football analysis project combining football analytics, Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and agentic AI.

The project initially focuses on the 2022 FIFA World Cup and progressively explores football event analysis, semantic search, vector databases, RAG, tool calling, and AI agents.

## Status

Project under development.

## Development environment

MatchMind currently targets CPython 3.12 on Intel macOS. PyTorch 2.2.2 is
intentionally pinned because it is the final release with an official macOS
x86_64 wheel. NumPy stays on the compatible 1.x ABI.

Create the project environment from the repository root:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv matchmind-env
source matchmind-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --prefix "$VIRTUAL_ENV" \
  --name matchmind --display-name "Python (MatchMind)"
```

Start Jupyter with the same environment:

```bash
source matchmind-env/bin/activate
jupyter lab
```

Both notebooks use the `Python (MatchMind)` kernel. The embedding notebook
downloads `sentence-transformers/all-MiniLM-L6-v2` on its first run.
