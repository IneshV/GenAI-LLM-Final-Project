# Research Assistant Agent

A LangChain-based research assistant that answers research questions using external tools instead of relying only on the language model's internal knowledge.

The project compares two agent strategies:

1. **ReAct agent**: chooses tools step-by-step during execution after observing previous tool outputs.
2. **Chain-of-Thought planning agent**: creates a fixed tool-use plan first, executes the planned tools, and then synthesizes an answer.

The system can search the web, query Wikipedia, fetch and read PDFs, summarize an active document, retrieve exact verified quotes from a loaded document, and generate conservative citations without inventing missing metadata. The repository also includes automated tests, documentation for reproducing experiments, and evaluation notebooks comparing ReAct and CoT.

---

## Features

- Web search for current information using the Serper API
- Wikipedia lookup for stable background information
- PDF fetching and text extraction with `PyPDF2`
- OCR helper for scanned PDFs
- Active document summarization
- Verified quote retrieval from a loaded document
- Semantic quote search using `intfloat/e5-small-v2`
- Quote reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Conservative citation helper that avoids guessing missing metadata
- Academic quote-finding pipeline for claim-level evidence retrieval
- ReAct vs. CoT agent comparison
- Curated benchmark evaluation
- HotpotQA multi-hop QA evaluation
- Statistical significance tests
- Ablation study for the quote-search tool
- Unit tests, demo-input tests, and GitHub Actions workflow
- Generated evaluation figures and reproducibility documentation

---

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml                     # GitHub Actions workflow for automated testing
├── docs/
│   ├── experiment_setup.md               # Experimental setup and reproducibility notes
│   └── troubleshooting.md                # Common setup/runtime issues and fixes
├── notebooks/
│   ├── main_agent_demo_version.ipynb     # Main notebook with tools, agents, evaluation, and demo
│   └── Run_all_tests.ipynb               # Notebook for full evaluation workflows
├── results/
│   └── react_vs_cot_comparison.png       # Generated ReAct vs. CoT comparison figure
├── src/
│   └── extract_pdf_text_with_ocr.py      # OCR-based PDF text extraction helper
├── test_cases/                           # Test documents for upload, OCR, and retrieval experiments
│   ├── 400-motivational-quotes.pdf
│   ├── Untitled document_copy.pdf
│   ├── sample_upload_test_document.txt
│   └── scanned_pdf_test_document.pdf
├── tests/
│   ├── test_basic.py                     # Basic repository/import/path tests
│   └── test_demo_inputs.py               # Tests for demo/test-case inputs
├── .gitignore
├── README.md
├── requirements.txt                      # Core project dependencies
└── requirements-evaluation.txt           # Extra dependencies for evaluation experiments
```

---

## Repository Documentation

Additional documentation is included in the `docs/` folder:

- `docs/experiment_setup.md`: explains the experimental setup, benchmark organization, and how to reproduce results.
- `docs/troubleshooting.md`: lists common issues with API keys, package installation, PDF extraction, OCR, and notebook execution.

These files are intended to support reproducibility and make the project easier to grade, run, and debug.

---

## Requirements

This project was developed primarily in Google Colab with Python 3.12.

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

For evaluation workflows, also install:

```bash
pip install -r requirements-evaluation.txt
```

Main libraries used:

```text
langchain
langchain-openai
langchain-community
langchain-classic
openai
wikipedia-api
requests
beautifulsoup4
PyPDF2
nltk
sentence-transformers
transformers
torch
pandas
numpy
matplotlib
scipy
datasets
gradio
```

---

## API Keys

The notebook expects the following keys:

```text
OPENAI_API_KEY   # used for ChatOpenAI
SERPER_API_KEY   # used for web search through google.serper.dev
HF_TOKEN         # optional, used for Hugging Face model access if needed
```

In Google Colab, add these keys through the Secrets manager. The notebook loads them with:

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = userdata.get("SERPER_API_KEY")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

For local use, create a `.env` file:

```text
OPENAI_API_KEY=your_openai_key
SERPER_API_KEY=your_serper_key
HF_TOKEN=your_huggingface_token
```

Do not commit API keys to GitHub.

---

## How to Run

### Option 1: Google Colab

1. Open `notebooks/main_agent_demo_version.ipynb` in Google Colab.
2. Add `OPENAI_API_KEY` and `SERPER_API_KEY` in Colab Secrets.
3. Run the setup cells.
4. Run the API-key loading cell.
5. Run the tool-definition cells.
6. Run the agent implementation cells.
7. Test the agent with:

```python
question = "What were the key contributions of the Attention Is All You Need paper?"
result = run_react(question, verbose=False)

print(result["output"])
print(result["tools_used"])
print(result["num_steps"])
```

8. Run the evaluation sections to compare ReAct and CoT.
9. Run the plotting cells to generate figures in `results/`.

### Option 2: Local Notebook

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add API keys to a `.env` file.
5. Start Jupyter:

```bash
jupyter notebook
```

6. Open and run:

```text
notebooks/main_agent_demo_version.ipynb
```

---

## Tool Overview

### `web_search(query)`

Searches the web using the Serper API. This tool is best for current facts, recent events, statistics, and information that may have changed recently.

### `wikipedia_search(query)`

Retrieves background information from Wikipedia. This is best for stable concepts, historical topics, biographies, and general definitions.

### `fetch_pdf(url)`

Downloads a PDF from a URL and extracts text with `PyPDF2`. It also converts arXiv abstract links into PDF links automatically.

### `generate_citation(...)`

Creates a conservative citation string from available metadata. It does not invent missing authors, years, or venues.

### `quote_search(query)`

Searches the active document for exact verified quotes. By default, the active document is *Attention Is All You Need*, but the Gradio interface can swap in an uploaded PDF, TXT, or MD file.

The quote-search pipeline:

1. Loads the active document.
2. Splits the document into sentences.
3. Filters noisy or low-quality sentences.
4. Embeds sentences with `intfloat/e5-small-v2`.
5. Retrieves semantically similar candidates.
6. Reranks candidates with a cross-encoder.
7. Verifies that each returned quote appears exactly in the source text.

### `summarize_active_document(query)`

Summarizes or answers questions about the currently loaded document.

### `find_paper_quotes(query)`

Finds academic sources and verified supporting quotes for citation-ready writing. Unlike `quote_search`, which searches one active document, this pipeline searches the web, retrieves candidate papers or pages, parses them, extracts candidate passages, reranks them, and verifies exact quote text.

It is designed for prompts such as:

```text
XGBoost is a tree-based ensemble method that iteratively fits decision trees to minimize a specified loss function while controlling model complexity through regularization.

The model is well suited for soil organic carbon prediction due to its ability to capture nonlinear relationships, feature interactions, and heterogeneous environmental responses across spatial and depth gradients.

Find papers to reference and quotes to support this.
```

---

## Agent Implementations

### ReAct Agent

The ReAct agent uses LangChain's tool-calling agent abstraction. It chooses tools iteratively during execution based on the current question and previous tool outputs.

```python
result = run_react(
    "What is LoRA and why is it more efficient than full fine-tuning?",
    verbose=False
)

print(result["output"])
print(result["tools_used"])
print(result["num_steps"])
```

### Chain-of-Thought Planning Agent

The CoT agent first generates a JSON research plan, then executes that plan and synthesizes the final answer. It is less adaptive than ReAct but usually uses fewer tool calls.

```python
result = run_cot(
    "Explain the HNSW algorithm for approximate nearest neighbor search.",
    verbose=False
)

print(result["output"])
print(result["tools_used"])
print(result["num_steps"])
```

---

## Testing

This repository includes automated tests under `tests/` and a GitHub Actions workflow under `.github/workflows/tests.yml`.

### Run tests locally

From the repository root, run:

```bash
pip install -r requirements.txt
pip install -r requirements-evaluation.txt
pytest tests/ -v
```

### Test files

```text
tests/test_basic.py          # Checks basic repository structure and required files
tests/test_demo_inputs.py    # Checks demo/test-case files used by the notebook
```

### GitHub Actions

The workflow file:

```text
.github/workflows/tests.yml
```

runs the test suite automatically when code is pushed or a pull request is opened. This supports the automated testing bonus and helps catch missing files or broken setup before submission.

### Notes on testing

The tests are intentionally lightweight. They are designed to verify repository organization, required files, and demo inputs without requiring paid API calls. Full agent evaluation still requires valid OpenAI and Serper API keys and is run from the notebooks.

---

## Evaluation

The project includes several evaluations.

### 1. Curated ReAct vs. CoT Benchmark

The curated benchmark tests research-style questions involving web search, Wikipedia lookup, PDF/document use, and quote extraction.

Metrics include:

- **Tool Selection F1**
- **Keyword Coverage**
- **Answer Similarity**
- **Task Completion Rate**
- **Average Tool Steps**
- **Runtime**

Example prompts:

```text
What is RAG and how does it reduce hallucinations?
Who is the CEO of OpenAI and what was his previous role?
What were the main contributions of the Attention Is All You Need paper?
Find 3 quotes from the Transformer paper about why attention is better than recurrence.
What is LoRA and why is it more efficient than full fine-tuning?
Explain the HNSW algorithm for approximate nearest neighbor search.
```

### 2. Ablation Study

The ablation study removes `quote_search` from the tool registry and tests whether the agent can still answer quote-oriented questions using general tools such as `web_search` and `fetch_pdf`.

Main finding:

```text
General-purpose tools can sometimes recover relevant information, but they do not provide the same exact-substring quote verification and character-level provenance as quote_search.
```

### 3. Academic Quote Retrieval Benchmark

The `find_paper_quotes` pipeline is evaluated against a generic web-search baseline on citation-ready evidence retrieval.

Metrics include:

- Source found rate
- Quote found rate
- Verified quote found rate
- Complete claim quote coverage
- Complete verified claim coverage
- Quote density
- Verified quote density
- Source quality
- Citation readiness

Main finding:

```text
find_paper_quotes produced verified, claim-level supporting quotes, while the generic web baseline often returned plausible sources or snippets without exact quote verification.
```

### 4. HotpotQA Multi-Hop QA Evaluation

HotpotQA evaluation is included in `notebooks/Run_all_tests.ipynb` to avoid runtime and memory issues in the main demo notebook.

This evaluation compares ReAct and CoT on 100 HotpotQA validation questions:

```text
85 bridge questions
15 comparison questions
```

Metrics include:

- Exact Match
- Token F1
- Average Tool Steps
- Average Time
- Max-Iteration Failure Rate

Summary result:

```text
CoT slightly outperformed ReAct on answer accuracy, but the difference was not statistically significant.
ReAct required more tool calls, took longer, and had more max-iteration failures.
```

---

## Results Summary

Key findings:

```text
1. ReAct was more adaptive and better at exploring tools in the curated benchmark.
2. CoT was usually faster and more efficient because it used fewer tool calls.
3. On HotpotQA, CoT slightly outperformed ReAct in accuracy, but not significantly.
4. ReAct had a higher cost due to extra tool calls and max-iteration failures.
5. Specialized quote tools did not always improve surface-level keyword metrics, but they greatly improved evidence quality and provenance.
6. The find_paper_quotes pipeline was strongest for citation-ready evidence retrieval because it returned verified quotes rather than only plausible URLs.
```

---

## Generated Outputs

The `results/` folder contains generated evaluation outputs and figures, including:

```text
react_vs_cot_comparison.png
```

Additional HotpotQA figures may be generated from `Run_all_tests.ipynb`, such as:

```text
hotpotqa_n100_accuracy_failure.png
hotpotqa_n100_efficiency.png
```

---

## Troubleshooting

Common issues and fixes are documented in:

```text
docs/troubleshooting.md
```

Typical issues include:

- Missing `OPENAI_API_KEY` or `SERPER_API_KEY`
- Serper API request failures
- PDF extraction failures on scanned or image-based PDFs
- Hugging Face model download delays
- Colab dependency warnings
- Notebook runtime disconnections
- Slow evaluation due to repeated web/API calls

---

## Notes and Limitations

- Web search requires a valid Serper API key.
- OpenAI model calls require a valid OpenAI API key.
- The default quote-search document is *Attention Is All You Need*.
- `quote_search` searches one active document at a time.
- The Gradio UI can swap the active document, but the agent does not yet autonomously select and load papers.
- PDF extraction may fail on scanned PDFs or complex layouts.
- OCR support is included, but OCR quality depends on scan quality.
- The citation tool is intentionally conservative and may leave metadata blank rather than guessing.
- Some Colab dependency warnings may appear, especially around `requests`.
- HotpotQA results depend on web search rather than the original gold context paragraphs, so search result quality can affect performance.
- The project uses GPT-4o-mini only; results may differ with other backbone models.

---

## Future Improvements

- Add automatic paper retrieval and loading for quote search.
- Support indexing multiple uploaded documents at once.
- Store document embeddings in a vector database.
- Add stronger source-grounded answer verification.
- Improve citation formatting with structured metadata extraction.
- Expand evaluation beyond the current benchmark sizes.
- Add better stopping rules for ReAct to reduce over-searching.
- Cache web and PDF results to reduce repeated calls and latency.
- Evaluate additional backbone models such as Llama or Qwen.
- Add LLM-as-judge or embedding-based evaluation metrics.

---

## Summary

This project demonstrates how a research assistant agent can combine LLM reasoning with external tools for search, reading, citation, and quote verification. ReAct is more adaptive, while CoT is more efficient. The strongest result comes from the citation-focused pipeline: verified quote retrieval requires more than web search, because research writing depends on exact, inspectable, source-grounded evidence.
