# Research Assistant Agent

A LangChain-based research assistant that answers research questions using external tools instead of relying only on the language model. The project compares two agent styles:

1. **ReAct agent**: reasons and calls tools step-by-step during execution.
2. **Chain-of-Thought planning agent**: creates a tool-use plan first, executes the planned tools, then synthesizes an answer.

The agent can search the web, query Wikipedia, fetch and read PDFs, retrieve exact quotes from a loaded document, summarize an active document, and generate conservative citations without inventing missing metadata.

---

## Features

- Web search for current information using the Serper API
- Wikipedia lookup for background information
- PDF fetching and text extraction with `PyPDF2`
- Quote retrieval from the default Transformer paper, *Attention Is All You Need*
- Semantic quote search using `intfloat/e5-small-v2`
- Quote reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Active document summarization
- Citation helper that avoids guessing missing authors, years, or venues
- ReAct vs. CoT agent comparison
- Evaluation metrics for tool selection, keyword coverage, number of steps, and runtime
- Ablation study showing the effect of removing the quote-search tool

---

## Project Structure

```text
main_agent.ipynb              # Main notebook with setup, tools, agents, evaluation, and demo
README.md                     # Project documentation
results/                      # Evaluation outputs and generated figures
test_cases/                   # Test files used for document upload / retrieval experiments
```

The main agent notebook is organized into these sections:

```text
1. Setup
2. Web Search Tool
3. Wikipedia Search Tool
4. PDF Fetch Tool
5. Citation Tool
6. Quote Retrieval Pipeline
7. Agent Implementation
8. Testing
9. Evaluation
10. Statistical Significance
11. Ablation Study
12. Demo
```

---

## Requirements

This project was built in Google Colab and uses Python 3.12. Install the required packages with:

```bash
pip install langchain langchain-openai langchain-community langchain-classic openai wikipedia-api requests beautifulsoup4 PyPDF2 python-dotenv sentence-transformers transformers torch "gradio>=5.0,<6.0"
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
PyPDF2
nltk
sentence-transformers
transformers
torch
pandas
numpy
matplotlib
scipy
gradio
```

---

## API Keys

The notebook expects two API keys:

```text
OPENAI_API_KEY   # used for ChatOpenAI
SERPER_API_KEY   # used for web search through google.serper.dev
HF_TOKEN         # used for loading model
```

In Google Colab, add both keys using the built-in Secrets manager. Then the notebook loads them with:

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = userdata.get("SERPER_API_KEY")
os.environ["HF_TOKEN"]       = userdata.get("HF_TOKEN")

```

---

## How to Run

### Option 1: Run in Google Colab

1. Open `main_agent.ipynb` in Google Colab.
2. Add `OPENAI_API_KEY` and `SERPER_API_KEY` in Colab Secrets.
3. Run the setup cell to install dependencies.
4. Run the import/API-key cell.
5. Run the tool-definition cells:
   - `web_search`
   - `wikipedia_search`
   - `fetch_pdf`
   - `generate_citation`
   - `quote_search`
   - `summarize_active_document`
6. Run the agent implementation cells for:
   - `run_react(...)`
   - `run_cot(...)`
7. Test the agent with:

```python
question = "What were the key contributions of the Attention Is All You Need paper?"
result = run_react(question)
print(result["output"])
print(result["tools_used"])
```

8. Run the evaluation section to compare ReAct and CoT.
9. Run the plotting cell to generate `react_vs_cot_comparison.png`.

### Option 2: Run Locally

1. Clone or download the project files.
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
pip install langchain langchain-openai langchain-community langchain-classic openai wikipedia-api requests beautifulsoup4 PyPDF2 python-dotenv sentence-transformers transformers torch "gradio>=5.0,<6.0"
```

4. Set your API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export SERPER_API_KEY="your_serper_key"
```

5. Open the notebook:

```bash
jupyter notebook main_agent.ipynb
```

6. Run the notebook cells from top to bottom.

---

## Tool Overview

### `web_search(query)`

Searches the web using the Serper API. This is best for current facts, recent events, statistics, and anything that may have changed recently.

### `wikipedia_search(query)`

Retrieves background context from Wikipedia. This is best for established concepts, historical topics, biographies, and general definitions.

### `fetch_pdf(url)`

Downloads a PDF from a URL and extracts text from the first several pages. It also converts arXiv abstract links into PDF links automatically.

### `generate_citation(...)`

Creates a conservative citation string. It does not guess missing metadata. Missing authors or years are marked as unknown, `n.d.`, or source-specific defaults.

### `quote_search(query)`

Searches the active document for exact verified quotes. By default, the active document is *Attention Is All You Need*. The pipeline:

1. Loads the source document.
2. Splits the document into sentences.
3. Filters noisy or low-quality sentences.
4. Embeds sentences with `intfloat/e5-small-v2`.
5. Retrieves semantic matches.
6. Reranks candidates with a cross-encoder.
7. Verifies that each returned quote exists exactly in the source text.

### `summarize_active_document(query)`

Summarizes or answers questions about the currently loaded document using the extracted document text.

---

## Agent Implementations

### ReAct Agent

The ReAct agent uses LangChain's tool-calling agent. It decides during execution which tool to use next based on the current question and previous observations.

Run it with:

```python
result = run_react("What is LoRA and why is it more efficient than full fine-tuning?")
print(result["output"])
print(result["tools_used"])
print(result["num_steps"])
```

### Chain-of-Thought Planning Agent

The CoT agent first asks the model to produce a JSON research plan. Then it executes the planned tool calls and synthesizes the final answer.

Run it with:

```python
result = run_cot("Explain the HNSW algorithm for approximate nearest neighbor search.")
print(result["output"])
print(result["tools_used"])
print(result["num_steps"])
```

---

## Evaluation

The notebook evaluates ReAct and CoT on ten research questions covering topics such as RAG, OpenAI, Transformer contributions, quote retrieval, LoRA, quantum computing, HNSW, Chinchilla scaling laws, and BERT vs. GPT.

The evaluation tracks:

- **Tool Selection F1**: whether the agent chose the expected tools
- **Keyword Coverage**: whether the answer contained expected topic keywords
- **Number of Steps**: how many tool calls the agent used
- **Runtime**: total time per question
- **Task Completion Rate**: whether the agent returned a non-empty useful answer

Example summary from the notebook:

```text
Metric                         ReAct        CoT
---------------------------------------------
Avg Tool Selection F1          0.967      0.933
Avg Keyword Coverage           0.933      0.850
Avg Steps                        2.7        1.6
Avg Time (seconds)              14.9       10.9
Task Completion Rate           1.000      1.000
```

Interpretation:

- ReAct achieved slightly better tool selection and keyword coverage.
- CoT used fewer steps and was faster on average.
- Both agents completed all tasks.
- The statistical significance tests did not show a significant difference at `p < 0.05` for keyword coverage or tool selection F1.

---

## Ablation Study

The notebook also tests what happens when `quote_search` is removed from the available tools.

Purpose:

```text
Compare quote-related performance with and without the specialized quote-search tool.
```

Finding:

```text
The agent can sometimes compensate by using web search and PDF fetching, but quote_search gives a more direct and reliable path for exact evidence retrieval from the active document.
```

---

## Example Questions

Try these prompts:

```text
What is RAG and how does it reduce hallucinations?
Who is the CEO of OpenAI and what was his previous role?
What were the main contributions of the Attention Is All You Need paper?
Find 3 quotes from the Transformer paper about why attention is better than recurrence.
What is LoRA and why is it more efficient than full fine-tuning?
Explain the HNSW algorithm for approximate nearest neighbor search.
What is the Chinchilla scaling law and how did it change LLM training?
What are the differences between BERT and GPT architectures?
```

---

## Notes and Limitations

- The web-search tool requires a valid Serper API key.
- The OpenAI model calls require a valid OpenAI API key.
- The quote-search system only searches the currently loaded document.
- The default quote document is *Attention Is All You Need*.
- PDF extraction may fail for scanned PDFs or PDFs with complex formatting.
- The citation tool is intentionally conservative and may leave metadata blank instead of guessing.
- Some package versions may create dependency warnings in Colab, especially around `requests`.
- The evaluation set is small, so the results should be treated as a project-level comparison, not a definitive benchmark.

---

## Future Improvements

- Add support for uploading and indexing multiple documents at once.
- Store embeddings in a vector database instead of rebuilding them in memory.
- Add source-grounded answer verification.
- Improve citation formatting with structured metadata extraction.
- Expand the evaluation set beyond ten questions.
- Add a stronger document loader for scanned PDFs using OCR.
- Add a Gradio interface for interactive use.
- Cache web and PDF results to reduce repeated calls and latency.

---

## Summary

This project demonstrates how a research assistant agent can combine LLM reasoning with external tools. The ReAct agent performs better on tool selection and answer coverage in the small evaluation, while the CoT planning agent is more efficient. The specialized quote-search pipeline is especially useful for exact evidence retrieval from the Transformer paper.

Note: HotpotQA was included in results/Run_all_tests.ipynb to avoid run time and memory issues
