# Wiki QA Conciseness Synthetic Dataset

This project uses various large language models (LLMs) to synthetically generate questions based on sentences from the [Wiki Conciseness Dataset](https://github.com/google-research-datasets/wiki-conciseness-dataset).

The goal is to detect how various prompting techniques affect the conciseness of the LLM answer.

## Process Overview

1. **Question Generation**: For each sentence with a specific conciseness level, a corresponding question is generated using LLMs.
2. **Answer Generation**: For each question, we use various prompts to control the verbosity of the answer.
3. **Similarity Comparison**: The generated answers are compared to the expected concise sentences using similarity metrics.

## Similarity Metrics

- **BERTScore Similarity**: Measures the semantic similarity between sentences.
- **ROUGE Score**: Measures n-gram similarities.
- **Length Ratio**: Comparing lengths of two sentences.
- **Verbosity Score**: A weighted combination of the metrics above.

## Repo structure

```
wiki-conciseness-data/          # original Wiki Conciseness Dataset
│
wiki-qa-data/
│   ├── data/                   # Progress data for comparing different prompts' effect on LLM responses
│   ├── llm_factory.py          # Centralised script for LLM API Calls [Ollama, OpenAI, Google, ...]       
│   ├── questions_gen.py        # Script to generate Questions based on the Wiki Conciseness Dataset
│   ├── llm_consistency.py      # Script to generate answers to the synthetic questions & apply similarity metrics
│   ├── analyse_data.py         # Script to analyse metrics over all data-points and graph results
│
│   ├── literature.md           # Literature Referenced for Metrics
│   ├── README.md               # Project documentation
