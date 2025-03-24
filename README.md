# Wiki QA Conciseness Synthetic Dataset

This project uses various large language models (LLMs) to synthetically generate questions based on sentences from the [Wiki Conciseness Dataset](https://github.com/google-research-datasets/wiki-conciseness-dataset).

The goal is to detect how various prompting techniques affect the conciseness of the LLM answer.

## Process Overview

1. **Question Generation**: For each sentence with a specific conciseness level, a corresponding question is generated using LLMs.
2. **Answer Generation**: For each question, we use various prompts to control the verbosity of the answer.
3. **Similarity Comparison**: The generated answers are compared to the expected concise sentences using similarity metrics.

## Similarity Metrics

- **Cosine Similarity**: Measures the semantic similarity between sentences.
- **Verbosity Similarity**: (Metric to be determined).