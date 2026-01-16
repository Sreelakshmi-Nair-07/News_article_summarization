#  News Article Summarization using LLM

## Overview
This project involves text summarization using a pre-trained Large Language Model. The solution focuses on clarity of approach, use of modern LLM pipelines, and practical deployment via a web application.

## Dataset
- XSum (Extreme Summarization) dataset from Hugging Face
- Contains BBC news articles and reference summaries

## Model Used
- facebook/bart-large-cnn
- Chosen because it is optimized for news summarization and works well without fine-tuning

## Tech Stack
- Python
- Hugging Face Datasets
- Hugging Face Transformers
- PyTorch
- Streamlit

## Why This Approach
- Requires minimal processing since the transformer models handle raw text effectively
- Hugging Face pipelines handles tokenization and decoding complexity
- Focus is on inference and pipeline design rather than heavy training
- Pre-trained BART model optimized for news summarization.  
        Minimal preprocessing was applied since transformers handle raw text well.  Inference is abstracted using Hugging Face pipelines, which simplifies model usage.
- A Streamlit app demonstrates real-world usability.

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
