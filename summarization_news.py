from datasets import load_dataset
from transformers import pipeline

#Loading the dataset
def load_xsum_data(sample_size=5):
    dataset=load_dataset("xsum", split=f"train[:{sample_size}]")
    return dataset
#Summarization Pipeline
def load_summarizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

#preprocessing the text
""" Here the preprocessing done is minimal.
strip the whitespaces and truncate long articles"""
def preprocess_text(text, max_length=1024):
    text=text.strip()
    return text[:max_length]


# inference function
'''generating summmaries using  LLM'''
def summarize_article(summarizer, article):
    summary=summarizer(
        article,
        max_length=150,
        min_length=30,
        do_sample=False
    )
    return summary[0]["summary_text"]

#Main function

def main():
    print("loading the dataset")
    dataset=load_xsum_data()
    summarizer=load_summarizer()
    
    for i, sample in enumerate(dataset):
        print(f"\n---------ARTICLE {i+1}-------")
        article = preprocess_text(sample["document"])
        reference_summary = sample["summary"]

        generated_summary = summarize_article(summarizer, article)

        print("\nOriginal Article:\n", article)
        print("\nReference Summary:\n", reference_summary)
        print("\nGenerated Summary:\n", generated_summary)

if __name__ == "__main__":
    main()