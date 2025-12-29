from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

def get_tokenizer(model_name: str = "xlm-roberta-base"):
    """
    Initialize and return the tokenizer for the base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_and_align_labels(dataset: DatasetDict, tokenizer, label_all_tokens: bool = True):
    """
    Tokenize the dataset and align NER labels with subword tokens.
    """
    def tokenize_example(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=512
        )
        labels = []
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels.append(-100)
            else:
                label = example["ner_tags"][word_idx]
                # If a word is split into multiple tokens
                if i != 0 and not label_all_tokens:
                    labels.append(-100)
                else:
                    labels.append(label)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = {}
    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(tokenize_example, remove_columns=["tokens", "ner_tags"])

    return DatasetDict(tokenized_dataset)

if __name__ == "__main__":
    from src.data.dataset_loader import load_hf_dataset, split_dataset

    # Load & split
    ds = load_hf_dataset()
    dataset = split_dataset(ds)

    tokenizer = get_tokenizer()

    # Tokenize & align labels
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer)
    print(tokenized_dataset)
