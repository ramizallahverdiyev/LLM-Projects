from datasets import load_dataset, DatasetDict
import ast

def load_hf_dataset(dataset_name: str):
    """
    Load the dataset from HuggingFace and split it.
    """
    dataset = load_dataset(dataset_name)
    
    # If the dataset is not already split, split it
    if "train" not in dataset or "validation" not in dataset or "test" not in dataset:
        train_testvalid = dataset["train"].train_test_split(test_size=0.2, seed=42)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
        dataset = DatasetDict({
            "train": train_testvalid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        })
        
    def convert_columns(example):
        try:
            if example["tokens"] is not None:
                example["tokens"] = ast.literal_eval(example["tokens"])
            else:
                example["tokens"] = []
        except (ValueError, SyntaxError):
            example["tokens"] = []

        try:
            if example["ner_tags"] is not None:
                example["ner_tags"] = [int(tag) for tag in ast.literal_eval(example["ner_tags"])]
            else:
                example["ner_tags"] = []
        except (ValueError, SyntaxError):
            example["ner_tags"] = []
            
        return example

    dataset = dataset.remove_columns("index")
    dataset = dataset.map(convert_columns)
    dataset = dataset.filter(lambda example: len(example['tokens']) == len(example['ner_tags']))
    return dataset

def get_label_mapping(label_list):
    """
    Returns a mapping from integer to entity type and vice versa.
    """
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    return id2label, label2id
