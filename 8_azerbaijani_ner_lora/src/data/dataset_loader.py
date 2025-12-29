import os
from datasets import load_dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

def load_hf_dataset(dataset_name: str = "LocalDoc/azerbaijani-ner-dataset"):
    """
    Load the Azerbaijani NER dataset from HuggingFace.
    Returns a HuggingFace DatasetDict.
    """
    ds = load_dataset(dataset_name)
    return ds

def get_label_mapping():
    """
    Returns a mapping from integer to entity type and vice versa.
    """
    label_list = [
        "O", "PERSON", "LOCATION", "ORGANISATION", "DATE", "TIME", "MONEY",
        "PERCENTAGE", "FACILITY", "PRODUCT", "EVENT", "ART", "LAW", "LANGUAGE",
        "GPE", "NORP", "ORDINAL", "CARDINAL", "DISEASE", "CONTACT",
        "ADAGE", "QUANTITY", "MISCELLANEOUS", "POSITION", "PROJECT"
    ]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    return id2label, label2id

def split_dataset(ds, test_size=0.1, val_size=0.1, seed=42):
    """
    Split the HuggingFace dataset into train/validation/test sets.
    Returns a DatasetDict with pandas DataFrames.
    """
    # Convert to pandas temporarily for splitting
    df = ds["train"].to_pandas()
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=seed)

    # Reset indices
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Return as DatasetDict
    return DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })


def save_splits_to_csv(dataset: DatasetDict, save_dir: str = "data/raw/"):
    """
    Save train, validation, and test splits as CSV files.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset["train"].to_csv(os.path.join(save_dir, "az_ner_dataset_train.csv"), index=False)
    dataset["validation"].to_csv(os.path.join(save_dir, "az_ner_dataset_val.csv"), index=False)
    dataset["test"].to_csv(os.path.join(save_dir, "az_ner_dataset_test.csv"), index=False)
    print(f"Datasets saved to {save_dir}")

if __name__ == "__main__":
    # 1. Load dataset
    ds = load_hf_dataset()
    print("Dataset loaded from HuggingFace.")
    
    # 2. Label mapping
    id2label, label2id = get_label_mapping()
    print(f"Labels mapping: {id2label}")

    # 3. Split dataset
    dataset = split_dataset(ds)
    print(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['validation'])}, Test size: {len(dataset['test'])}")

    # 4. Save CSV files
    save_splits_to_csv(dataset)
