from evaluate import load as load_metric

def compute_metrics(predictions, references):
    """
    Compute precision, recall, F1 using seqeval.
    predictions: list of list of predicted labels
    references: list of list of true labels
    Returns a dict with micro, macro, and overall metrics.
    """
    metric = load_metric("seqeval")
    results = metric.compute(predictions=predictions, references=references)
    return results


if __name__ == "__main__":
    # Example
    preds = [[0,1,0], [2,0,0]]
    labels = [[0,1,0], [2,0,0]]
    results = compute_metrics(preds, labels)
    print(results)
