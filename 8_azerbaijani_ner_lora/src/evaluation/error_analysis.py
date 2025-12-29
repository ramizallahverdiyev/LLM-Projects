from collections import Counter

def analyze_errors(predictions, references, id2label):
    """
    Analyze False Positives, False Negatives, and rare entity performance.
    predictions: list of list of predicted labels
    references: list of list of true labels
    id2label: dict mapping id to entity string
    """
    fp_counter = Counter()
    fn_counter = Counter()

    for pred_seq, ref_seq in zip(predictions, references):
        for p, r in zip(pred_seq, ref_seq):
            if r != p:
                if p != 0 and r == 0:
                    fp_counter[id2label[p]] += 1
                elif r != 0 and p == 0:
                    fn_counter[id2label[r]] += 1

    return {"false_positives": dict(fp_counter), "false_negatives": dict(fn_counter)}

if __name__ == "__main__":
    id2label = {0:"O",1:"PERSON",2:"LOCATION"}
    preds = [[0,1,0],[2,0,0]]
    labels = [[0,1,0],[2,1,0]]
    errors = analyze_errors(preds, labels, id2label)
    print(errors)
