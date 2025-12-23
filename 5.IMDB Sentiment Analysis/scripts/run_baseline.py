from src.baseline.train_baseline import train_baseline_model
from src.baseline.evaluate_baseline import evaluate_baseline_model


def main():
    print("=== Training Baseline Model ===")
    train_baseline_model()

    print("\n=== Evaluating Baseline Model ===")
    evaluate_baseline_model()


if __name__ == "__main__":
    main()
