from src.transformer.train_transformer import train_transformer_model
from src.transformer.evaluate_transformer import evaluate_transformer_model


def main():
    print("=== Training Transformer Model ===")
    train_transformer_model()

    print("\n=== Evaluating Transformer Model ===")
    evaluate_transformer_model()


if __name__ == "__main__":
    main()
