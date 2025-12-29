from src.utils.helpers import set_seed
from src.utils.logging import get_logger
from src.inference.predict import load_model_and_tokenizer, predict_entities
from src.data.dataset_loader import get_label_mapping

def main():
    set_seed(42)
    logger = get_logger("Predict")

    # Load label mapping
    id2label, label2id = get_label_mapping()

    # Load trained model & tokenizer
    model_path = "experiments/run_001/checkpoints/best_model"
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Example text
    tokens = ["Prezidentin", "müvafiq", "sərəncamlarına", "uyğun", "Bibiheybət", "məscid-ziyarətgah", "də", "təmir", "işləri", "aparıldı", "."]
    
    # Predict entities
    entities = predict_entities(tokens, model, tokenizer, id2label)
    logger.info(f"Predicted entities: {entities}")

if __name__ == "__main__":
    main()
