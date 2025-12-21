import os

# TEMPLATE FOR A THIRD PROJECT STRUCTURE
# PROJECT_NAME = "3.Embedding Similarity Search"

# STRUCTURE = {
#     "data": [
#         "raw_texts.py",
#         "queries.py",
#     ],
#     "embeddings": [
#         "embedder.py",
#         "embedding_store.py",
#     ],
#     "similarity": [
#         "metrics.py",
#         "search.py",
#     ],
#     "evaluation": [
#         "manual_checks.md",
#     ],
#     "config": [
#         "settings.py",
#     ],
#     "": [
#         "main.py",
#         "requirements.txt",
#         "README.md",
#     ]
# }


# def create_structure(base_path: str):
#     for folder, files in STRUCTURE.items():
#         folder_path = os.path.join(base_path, folder)
#         if folder:
#             os.makedirs(folder_path, exist_ok=True)

#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             if not os.path.exists(file_path):
#                 with open(file_path, "w", encoding="utf-8") as f:
#                     pass


# if __name__ == "__main__":
#     base_dir = os.path.join(os.getcwd(), PROJECT_NAME)
#     os.makedirs(base_dir, exist_ok=True)
#     create_structure(base_dir)
#     print(f"Project structure created at: {base_dir}")
