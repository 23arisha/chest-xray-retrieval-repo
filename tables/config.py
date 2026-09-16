"""Constants shared across the tables/ package: finding names, model keys,
and the statistical-procedure parameters used to reproduce manuscript
Tables 1-4 exactly (pooling depth, bootstrap/permutation counts, seed)."""

FINDINGS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

MODELS = ["ours", "repair", "redone"]
MODEL_DISPLAY = {"ours": "Our model", "repair": "CXR-RePaiR", "redone": "CXR-ReDonE"}

# Verified: pooling the top-5 retrieved sentences (positive-if-any-of-top-5)
# exactly reproduces the reported per-finding recall values used in Tables 2-4.
TOP_K_POOL = 5

N_BOOTSTRAP_IMAGE_LEVEL = 3000
RNG_SEED = 2024
N_PERMUTATIONS = 100000  # manuscript-reproducing precision; reduce for faster iteration
