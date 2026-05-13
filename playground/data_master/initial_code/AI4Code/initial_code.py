import os
import json
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set paths
INPUT_PATH = "./input"
TRAIN_PATH = os.path.join(INPUT_PATH, "train")
TEST_PATH = os.path.join(INPUT_PATH, "test")
TRAIN_ORDERS = os.path.join(INPUT_PATH, "train_orders.csv")
TRAIN_ANCESTORS = os.path.join(INPUT_PATH, "train_ancestors.csv")
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_orders = pd.read_csv(TRAIN_ORDERS)
train_ancestors = pd.read_csv(TRAIN_ANCESTORS)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
BATCH_SIZE = 32
MAX_LENGTH = 128

# Load pre-trained embeddings model efficiently
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def get_cell_embeddings_batch(cell_texts):
    """Get embeddings for a batch of cell texts efficiently."""
    if not cell_texts:
        return np.array([])
    # SentenceTransformer handles batching internally
    embeddings = embedding_model.encode(
        cell_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings


def load_notebook_data(notebook_id, train=True):
    """Load notebook and return structured data."""
    path = os.path.join(TRAIN_PATH if train else TEST_PATH, f"{notebook_id}.json")
    with open(path, "r") as f:
        data = json.load(f)

    cells = []
    for cell_id, cell_type in data["cell_type"].items():
        source = data["source"][cell_id] if cell_id in data["source"] else ""
        cells.append(
            {"id": cell_id, "type": cell_type, "source": source, "cell_type": cell_type}
        )

    # Code cells are in original order in JSON, markdown cells are shuffled
    code_cells = [c for c in cells if c["type"] == "code"]
    markdown_cells = [c for c in cells if c["type"] == "markdown"]

    # Get embeddings for markdown cells only (code cells stay in place)
    if markdown_cells:
        markdown_texts = [c["source"] for c in markdown_cells]
        markdown_embeddings = get_cell_embeddings_batch(markdown_texts)
        for i, cell in enumerate(markdown_cells):
            cell["embedding"] = markdown_embeddings[i]

    return code_cells, markdown_cells, cells


def predict_cell_order(code_cells, markdown_cells):
    """Predict cell order using sliding window similarity."""
    if not markdown_cells:
        return [c["id"] for c in code_cells]

    # Code cells stay in original order from JSON
    code_ids = [c["id"] for c in code_cells]
    code_sources = [c["source"] for c in code_cells]

    # Get embeddings for code cells in batches
    code_embeddings = get_cell_embeddings_batch(code_sources)
    markdown_embeddings = np.array([c["embedding"] for c in markdown_cells])

    # Create sliding windows for code cells
    window_size = min(5, len(code_cells))
    positions = []

    for md_idx, md_emb in enumerate(markdown_embeddings):
        best_score = -1
        best_pos = 0

        # Try inserting at beginning
        if len(code_embeddings) > 0:
            sim = cosine_similarity([md_emb], code_embeddings[0:window_size])[0]
            avg_sim = np.mean(sim) if len(sim) > 0 else 0
            if avg_sim > best_score:
                best_score = avg_sim
                best_pos = 0

        # Try positions between code cells
        for i in range(len(code_embeddings)):
            start = max(0, i - window_size // 2)
            end = min(len(code_embeddings), i + window_size // 2 + 1)
            if start < end:
                sim = cosine_similarity([md_emb], code_embeddings[start:end])[0]
                avg_sim = np.mean(sim) if len(sim) > 0 else 0
                if avg_sim > best_score:
                    best_score = avg_sim
                    best_pos = i + 1

        # Try inserting at end
        if len(code_embeddings) > 0:
            start = max(0, len(code_embeddings) - window_size)
            sim = cosine_similarity([md_emb], code_embeddings[start:])[0]
            avg_sim = np.mean(sim) if len(sim) > 0 else 0
            if avg_sim > best_score:
                best_score = avg_sim
                best_pos = len(code_embeddings)

        positions.append((md_idx, best_pos, best_score))

    # Sort markdown cells by their predicted positions
    # For cells with same position, sort by score descending
    positions.sort(key=lambda x: (x[1], -x[2]))

    # Build final order
    final_order = []
    code_ptr = 0
    markdown_ptr = 0

    while code_ptr < len(code_cells) or markdown_ptr < len(positions):
        # Insert markdown cells that belong at current position
        while markdown_ptr < len(positions) and positions[markdown_ptr][1] == code_ptr:
            md_idx = positions[markdown_ptr][0]
            final_order.append(markdown_cells[md_idx]["id"])
            markdown_ptr += 1

        # Insert code cell
        if code_ptr < len(code_cells):
            final_order.append(code_cells[code_ptr]["id"])
            code_ptr += 1

    # Add any remaining markdown cells at the end
    while markdown_ptr < len(positions):
        md_idx = positions[markdown_ptr][0]
        final_order.append(markdown_cells[md_idx]["id"])
        markdown_ptr += 1

    return final_order


def kendall_tau(order_true, order_pred):
    """Compute Kendall tau correlation between two orders."""
    n = len(order_true)
    if n <= 1:
        return 1.0

    pos_true = {cell_id: i for i, cell_id in enumerate(order_true)}
    pos_pred = {cell_id: i for i, cell_id in enumerate(order_pred)}

    # Check that both orders contain the same cells
    if set(pos_true.keys()) != set(pos_pred.keys()):
        common_cells = set(pos_true.keys()) & set(pos_pred.keys())
        if len(common_cells) <= 1:
            return 0.0
        order_true = [c for c in order_true if c in common_cells]
        order_pred = [c for c in order_pred if c in common_cells]
        pos_true = {cell_id: i for i, cell_id in enumerate(order_true)}
        pos_pred = {cell_id: i for i, cell_id in enumerate(order_pred)}
        n = len(order_true)

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            cell_i = order_true[i]
            cell_j = order_true[j]
            if (pos_pred[cell_i] < pos_pred[cell_j]) == (i < j):
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    tau = (concordant - discordant) / total_pairs
    return tau


# Create validation split using ancestors
print("Creating validation split...")
ancestor_groups = train_ancestors["ancestor_id"].values
unique_ancestors = np.unique(ancestor_groups)
np.random.seed(42)
val_ancestors = np.random.choice(
    unique_ancestors, size=int(0.2 * len(unique_ancestors)), replace=False
)
val_mask = train_ancestors["ancestor_id"].isin(val_ancestors)
val_ids = train_ancestors[val_mask]["id"].tolist()[:1000]  # Limit to 1000 for speed
train_ids = train_ancestors[~val_mask]["id"].tolist()[:5000]  # Limit to 5000 for speed

print(f"Validation notebooks: {len(val_ids)}")

# Validation
print("Evaluating on validation set...")
val_scores = []
for notebook_id in tqdm(val_ids[:500]):  # Limit to 500 for speed
    try:
        code_cells, markdown_cells, _ = load_notebook_data(notebook_id, train=True)

        true_order_str = train_orders[train_orders["id"] == notebook_id][
            "cell_order"
        ].values[0]
        true_order = true_order_str.split()

        predicted_order = predict_cell_order(code_cells, markdown_cells)

        score = kendall_tau(true_order, predicted_order)
        val_scores.append(score)
    except Exception as e:
        # Fallback: code cells first, then markdown cells
        code_ids = [c["id"] for c in code_cells]
        markdown_ids = [c["id"] for c in markdown_cells]
        predicted_order = code_ids + markdown_ids
        if true_order:
            score = kendall_tau(true_order, predicted_order)
        else:
            score = 0.0
        val_scores.append(score)

avg_val_score = np.mean(val_scores) if val_scores else 0.0
print(f"Validation Kendall Tau: {avg_val_score:.4f}")

# Generate test predictions
print("Generating test predictions...")
test_files = glob.glob(os.path.join(TEST_PATH, "*.json"))
test_ids = [os.path.basename(f).replace(".json", "") for f in test_files]

submission_data = []
for notebook_id in tqdm(test_ids):
    try:
        code_cells, markdown_cells, _ = load_notebook_data(notebook_id, train=False)
        predicted_order = predict_cell_order(code_cells, markdown_cells)

        # Ensure all cells are included
        if len(predicted_order) != (len(code_cells) + len(markdown_cells)):
            # Fallback: code cells first, then markdown cells
            code_ids = [c["id"] for c in code_cells]
            markdown_ids = [c["id"] for c in markdown_cells]
            predicted_order = code_ids + markdown_ids

        submission_data.append(
            {"id": notebook_id, "cell_order": " ".join(predicted_order)}
        )
    except Exception as e:
        # Fallback in case of error
        code_ids = [c["id"] for c in code_cells] if "code_cells" in locals() else []
        markdown_ids = (
            [c["id"] for c in markdown_cells] if "markdown_cells" in locals() else []
        )
        predicted_order = code_ids + markdown_ids
        submission_data.append(
            {"id": notebook_id, "cell_order": " ".join(predicted_order)}
        )

# Create submission file
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Generated predictions for {len(submission_df)} notebooks")
print(f"Validation Kendall Tau: {avg_val_score:.4f}")

# Also save a copy in working directory for backup
submission_df.to_csv("./working/submission.csv", index=False)
