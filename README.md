# Coding Agent Prompt — V-JEPA 2 / 2.1 Art History Corpus Experiments
## Phase 1: Embedding Extraction, Similarity Matrix, UMAP/t-SNE, Attention Heatmaps

---

## CONTEXT

You are building a modular Python experiment pipeline for analyzing an art history image corpus using Meta's V-JEPA 2 and V-JEPA 2.1 video world models. The corpus contains images of three categories: (A) sculptures and paintings depicting figures in tension or implied motion (e.g. Rubens), (B) artifacts containing mirrors or reflections, (C) picture-in-picture compositions (images of images, frames within frames).

The goal of Phase 1 is to build four foundational, composable modules — embedding extraction, pairwise similarity matrix, UMAP/t-SNE visualization, and attention heatmaps — that will serve as reusable primitives for all future experiments. Do NOT build anything beyond what is specified here. A proxy corpus (SSv2) comparison comes in Phase 2.

The pipeline must support **both V-JEPA 2 and V-JEPA 2.1** via a single config flag, with no code changes required to switch between them.

---

## HARDWARE

- 2× NVIDIA RTX 4500 Ada Generation (24 GB VRAM each, no NVLink)
- Intel Xeon w7-3445 (40 threads)
- Default device: `cuda:0` for all Phase 1 work (single card is sufficient)

---

## REPOSITORY & MODEL REFERENCES

**Main repo:** https://github.com/facebookresearch/vjepa2
Clone it locally — do not install as a package. All imports must reference the cloned repo.

**Demo notebook for reference:** https://github.com/facebookresearch/vjepa2/blob/main/notebooks/vjepa2_demo.ipynb

**Model loading — use `torch.hub.load` for both versions:**

```python
import torch

# === V-JEPA 2 models (choose one) ===
# ViT-g/16 at 256px — lighter, faster
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
# ViT-g/16 at 384px — higher resolution (default for experiments)
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant_384')

# === V-JEPA 2.1 models (choose one) ===
# ViT-g at 384px — recommended default for 2.1
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_giant_384')
# ViT-G (gigantic) at 384px — largest, most VRAM
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')
# ViT-L at 384px — lighter option
model = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')

# === Preprocessor (shared across all variants) ===
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
```

**Important notes on V-JEPA 2.1 vs 2.0:**
- V-JEPA 2.1 uses a **modality-specific tokenizer**: 2D convolutions for images, 3D for video. This means single images are handled natively and correctly without the "1-frame video" workaround.
- V-JEPA 2.1 uses **deep self-supervision** (loss applied at 4 intermediate encoder layers). When extracting patch features for attention maps, extract from the **final encoder layer** by default, but make the layer index configurable.
- V-JEPA 2.1 does NOT yet have an action-conditioned (AC) variant — that is for Phase 3 and uses V-JEPA 2 only.
- Both 2.0 and 2.1 share the same `torch.hub` entry point and the same repo.

**V-JEPA 2 direct checkpoint download (alternative to torch.hub):**
```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt -P ./checkpoints/
```

**HuggingFace (V-JEPA 2 only, not 2.1):**
```python
# Only V-JEPA 2 is on HuggingFace — do NOT use for 2.1
from transformers import AutoModel, AutoVideoProcessor
model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-384")
```
Use `torch.hub` as the primary loading method for both versions to keep the code consistent.

---

## PROJECT STRUCTURE

Create the following directory layout:

```
vjepa_experiments/
├── config.py                  # Central config — ALL paths and model choice live here
├── model_loader.py            # Module 0: model loading, shared across all experiments
├── embed_image.py             # Module 1: single image → embedding
├── embed_corpus.py            # Module 2: batch embed entire corpus
├── similarity_matrix.py       # Module 3: pairwise cosine similarity within corpus
├── umap_viz.py                # Module 4: UMAP and t-SNE visualization
├── attention_heatmap.py       # Module 5: attention map extraction and overlay
├── run_phase1.py              # Orchestrator: runs all modules in sequence
├── outputs/                   # All generated artifacts land here
│   ├── embeddings/
│   ├── similarity/
│   ├── visualizations/
│   └── heatmaps/
└── requirements.txt
```

---

## MODULE 0 — `config.py`

This is the SINGLE place where the model version is configured. All other modules import from here and never hardcode model names or paths.

```python
# config.py

# ============================================================
# MODEL SELECTION — change only this to switch between 2.0 and 2.1
# ============================================================
MODEL_VERSION = "2.1"  # Options: "2.0" or "2.1"

# Model variant within the chosen version
# For 2.0: "vjepa2_vit_giant_384" | "vjepa2_vit_giant"
# For 2.1: "vjepa2_1_vit_giant_384" | "vjepa2_1_vit_gigantic_384" | "vjepa2_1_vit_large_384"
MODEL_VARIANT = "vjepa2_1_vit_giant_384"

# ============================================================
# PATHS — update these to match your filesystem
# ============================================================
CORPUS_DIR = "./corpus"              # Directory of art history images (jpg/png)
OUTPUT_DIR = "./outputs"             # All artifacts written here
CHECKPOINT_DIR = "./checkpoints"     # Optional: for manually downloaded .pt files

# ============================================================
# HARDWARE
# ============================================================
DEVICE = "cuda:0"
BATCH_SIZE = 16                      # Reduce to 8 if OOM on ViT-G at 384px

# ============================================================
# EMBEDDING SETTINGS
# ============================================================
IMAGE_SIZE = 384                     # Must match model variant (384 for all *_384 variants)
NUM_FRAMES = 1                       # Phase 1: images only. Phase 2+: set to 8 for video
ENCODER_LAYER = -1                   # Which encoder layer to extract from (-1 = final layer)
                                     # For 2.1, try intermediate layers: -4, -8 for richer spatial features

# ============================================================
# SIMILARITY MATRIX SETTINGS
# ============================================================
SIMILARITY_METRIC = "cosine"         # Only cosine supported in Phase 1

# ============================================================
# UMAP / t-SNE SETTINGS
# ============================================================
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
TSNE_PERPLEXITY = 30
PROJECTION_METHOD = "umap"           # Options: "umap", "tsne", "both"

# ============================================================
# ATTENTION HEATMAP SETTINGS
# ============================================================
ATTENTION_HEAD = "mean"              # "mean" averages across all heads; or set int (0-15) for specific head
ATTENTION_LAYER = -1                 # Which layer's attention to extract (-1 = final layer)
HEATMAP_ALPHA = 0.6                  # Overlay transparency (0=invisible, 1=opaque heatmap)
```

---

## MODULE 1 — `model_loader.py`

Loads the encoder once and provides it as a singleton. All other modules call `get_encoder()` — they never load the model themselves.

**Requirements:**
- Load using `torch.hub.load('facebookresearch/vjepa2', config.MODEL_VARIANT)`
- Set model to `eval()` mode, move to `config.DEVICE`, freeze all parameters
- Register forward hooks to capture intermediate patch token representations AND attention weights. Hooks must be registered at load time, not at inference time.
- Expose three functions:
  - `get_encoder()` → the model (nn.Module)
  - `get_patch_features(x)` → patch token tensor from `config.ENCODER_LAYER`
  - `get_attention_weights(x)` → attention weight tensor from `config.ATTENTION_LAYER`
- The hook mechanism must work for both V-JEPA 2 and V-JEPA 2.1 transformer blocks. Inspect the model architecture at load time and print the block names so the user can verify hook attachment points.
- Print on load: model version, variant name, parameter count, embedding dimension, device.

**Checkpoint:** After running `model_loader.py` standalone, it should print something like:
```
Loaded: vjepa2_1_vit_giant_384 (V-JEPA 2.1)
Parameters: 1.1B
Embedding dim: 1536
Device: cuda:0
Hooks registered at layers: [encoder.blocks.45, encoder.blocks.47]
```

---

## MODULE 2 — `embed_image.py`

Atomic primitive: single image path → 1D numpy embedding vector.

**Requirements:**
- Function signature: `embed(image_path: str | Path) -> np.ndarray`
- Load image with PIL, convert to RGB (handle RGBA, grayscale, palette modes)
- Resize to `(config.IMAGE_SIZE, config.IMAGE_SIZE)` with bicubic interpolation
- Apply standard ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- For V-JEPA 2: wrap as a 1-frame video tensor of shape `(1, C, 1, H, W)` (time dimension = 1)
- For V-JEPA 2.1: pass as a standard image tensor of shape `(1, C, H, W)` — the modality-specific tokenizer handles it natively. Detect via `config.MODEL_VERSION`.
- Extract the `[CLS]` token from the patch feature output → shape `(embedding_dim,)`. L2-normalize the output vector.
- Run with `torch.no_grad()` and `torch.cuda.amp.autocast()`
- Return as `np.ndarray` of dtype `float32`

**Also expose:** `embed_batch(image_paths: list) -> np.ndarray` that processes a list of paths in a single batched forward pass (shape `(N, embedding_dim)`). This is what `embed_corpus.py` calls — never call `embed()` in a loop.

**Checkpoint:** Running `python embed_image.py path/to/image.jpg` should print the vector shape and L2 norm (should be 1.0 after normalization). Embedding the same image twice must produce bit-identical results.

---

## MODULE 3 — `embed_corpus.py`

Batch-embeds the entire corpus directory and saves to disk.

**Requirements:**
- Walk `config.CORPUS_DIR` recursively, collect all `.jpg`, `.jpeg`, `.png`, `.webp` files
- Process in batches of `config.BATCH_SIZE` using `embed_batch()` from Module 2
- Show a tqdm progress bar
- Implement checkpoint/resume: save a partial `.npz` every 100 images. On restart, skip already-embedded images by checking which paths are already in the checkpoint file.
- Save final output to `{config.OUTPUT_DIR}/embeddings/corpus_embeddings.npz` containing:
  - `embeddings`: float32 array of shape `(N, embedding_dim)`
  - `image_paths`: array of N strings (absolute paths)
  - `model_version`: string — which model produced these embeddings (from config)
  - `image_size`: int — what resolution was used
- After saving, print a summary: N images embedded, embedding shape, any images that failed (log failed paths to `failed_images.txt`).

**Checkpoint:** Open the `.npz` in a Python REPL and verify:
```python
data = np.load("corpus_embeddings.npz", allow_pickle=True)
assert data['embeddings'].shape[0] == len(data['image_paths'])
assert not np.any(np.isnan(data['embeddings']))
# Verify L2 norms are all ~1.0
norms = np.linalg.norm(data['embeddings'], axis=1)
assert np.allclose(norms, 1.0, atol=1e-5)
```

---

## MODULE 4 — `similarity_matrix.py`

Computes the pairwise cosine similarity matrix within the corpus and produces a clustered heatmap.

**Requirements:**

**Computation:**
- Load `corpus_embeddings.npz`
- Verify embeddings are L2-normalized (re-normalize if not)
- Compute the `(N, N)` similarity matrix as `embeddings @ embeddings.T` — this is exact cosine similarity for unit-norm vectors. Use float32. For large N (>2000), compute in blocks to avoid OOM.
- Save matrix to `{config.OUTPUT_DIR}/similarity/corpus_similarity_matrix.npz`

**Visualization:**
- Apply hierarchical clustering (scipy `linkage` with `ward` method) to reorder rows and columns so similar images are adjacent
- Produce two heatmap visualizations using seaborn/matplotlib:
  1. `corpus_similarity_heatmap_clustered.png` — full matrix, clustered ordering, colormap `coolwarm`, diagonal forced to 1.0. Include a colorbar. Tick labels are image filenames (truncated to 20 chars if long). Only render tick labels if N < 100; omit them for larger corpora.
  2. `corpus_similarity_heatmap_raw.png` — same matrix but in original directory order (no clustering). Useful for spotting patterns by category if images are organized by folder.
- Also save a `cluster_order.npy` array of reordered indices, so other modules can use the same ordering.

**Analysis output:** Print to stdout:
- Top 10 most similar pairs (excluding diagonal), with filenames and similarity scores
- Top 10 most dissimilar pairs
- Mean and std of off-diagonal similarities
- Any pairs with similarity > 0.99 (near-duplicates — flag these)

**Checkpoint:** Diagonal of the similarity matrix must be exactly 1.0. Off-diagonal values must be in [-1, 1]. Near-duplicate detection should catch any image that appears twice in the corpus.

---

## MODULE 5 — `umap_viz.py`

Projects corpus embeddings to 2D using UMAP and/or t-SNE and produces interactive and static visualizations.

**Requirements:**

**Dimensionality reduction:**
- Load `corpus_embeddings.npz`
- If `config.PROJECTION_METHOD == "umap"` or `"both"`: fit UMAP with `n_neighbors=config.UMAP_N_NEIGHBORS`, `min_dist=config.UMAP_MIN_DIST`, `metric=config.UMAP_METRIC`, `random_state=42`
- If `config.PROJECTION_METHOD == "tsne"` or `"both"`: fit t-SNE with `perplexity=config.TSNE_PERPLEXITY`, `metric="cosine"`, `random_state=42`, `n_iter=1000`
- Save 2D coordinates to `corpus_umap_coords.npz` and/or `corpus_tsne_coords.npz`

**Static visualization (matplotlib):**
- Scatter plot with each point colored by subdirectory/category (infer category from the parent folder name of each image path)
- If no subdirectory structure exists, color all points the same
- Save as `{config.OUTPUT_DIR}/visualizations/corpus_umap_static.png` (and t-SNE equivalent)
- Each point labeled with the filename on hover — use `mplcursors` if available, otherwise skip labels

**Interactive visualization (plotly — primary output):**
- Each point in the scatter is a circle, colored by category
- On hover: show image thumbnail (encode as base64 in the hover template), filename, and category
- Legend shows category names with point counts
- Save as `{config.OUTPUT_DIR}/visualizations/corpus_umap_interactive.html` — self-contained HTML, no external dependencies
- The HTML file must be openable in a browser without a server (use `include_plotlyjs='cdn'` for plotly, or `include_plotlyjs=True` for fully offline)

**Checkpoint:** Open the interactive HTML in a browser. Hovering over a point should show the image. Points from the same category (subdirectory) should visually cluster — if your categories are art-historically meaningful, some cluster structure should be visible even without labels. If the plot is uniformly random, check that embeddings loaded correctly and that UMAP was fit on the correct array.

---

## MODULE 6 — `attention_heatmap.py`

Extracts attention weights from the encoder and overlays them as heatmaps on the original images.

**IMPORTANT architectural note:** V-JEPA 2 and V-JEPA 2.1 are Vision Transformers (ViT). Their self-attention maps are available from the transformer blocks. The `[CLS]` token's attention to all patch tokens (across all heads) is the most semantically meaningful: it shows which spatial regions of the image the model considers most relevant. This is the primary extraction target.

**Requirements:**

**Hook registration (in `model_loader.py`, called at load time):**
- Register a forward hook on the attention module of `config.ATTENTION_LAYER`
- The hook should capture the raw attention weight tensor before softmax dropout, shape `(batch, num_heads, seq_len, seq_len)`
- Store in a module-level dict `_attention_cache = {}` keyed by layer index

**Extraction function (`get_attention_map(image_path) -> np.ndarray`):**
- Run a forward pass through the encoder for a single image
- Retrieve cached attention weights from `_attention_cache`
- Extract the `[CLS]`-to-patch slice: `attn[:, :, 0, 1:]` → shape `(num_heads, num_patches)`
- Handle `config.ATTENTION_HEAD`:
  - If `"mean"`: average across heads → shape `(num_patches,)`
  - If int: select that head → shape `(num_patches,)`
- Reshape from flat patch sequence to 2D spatial grid. For a 384px image with patch size 16: grid is `(24, 24)`. Compute grid size as `image_size // patch_size`.
- Upsample the `(grid_H, grid_W)` attention map to `(image_H, image_W)` using bilinear interpolation
- Normalize to `[0, 1]` range
- Return as `np.ndarray` of shape `(H, W)`, dtype float32

**Visualization function (`visualize_attention(image_path, output_path, colormap='jet')):`**
- Load original image as numpy array
- Get attention map via `get_attention_map()`
- Create overlay: `overlay = (1 - alpha) * image + alpha * colormap(attention_map)`
- Produce a side-by-side figure: [original image | attention heatmap only | overlay]
- Add title: filename, model version, attention layer, head setting
- Save to `output_path` as PNG at 150 DPI

**Batch function (`visualize_corpus_attention(top_n=None)):`**
- Process all images in the corpus (or `top_n` if specified)
- Save individual heatmaps to `{config.OUTPUT_DIR}/heatmaps/{image_stem}_attn.png`
- Also produce a summary grid: all heatmap overlays tiled into a single large PNG (`corpus_attention_grid.png`). Limit grid to max 100 images — if corpus is larger, sample randomly.

**Checkpoint:** For a figure in motion (e.g. a Rubens painting of a throwing figure), attention should concentrate on the figure, not the background. For a mirror painting, attention should concentrate on the mirror and reflected figure region. If the heatmap is uniformly bright or uniformly dark, the hook is likely not capturing the correct attention tensor — verify hook attachment point by printing the captured tensor's shape.

---

## MODULE 7 — `run_phase1.py`

Orchestrator that runs all modules in sequence with stop-and-report checkpoints.

```
STOP POINT 1: After model_loader.py
  → Print model summary. Confirm with user before proceeding.

STOP POINT 2: After embed_corpus.py
  → Print embedding summary (N images, shape, failed count).
  → Confirm with user before proceeding.

STOP POINT 3: After similarity_matrix.py
  → Print top-10 most similar pairs. Show the heatmap PNG path.
  → Confirm with user before proceeding.

STOP POINT 4: After umap_viz.py
  → Print paths to static PNG and interactive HTML.
  → Confirm with user before proceeding.

STOP POINT 5: After attention_heatmap.py (sample run on 5 images only)
  → Show the 5 heatmap PNG paths for manual review.
  → Confirm with user before running full corpus batch.
```

Implement stop points as `input("Press Enter to continue, or Ctrl+C to stop: ")` calls.

---

## DEPENDENCIES — `requirements.txt`

```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.24.0
pillow>=10.0.0
tqdm>=4.66.0
scipy>=1.11.0
scikit-learn>=1.3.0
umap-learn>=0.5.5
plotly>=5.18.0
seaborn>=0.13.0
matplotlib>=3.8.0
mplcursors>=0.5.2
einops>=0.7.0
timm>=0.9.0
```

Install with: `pip install -r requirements.txt`

The vjepa2 repo itself is NOT installed as a package — it is cloned and used via `torch.hub` or by adding its path to `sys.path` if needed for internal utilities.

---

## CODING CONVENTIONS

- Every module must be runnable standalone (`if __name__ == "__main__"`) with a hardcoded example for testing
- All file I/O must use `pathlib.Path`, not raw strings
- All errors must be caught and logged to `outputs/errors.log` — never silently swallow exceptions
- All saved artifacts include a metadata header: `model_version`, `model_variant`, `image_size`, `timestamp`, `corpus_dir`
- No hardcoded paths anywhere except `config.py`
- Use `torch.no_grad()` and `model.eval()` consistently — never allow gradient computation during inference
- For any operation that might OOM: wrap in try/except, halve the batch size, retry once, then raise with a clear message

---

## WHAT NOT TO BUILD IN PHASE 1

Do not implement any of the following — they are for later phases:
- SSv2 proxy corpus embedding or retrieval
- Action classification probe (SSv2 or custom)
- Cross-model comparison (CLIP, DINOv2)
- Contrast pair analysis
- V-JEPA 2-AC action-conditioned rollouts
- Video embedding (all Phase 1 inputs are static images)
- Any fine-tuning or probe training

---

## DELIVERABLES

After completing all modules, the agent should confirm the following artifacts exist:

```
outputs/
├── embeddings/
│   └── corpus_embeddings.npz       ← (N, D) embeddings + metadata
├── similarity/
│   ├── corpus_similarity_matrix.npz
│   ├── corpus_similarity_heatmap_clustered.png
│   ├── corpus_similarity_heatmap_raw.png
│   └── cluster_order.npy
├── visualizations/
│   ├── corpus_umap_static.png
│   ├── corpus_umap_interactive.html
│   └── (corpus_tsne_static.png if PROJECTION_METHOD == "both")
└── heatmaps/
    ├── {image_stem}_attn.png        ← one per corpus image
    └── corpus_attention_grid.png    ← summary grid
```

Report the full path to each artifact and its file size. Stop and await instructions.
