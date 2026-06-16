# 1. Go to the project and activate the environment
cd "/home/riccardo/Desktop/V-JEPA 2.1/vjepa_experiments"
source ../venv/bin/activate

# 2. Clear old outputs (run this if you changed images or want a fresh run)
find outputs/ -type f ! -name ".gitkeep" -delete && echo "Outputs cleared."

# 3. Run the full pipeline (embed → similarity → heatmaps → layer-wise)
python3 -c "
import os; os.environ['TORCH_HOME'] = '/media/riccardo/DATA/.torch_hub'
import numpy as np
from pathlib import Path

print('=== STEP 1: Embedding ===')
import embed_corpus
out_npz = embed_corpus.run()
embed_corpus.verify_npz(out_npz)

print()
print('=== STEP 2: Similarity matrix ===')
import similarity_matrix
similarity_matrix.run()

print()
print('=== STEP 3: Single-layer heatmaps ===')
import attention_heatmap
data = np.load(out_npz, allow_pickle=True)
attention_heatmap.visualize_corpus_attention(list(data['image_paths']))

print()
print('=== STEP 4: All-layers + aggregated heatmaps ===')
import layerwise_heatmap
layerwise_heatmap.run()

print()
print('=== DONE ===')
print('Outputs:', Path('outputs').resolve())
"
