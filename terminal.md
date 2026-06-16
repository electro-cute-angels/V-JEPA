Step 1 — activate the environment (run once per terminal session):
cd "/home/riccardo/Desktop/V-JEPA 2.1/vjepa_experiments"
source ../venv/bin/activate

Step 2 — clear old outputs so everything runs fresh:
rm -f outputs/embeddings/corpus_embeddings.npz \
       outputs/similarity/corpus_similarity_matrix.npz \
       outputs/similarity/corpus_similarity_heatmap_clustered.png \
       outputs/similarity/corpus_similarity_heatmap_raw.png \
       outputs/similarity/cluster_order.npy \
       outputs/heatmaps/RP-P-1954-631_attn.png \
       outputs/heatmaps/RP-P-OB-19.650_attn.png \
       outputs/heatmaps/corpus_attention_grid.png \
       outputs/errors.log
       
Step 3 — run the full experiment:
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
print('=== STEP 3: Attention heatmaps ===')
import attention_heatmap
data = np.load(out_npz, allow_pickle=True)
attention_heatmap.visualize_corpus_attention(list(data['image_paths']))

print()
print('=== DONE — outputs at:', Path('outputs').resolve(), '===')
"
