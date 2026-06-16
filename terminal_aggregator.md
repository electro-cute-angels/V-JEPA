# 1. Go to the project and activate the environment
cd "/home/riccardo/Desktop/V-JEPA 2.1/vjepa_experiments"
source ../venv/bin/activate

# 2. Clear old outputs (run this if you changed images or want a fresh run)
find outputs/ -type f ! -name ".gitkeep" -delete && echo "Outputs cleared."

# 3. Run the full pipeline (embed → similarity → heatmaps → layer-wise)
python3 run_all.py
