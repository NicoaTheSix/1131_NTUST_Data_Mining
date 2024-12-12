# 1131_NTUST__Data_Mining# DataMining Final Report

## Environment setup

```bash
# Install virtual environment package if needed
sudo apt install python3.12-venv
# Create virtual environment
python3 -m venv .venv

source ./.venv/bin/activate

# Install dependencies
pip install numpy torch pandas Deprecated torch_geometric torchmetrics torch-summary ogb
pip install -U scikit-learn
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

## Run
```bash
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Actor/GraphCL.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name 'Actor' --gnn_type 'GCN' --prompt_type 'All-in-one' --device 0 --shot_num 5
```
