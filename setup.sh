

apt update
apt install -y unzip
apt install -y zip
apt install -y nvtop
apt install -y tmux
apt-get update -y
apt-get install -y build-essential
export CC=gcc


pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# uv pip install nbstripout
# nbstripout --install

uv pip install wandb
wandb login {YOUR_TOKEN}

curl https://getcroc.schollz.com | bash

uv pip install hf_transfer
uv pip install huggingface_hub
uv pip install hf_transfer

huggingface-cli login --token {YOUR_TOKEN}
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
