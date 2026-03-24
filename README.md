# Reinforcement Learning via Text Feedback

My implementation of the paper: [Expanding the Capabilities of Reinforcement Learning via Text Feedback](https://arxiv.org/abs/2602.02482)

Authors: Yuda Song, Lili Chen, Fahim Tajwar, Rémi Munos, Deepak Pathak, J. Andrew Bagnell, Aarti Singh, Andrea Zanette

## CLI Usage

### 1. Train a Model
```bash
uv run python main.py train --config hyperparams.json --algo rltf_sd
```
*(Options for `--algo`: `grpo`, `rltf_sd`, `rltf_fm`)*

### 2. Run Evaluation (Test Split)
```bash
uv run python main.py eval --config hyperparams.json --algo rltf_sd --num_samples 10
```

### 3. Interactive Chat
```bash
uv run python main.py eval --config hyperparams.json --algo rltf_fm --prompt "Let's think step by step. A store has 15 apples..."
```

### 4. Launch Jupyter Presentation
```bash
uv run python main.py nb --notebook ./notebooks --port 8888
```
