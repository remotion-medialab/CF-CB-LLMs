## Setup

**Requirements:**

- CUDA 12.1 or later
- Python 3.11
- PyTorch 2.2+

### MIT Engaging Cluster Setup

The MIT Engaging cluster provides GPU resources for training. Follow these steps to set up your environment:

#### 1. SSH into Engaging

```bash
ssh <username>@orcd-login002

```

#### 2. Create Directory and Clone repository

```bash
git clone git@github.com:remotion-medialab/CF-CB-LLMs.git
cd CF_CB_LLM/CF-CB-LLMs/classification

```

#### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (manually do instead of relying on requirements.txt - will update this later)
pip install <insert dependency from requirements.txt w/ no version>
pip install wandb
```

#### 4. Configure Weights & Biases

```bash
# Set W&B API key (get from https://wandb.ai/authorize)
export WANDB_API_KEY=<your-api-key>

# Login to W&B
wandb login

```

#### 5. Create wandb sweep agent

```bash
wandb sweep --project CB-LLM-Classification-Test config/cbllm_sweep_single.yaml
```

#### 6. Submit Batch Job

Go into train_model.slurm and replace the sweep id with the ID given in the previous command.

Also Replace Email with your own email

```bash
sbatch train_model.slurm
```

Check job status:

```bash
squeue -u <username>  # Check queue
tail -f cbllm_<job-id>.out  # Monitor output
scancel <job-id>  # Cancel job if needed
```

**View results:**

- Dashboard: `https://wandb.ai/<username>/CB-LLM-Classification`
- Run page shows metrics, plots, and model artifacts
