## Setup

Setup is pretty simple, requires that you get engaging working (if off campus you'll need the VPN)
- NOTE: if you want to be able to modify the code on the cluster, you'll need to set up ssh on the IDE
    - Open up VS Code
    - Install the **Remote - SSH** extension (by Microsoft)
    - Make sure you can SSH from your local terminal first
      `ssh <kerb>@orcd-login002`
    - In VS Code, open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
    - Select **“Remote-SSH: Add New SSH Host…”**
    - When it asks for the SSH command, paste something like:
      ```bash
      ssh <kerb>@orcd-login002
      ```
    - Choose the default SSH config file (`~/.ssh/config`) when prompted
    - (Optional but nice) Give it a short alias by editing `~/.ssh/config`:
      ```bash
      Host engaging
          HostName orcd-login002
          User <kerb>
          IdentityFile ~/.ssh/id_ed25519
      ```
    - Now open the Command Palette again → **“Remote-SSH: Connect to Host…”**
      - Pick `engaging` (or the full `ssh <kerb>@orcd-login002` entry)
      - VS Code will open a new window attached to the cluster
    - In the remote window, go to **File → Open Folder…** and open your home dir or project dir
    - Use **Terminal → New Terminal** inside that window — that terminal is now on Engaging and all the commands below should be run there


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

Also replace email with your own email so u can get updates

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
- Refer to WANDB_USAGE_GUIDE.md for more detailed WANDB instructions (this file was AI generated so lmk if any questions)
