# W&B Sweep Usage Guide for CB-LLM Classification

This guide explains how to use the Weights & Biases integration for the CB-LLM classification pipeline.

## Quick Start

### 1. Setup

```bash
cd CF-CB-LLMs/classification

# Install W&B (if not already installed)
pip install wandb

# Login to W&B
export WANDB_API_KEY=<your_api_key>
wandb login
```

Get your API key from: https://wandb.ai/authorize

### 2. Run Single Experiment (Recommended First)

This runs one experiment with the default parameters from the README:

```bash
# Create the sweep
wandb sweep --project CB-LLM-Classification-Test config/cbllm_sweep_single.yaml

# This will output something like:
# wandb: Created sweep with ID: abc123xyz
# wandb: Run sweep agent with: wandb agent username/CB-LLM-Classification-Test/abc123xyz

# Run the agent (use the exact command from above)
wandb agent username/CB-LLM-Classification-Test/abc123xyz
```

### 3. View Results

After the run completes, you can view results at:

```
https://wandb.ai/<username>/CB-LLM-Classification-Test
```

## What Gets Logged

### Stage 1: Concept Labeling

- `stage1_concept_generation_time` - Time to generate concept labels (hours)
- `stage1_num_concepts` - Number of concepts in the concept set
- `stage1_train_similarity_mean` - Mean similarity score on training set
- `stage1_train_similarity_std` - Standard deviation of similarities
- `stage1_val_similarity_mean` - Mean similarity on validation set (SST2 only)

### Stage 2: CBL Training

- `epoch` - Current epoch number
- `cbl_train_loss_epoch` - Training loss per epoch
- `cbl_val_loss_epoch` - Validation loss per epoch (SST2 only)
- `stage2_cbl_best_loss` - Best validation loss achieved
- `stage2_cbl_best_epoch` - Epoch with best validation loss
- `stage2_cbl_training_time` - Total CBL training time (hours)

### Stage 3: Final Layer Training

- `stage3_fl_train_features_mean` - Mean of concept features
- `stage3_fl_train_features_std` - Std dev of concept features
- `stage3_saga_iterations` - Number of SAGA iterations
- `stage3_fl_training_time` - Training time for final layer (hours)

### Stage 4: Evaluation

- `stage4_test_accuracy` - **Main metric**: Test set accuracy
- `stage4_test_accuracy_class_0`, `stage4_test_accuracy_class_1`, ... - Per-class accuracy
- `stage4_concept_features_mean` - Mean concept activation
- `stage4_active_concepts_ratio` - Fraction of active (non-zero) concepts
- `stage4_linear_layer_sparsity` - Sparsity of final layer weights
- `stage4_evaluation_time` - Evaluation time (hours)

### Summary Metrics

- `total_pipeline_time` - Total time for entire pipeline
- `final_test_accuracy` - Final test accuracy (main result)
- `cbl_best_loss` - Best CBL validation loss
- `active_concepts_ratio` - Ratio of active concepts

### Model Artifacts

- `cbl-{dataset}-{backbone}-{run_id}` - CBL checkpoint
- `final-layer-{dataset}-{backbone}-{run_id}` - Final layer weights

## File Structure

After running, you'll see:

```
CF-CB-LLMs/classification/
├── config/
│   └── cbllm_sweep_single.yaml       # Sweep configuration
├── main_sweep.py                      # Main orchestrator
├── concept_labeling.py                # Stage 1: Concept labeling
├── cbl_training.py                    # Stage 2: CBL training
├── final_layer_training.py            # Stage 3: Final layer
├── evaluation.py                      # Stage 4: Evaluation
├── mpnet_acs/                         # Generated concept labels
│   └── SetFit_sst2/
│       ├── concept_labels_train.npy
│       ├── concept_labels_val.npy
│       └── roberta_cbm/               # Model checkpoints
│           ├── cbl_acc.pt             # CBL model
│           └── linear_layer.pt        # Final layer
└── wandb/                             # Local W&B logs
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution: Make sure you're in the classification directory
cd CF-CB-LLMs/classification
python main_sweep.py  # Should work now
```

### Issue: "Concept labels not found"

This means Stage 1 failed or was interrupted. The pipeline should automatically run Stage 1, but if it fails:

```bash
# Manually run concept labeling
python -c "from concept_labeling import generate_concept_labels; generate_concept_labels('SetFit/sst2', 'mpnet')"
```

### Issue: "Linear layer not found"

This means Stage 3 failed. Check the W&B logs for Stage 3 errors.

### Issue: CUDA out of memory

```bash
# Solution: Reduce batch size
# Edit config/cbllm_sweep_single.yaml:
batch_size:
  values: [8]  # Instead of 16
```

### Issue: Sweep agent crashes

The sweep will automatically resume. Just run the agent command again:

```bash
wandb agent username/CB-LLM-Classification-Test/abc123xyz
```

It will skip completed runs.

## Advanced Usage

### Running Multiple Experiments

To run experiments with different hyperparameters, create a new sweep config:

```yaml
# config/cbllm_sweep_multi.yaml
program: main_sweep.py
method: grid
project: CB-LLM-Classification
parameters:
  dataset:
    values: ["SetFit/sst2"]
  backbone:
    values: ["roberta", "gpt2"] # Try both backbones
  concept_text_sim_model:
    values: ["mpnet"]
  automatic_concept_correction:
    values: [true, false] # Compare with and without ACC
  # ... other parameters
```

Then run:

```bash
wandb sweep --project CB-LLM-Classification config/cbllm_sweep_multi.yaml
wandb agent <sweep-id>
```

### Running on Multiple GPUs

```bash
# Terminal 1 - GPU 0
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id>

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-id>
```

The agents will automatically coordinate and avoid running the same experiment twice.

### Analyzing Results

After runs complete, use the W&B dashboard to:

1. **Compare runs**: Click "Compare" to see side-by-side metrics
2. **Create charts**: Plot accuracy vs. hyperparameters
3. **Filter runs**: Find best performing configurations
4. **Download artifacts**: Get trained model checkpoints

Example queries in the dashboard:

```python
# Find runs with accuracy > 0.9
stage4_test_accuracy > 0.9

# Compare ACC vs no ACC
automatic_concept_correction = true
automatic_concept_correction = false
```

### Accessing Models Programmatically

```python
import wandb

# Download best model
api = wandb.Api()
run = api.run("username/CB-LLM-Classification-Test/<run-id>")

# Download artifacts
for artifact in run.logged_artifacts():
    if artifact.type == "model":
        artifact_dir = artifact.download()
        print(f"Downloaded to {artifact_dir}")
```

## Expected Results

Based on the CB-LLM paper, you should see approximately:

| Configuration    | Expected Test Accuracy |
| ---------------- | ---------------------- |
| RoBERTa + ACC    | ~0.94                  |
| RoBERTa (no ACC) | ~0.90                  |
| GPT-2 + ACC      | ~0.92                  |
| GPT-2 (no ACC)   | ~0.87                  |

If your accuracy is significantly lower:

1. Check that Stage 1 completed successfully
2. Verify concept labels were generated correctly
3. Ensure CBL training converged (check `cbl_train_loss_epoch` plots)
4. Check for CUDA/device issues

## Next Steps

After confirming the single experiment works:

1. **Run comparative studies**: Test with/without ACC, different backbones
2. **Try other datasets**: ag_news, yelp_polarity, dbpedia_14
3. **Hyperparameter tuning**: Experiment with dropout, learning rates
4. **Analyze concepts**: Visualize which concepts are most active
