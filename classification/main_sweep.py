#!/usr/bin/env python3
"""
Main orchestrator for CB-LLM classification pipeline with W&B integration.
Runs all three stages: concept labeling, CBL training, and final layer training.
"""

import os
import sys
import wandb
import torch
import numpy as np
import time
from pathlib import Path

# Import modularized functions from the refactored scripts
from concept_labeling import generate_concept_labels
from cbl_training import train_cbl_model
from final_layer_training import train_final_layer
from evaluation import evaluate_model

import config as CFG


def main():
    """Main pipeline orchestrator with W&B integration."""

    # Initialize W&B with custom run name
    run = wandb.init()
    config = wandb.config

    # Create descriptive run name based on configuration
    dataset_short = config.dataset.split('/')[-1]
    acc_str = "ACC" if config.automatic_concept_correction else "noACC"
    run_name = f"{dataset_short}_{config.backbone}_{config.concept_text_sim_model}_{acc_str}_lr{config.learning_rate}_seed{config.seed}"

    # Update run name and add tags
    run.name = run_name
    run.tags = [
        config.dataset,
        config.backbone,
        config.concept_text_sim_model,
        acc_str,
        f"seed{config.seed}"
    ]
    run.notes = f"CB-LLM classification on {config.dataset} with {config.backbone} backbone"

    # Save the config update
    run.save()

    print("="*80)
    print("CB-LLM Classification Pipeline with W&B Integration")
    print("="*80)
    print(f"Run ID: {run.id}")
    print(f"Run name: {run.name}")
    print("="*80)

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Log basic configuration
    wandb.log({
        "num_concepts": len(CFG.concept_set[config.dataset]),
        "dataset_name": config.dataset,
    })

    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Backbone: {config.backbone}")
    print(f"  Concept Model: {config.concept_text_sim_model}")
    print(f"  ACC: {config.automatic_concept_correction}")
    print(f"  Tune CBL Only: {config.tune_cbl_only}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  CBL Epochs: {config.cbl_epochs}")
    print(f"  SAGA Epochs: {config.saga_epoch}")
    print(f"  Seed: {config.seed}")
    print("="*80)

    # Track total pipeline time
    pipeline_start_time = time.time()

    # =========================================================================
    # STAGE 1: Concept Labeling
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1: Generating Concept Labels")
    print("="*80)

    stage_start_time = time.time()

    try:
        concept_metrics = generate_concept_labels(
            dataset=config.dataset,
            concept_text_sim_model=config.concept_text_sim_model,
            max_length=config.max_length,
            num_workers=config.num_workers,
        )
        concept_metrics['concept_generation_time'] = (time.time() - stage_start_time) / 3600

        # Log Stage 1 metrics
        for key, value in concept_metrics.items():
            wandb.log({f"stage1_{key}": value})

        print(f"\n✓ Stage 1 completed in {concept_metrics['concept_generation_time']:.4f} hours")
        print(f"  Mean similarity: {concept_metrics['train_similarity_mean']:.4f}")

    except Exception as e:
        print(f"\n✗ Stage 1 failed: {e}")
        wandb.log({"stage1_error": str(e)})
        raise

    # =========================================================================
    # STAGE 2: CBL Training
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 2: Training Concept Bottleneck Layer")
    print("="*80)

    stage_start_time = time.time()

    try:
        cbl_metrics, cbl_path = train_cbl_model(
            dataset=config.dataset,
            backbone=config.backbone,
            concept_text_sim_model=config.concept_text_sim_model,
            automatic_concept_correction=config.automatic_concept_correction,
            tune_cbl_only=config.tune_cbl_only,
            dropout=config.dropout,
            batch_size=config.batch_size if not config.tune_cbl_only else config.cbl_only_batch_size,
            max_length=config.max_length,
            num_workers=config.num_workers,
            learning_rate=config.learning_rate,
            epochs=config.cbl_epochs,
            seed=config.seed,
            wandb_run=run,  # Pass run for epoch-level logging
        )
        cbl_metrics['cbl_training_time'] = (time.time() - stage_start_time) / 3600

        # Log Stage 2 metrics
        for key, value in cbl_metrics.items():
            wandb.log({f"stage2_{key}": value})

        print(f"\n✓ Stage 2 completed in {cbl_metrics['cbl_training_time']:.4f} hours")
        print(f"  Best loss: {cbl_metrics['cbl_best_loss']:.4f}")
        print(f"  Best epoch: {cbl_metrics['cbl_best_epoch']}")
        print(f"  Model saved to: {cbl_path}")

        # Save CBL checkpoint as W&B artifact
        artifact = wandb.Artifact(
            name=f"cbl-{config.dataset.replace('/', '_')}-{config.backbone}-{run.id}",
            type="model",
            description=f"CBL checkpoint for {config.dataset}",
            metadata=dict(config)
        )
        artifact.add_file(cbl_path)
        run.log_artifact(artifact)
        print(f"  Uploaded CBL to W&B artifacts")

    except Exception as e:
        print(f"\n✗ Stage 2 failed: {e}")
        wandb.log({"stage2_error": str(e)})
        raise

    # =========================================================================
    # STAGE 3: Final Layer Training
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 3: Training Final Layer")
    print("="*80)

    stage_start_time = time.time()

    try:
        fl_metrics, fl_path = train_final_layer(
            cbl_path=cbl_path,
            saga_epoch=config.saga_epoch,
            saga_batch_size=config.saga_batch_size,
            batch_size=config.batch_size,
            max_length=config.max_length,
            num_workers=config.num_workers,
            dropout=config.dropout,
        )
        fl_metrics['fl_training_time'] = (time.time() - stage_start_time) / 3600

        # Log Stage 3 metrics
        for key, value in fl_metrics.items():
            wandb.log({f"stage3_{key}": value})

        print(f"\n✓ Stage 3 completed in {fl_metrics['fl_training_time']:.4f} hours")
        print(f"  Linear layer saved to: {fl_path}")

        # Save final layer as W&B artifact
        artifact = wandb.Artifact(
            name=f"final-layer-{config.dataset.replace('/', '_')}-{config.backbone}-{run.id}",
            type="model",
            description=f"Final layer for {config.dataset}",
            metadata=dict(config)
        )
        artifact.add_file(fl_path)
        run.log_artifact(artifact)
        print(f"  Uploaded final layer to W&B artifacts")

    except Exception as e:
        print(f"\n✗ Stage 3 failed: {e}")
        wandb.log({"stage3_error": str(e)})
        raise

    # =========================================================================
    # STAGE 4: Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 4: Evaluating CB-LLM")
    print("="*80)

    stage_start_time = time.time()

    try:
        eval_metrics = evaluate_model(
            cbl_path=cbl_path,
            batch_size=256,
            max_length=config.max_length,
            num_workers=config.num_workers,
            dropout=config.dropout,
        )
        eval_metrics['evaluation_time'] = (time.time() - stage_start_time) / 3600

        # Log Stage 4 metrics
        for key, value in eval_metrics.items():
            if key != 'confusion_matrix':  # Skip confusion matrix for now
                wandb.log({f"stage4_{key}": value})

        # Log confusion matrix as a W&B table
        if 'confusion_matrix' in eval_metrics:
            n_classes = CFG.class_num[config.dataset]
            class_labels = [f"Class {i}" for i in range(n_classes)]

            # Create confusion matrix visualization
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=None,
                    preds=None,
                    class_names=class_labels,
                    title="Confusion Matrix"
                )
            })

        print(f"\n✓ Stage 4 completed in {eval_metrics['evaluation_time']:.4f} hours")
        print(f"  Test Accuracy: {eval_metrics['test_accuracy']:.4f}")
        print(f"  Active concepts ratio: {eval_metrics['active_concepts_ratio']:.4f}")

    except Exception as e:
        print(f"\n✗ Stage 4 failed: {e}")
        wandb.log({"stage4_error": str(e)})
        raise

    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    total_time = (time.time() - pipeline_start_time) / 3600

    print("\n" + "="*80)
    print("Pipeline Completed Successfully!")
    print("="*80)
    print(f"Total time: {total_time:.4f} hours")
    print(f"  Stage 1 (Concept Labeling): {concept_metrics['concept_generation_time']:.4f} hours")
    print(f"  Stage 2 (CBL Training): {cbl_metrics['cbl_training_time']:.4f} hours")
    print(f"  Stage 3 (Final Layer): {fl_metrics['fl_training_time']:.4f} hours")
    print(f"  Stage 4 (Evaluation): {eval_metrics['evaluation_time']:.4f} hours")
    print(f"\nFinal Test Accuracy: {eval_metrics['test_accuracy']:.4f}")
    print("="*80)

    # Log summary metrics
    wandb.summary.update({
        'total_pipeline_time': total_time,
        'final_test_accuracy': eval_metrics['test_accuracy'],
        'cbl_best_loss': cbl_metrics['cbl_best_loss'],
        'cbl_best_epoch': cbl_metrics['cbl_best_epoch'],
        'active_concepts_ratio': eval_metrics['active_concepts_ratio'],
    })

    # Close W&B run
    wandb.finish()

    print("\n✓ Results logged to W&B")
    print(f"View results at: {run.url}")


if __name__ == "__main__":
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        wandb.finish(exit_code=1)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        sys.exit(1)
