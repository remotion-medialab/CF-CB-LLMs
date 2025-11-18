"""
Modularized CBL training for W&B integration.
Refactored from train_CBL.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import cos_sim_cubed, get_labels, eos_pooling


def train_cbl_model(
    dataset,
    backbone,
    concept_text_sim_model,
    automatic_concept_correction,
    tune_cbl_only,
    dropout,
    batch_size,
    max_length,
    num_workers,
    learning_rate,
    epochs,
    seed,
    wandb_run=None,
):
    """
    Train Concept Bottleneck Layer.

    Args:
        dataset: Dataset name
        backbone: 'roberta' or 'gpt2'
        concept_text_sim_model: Concept similarity model used
        automatic_concept_correction: Whether to apply ACC
        tune_cbl_only: If True, freeze backbone
        dropout: Dropout rate
        batch_size: Batch size for training
        max_length: Max sequence length
        num_workers: Number of workers
        learning_rate: Learning rate
        epochs: Number of training epochs
        seed: Random seed
        wandb_run: W&B run object for logging

    Returns:
        tuple: (metrics_dict, cbl_checkpoint_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Training CBL for {dataset} with {backbone} backbone")
    print(f"ACC: {automatic_concept_correction}, tune_cbl_only: {tune_cbl_only}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("Loading data...")
    train_dataset = load_dataset(dataset, split='train')
    val_dataset = None
    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')

    print(f"Training data len: {len(train_dataset)}")
    if val_dataset:
        print(f"Val data len: {len(val_dataset)}")

    # Tokenize
    print("Tokenizing...")
    if backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("backbone should be roberta or gpt2")

    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True,
                           max_length=max_length), batched=True, batch_size=len(train_dataset))
    encoded_train_dataset = encoded_train_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_train_dataset = encoded_train_dataset.remove_columns(['title'])
    encoded_train_dataset = encoded_train_dataset[:len(encoded_train_dataset)]

    if val_dataset:
        encoded_val_dataset = val_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True,
                               max_length=max_length), batched=True, batch_size=len(val_dataset))
        encoded_val_dataset = encoded_val_dataset.remove_columns([CFG.example_name[dataset]])
        if dataset == 'SetFit/sst2':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['label_text'])
        if dataset == 'dbpedia_14':
            encoded_val_dataset = encoded_val_dataset.remove_columns(['title'])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

    # Load concept set
    concept_set = CFG.concept_set[dataset]
    print(f"Concept len: {len(concept_set)}")

    # Load concept similarity labels
    d_name = dataset.replace('/', '_')
    prefix = f"./{concept_text_sim_model}_acs/{d_name}/"
    train_similarity = np.load(f"{prefix}/concept_labels_train.npy")

    if val_dataset:
        val_similarity = np.load(f"{prefix}/concept_labels_val.npy")

    # Apply Automatic Concept Correction if enabled
    if automatic_concept_correction:
        print("Applying Automatic Concept Correction...")
        start = time.time()
        for i in range(train_similarity.shape[0]):
            for j in range(len(concept_set)):
                if get_labels(j, dataset) != encoded_train_dataset["label"][i]:
                    train_similarity[i][j] = 0.0
                else:
                    if train_similarity[i][j] < 0.0:
                        train_similarity[i][j] = 0.0

        if val_dataset:
            for i in range(val_similarity.shape[0]):
                for j in range(len(concept_set)):
                    if get_labels(j, dataset) != encoded_val_dataset["label"][i]:
                        val_similarity[i][j] = 0.0
                    else:
                        if val_similarity[i][j] < 0.0:
                            val_similarity[i][j] = 0.0

        elapsed = time.time() - start
        print(f"ACC completed in {elapsed:.2f} seconds")

    # Create dataloaders
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, encode_roberta, s):
            self.encode_roberta = encode_roberta
            self.s = s

        def __getitem__(self, idx):
            t = {key: torch.tensor(values[idx]) for key, values in self.encode_roberta.items()}
            y = torch.FloatTensor(self.s[idx])
            return t, y

        def __len__(self):
            return len(self.encode_roberta['input_ids'])

    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(encoded_train_dataset, train_similarity),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            ClassificationDataset(encoded_val_dataset, val_similarity),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

    # Initialize model
    print("Initializing model...")
    if backbone == 'roberta':
        if tune_cbl_only:
            print("Preparing CBL only (frozen backbone)...")
            cbl = CBL(len(concept_set), dropout).to(device)
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=learning_rate)
        else:
            print("Preparing backbone (RoBERTa) + CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=learning_rate)
    elif backbone == 'gpt2':
        if tune_cbl_only:
            print("Preparing CBL only (frozen backbone)...")
            cbl = CBL(len(concept_set), dropout).to(device)
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
            optimizer = torch.optim.Adam(cbl.parameters(), lr=learning_rate)
        else:
            print("Preparing backbone (GPT-2) + CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), dropout).to(device)
            optimizer = torch.optim.Adam(backbone_cbl.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}:")

        # Train
        if tune_cbl_only:
            cbl.train()
        else:
            backbone_cbl.train()

        training_loss = []
        for i, batch in enumerate(train_loader):
            batch_text, batch_sim = batch[0], batch[1]
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_sim = batch_sim.to(device)

            if tune_cbl_only:
                with torch.no_grad():
                    LM_features = preLM(input_ids=batch_text["input_ids"],
                                       attention_mask=batch_text["attention_mask"]).last_hidden_state
                    if backbone == 'roberta':
                        LM_features = LM_features[:, 0, :]
                    elif backbone == 'gpt2':
                        LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                cbl_features = cbl(LM_features)
            else:
                cbl_features = backbone_cbl(batch_text)

            loss = -cos_sim_cubed(cbl_features, batch_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch {i+1}/{len(train_loader)}, loss: {loss.item():.4f}", end="\r")
            training_loss.append(loss.item())

        avg_train_loss = sum(training_loss) / len(training_loss)
        print(f"\nTraining loss: {avg_train_loss:.4f}")

        # Validate
        if val_dataset:
            if tune_cbl_only:
                cbl.eval()
            else:
                backbone_cbl.eval()

            val_loss = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_text, batch_sim = batch[0], batch[1]
                    batch_text = {k: v.to(device) for k, v in batch_text.items()}
                    batch_sim = batch_sim.to(device)

                    if tune_cbl_only:
                        LM_features = preLM(input_ids=batch_text["input_ids"],
                                           attention_mask=batch_text["attention_mask"]).last_hidden_state
                        if backbone == 'roberta':
                            LM_features = LM_features[:, 0, :]
                        elif backbone == 'gpt2':
                            LM_features = eos_pooling(LM_features, batch_text["attention_mask"])
                        cbl_features = cbl(LM_features)
                    else:
                        cbl_features = backbone_cbl(batch_text)

                    loss = -cos_sim_cubed(cbl_features, batch_sim)
                    val_loss.append(loss.item())

            avg_val_loss = sum(val_loss) / len(val_loss)
            print(f"Validation loss: {avg_val_loss:.4f}")

            # Log to W&B
            if wandb_run:
                wandb_run.log({
                    'epoch': epoch + 1,
                    'cbl_train_loss_epoch': avg_train_loss,
                    'cbl_val_loss_epoch': avg_val_loss,
                })

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                print(f"New best validation loss: {best_loss:.4f}")
        else:
            # No validation set, log training loss only
            if wandb_run:
                wandb_run.log({
                    'epoch': epoch + 1,
                    'cbl_train_loss_epoch': avg_train_loss,
                })

    training_time = (time.time() - start_time) / 3600
    print(f"\nCBL training completed in {training_time:.4f} hours")

    # Save model
    save_dir = f"./{concept_text_sim_model}_acs/{d_name}/{backbone}_cbm/"
    os.makedirs(save_dir, exist_ok=True)

    model_name = "cbl"
    if tune_cbl_only:
        model_name += "_no_backbone"
    if automatic_concept_correction:
        model_name += "_acc"

    checkpoint_path = f"{save_dir}/{model_name}.pt"
    print(f"Saving model to {checkpoint_path}")

    if tune_cbl_only:
        torch.save(cbl.state_dict(), checkpoint_path)
    else:
        torch.save(backbone_cbl.state_dict(), checkpoint_path)

    # Prepare metrics
    metrics = {
        'cbl_final_train_loss': avg_train_loss,
    }

    if val_dataset:
        metrics['cbl_best_loss'] = best_loss
        metrics['cbl_final_val_loss'] = avg_val_loss
        metrics['cbl_best_epoch'] = best_epoch
    else:
        metrics['cbl_best_loss'] = avg_train_loss
        metrics['cbl_best_epoch'] = epochs

    return metrics, checkpoint_path
