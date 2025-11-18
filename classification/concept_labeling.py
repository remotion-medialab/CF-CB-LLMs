"""
Modularized concept labeling for W&B integration.
Refactored from get_concept_labels.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import config as CFG
from utils import mean_pooling


def generate_concept_labels(
    dataset,
    concept_text_sim_model='mpnet',
    max_length=512,
    num_workers=0,
    force_regenerate=False,
):
    """
    Generate concept similarity labels for dataset.

    Args:
        dataset: Dataset name (e.g., 'SetFit/sst2')
        concept_text_sim_model: Model for computing concept similarities ('mpnet' or 'simcse')
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        force_regenerate: If True, regenerate even if labels exist. If False, load existing labels.

    Returns:
        dict: Metrics including similarity statistics and timing
    """
    # Check if concept labels already exist
    d_name = dataset.replace('/', '_')
    prefix = f"{concept_text_sim_model}_acs/{d_name}"
    train_labels_path = f"{prefix}/concept_labels_train.npy"
    val_labels_path = f"{prefix}/concept_labels_val.npy"

    if not force_regenerate and os.path.exists(train_labels_path):
        print(f"Found existing concept labels at {prefix}")
        print("Loading pre-computed concept labels...")

        # Load existing labels
        train_similarity = np.load(train_labels_path)
        val_similarity = None
        if os.path.exists(val_labels_path):
            val_similarity = np.load(val_labels_path)

        # Get concept set for metrics
        concept_set = CFG.concept_set[dataset]

        # Compute metrics from loaded labels
        metrics = {
            'num_concepts': len(concept_set),
            'train_similarity_mean': float(train_similarity.mean()),
            'train_similarity_std': float(train_similarity.std()),
            'train_similarity_min': float(train_similarity.min()),
            'train_similarity_max': float(train_similarity.max()),
        }

        if val_similarity is not None:
            metrics['val_similarity_mean'] = float(val_similarity.mean())
            metrics['val_similarity_std'] = float(val_similarity.std())
            metrics['val_similarity_min'] = float(val_similarity.min())
            metrics['val_similarity_max'] = float(val_similarity.max())

        print(f"Loaded concept labels: {train_similarity.shape}")
        print(f"Mean similarity: {metrics['train_similarity_mean']:.4f}")

        return metrics

    # Otherwise, generate concept labels from scratch
    print(f"Generating concept labels from scratch...")
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading {dataset}...")

    # Load datasets
    train_dataset = load_dataset(dataset, split='train')
    val_dataset = None
    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')

    print(f"Training data len: {len(train_dataset)}")
    if val_dataset:
        print(f"Val data len: {len(val_dataset)}")

    # Get concept set
    concept_set = CFG.concept_set[dataset]
    print(f"Concept len: {len(concept_set)}")

    # Load similarity model
    print(f"Loading {concept_text_sim_model} model...")
    if concept_text_sim_model == 'mpnet':
        tokenizer_sim = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        sim_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    elif concept_text_sim_model == 'simcse':
        tokenizer_sim = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    else:
        raise ValueError(f"Unknown concept_text_sim_model: {concept_text_sim_model}")

    sim_model.eval()

    # Encode and process datasets
    print("Tokenizing datasets...")
    encoded_sim_train_dataset = train_dataset.map(
        lambda e: tokenizer_sim(e[CFG.example_name[dataset]], padding=True, truncation=True,
                                max_length=max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_sim_train_dataset = encoded_sim_train_dataset.remove_columns(['title'])
    encoded_sim_train_dataset = encoded_sim_train_dataset[:len(encoded_sim_train_dataset)]

    if val_dataset:
        encoded_sim_val_dataset = val_dataset.map(
            lambda e: tokenizer_sim(e[CFG.example_name[dataset]], padding=True, truncation=True,
                                    max_length=max_length), batched=True,
            batch_size=len(val_dataset))
        encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns([CFG.example_name[dataset]])
        if dataset == 'SetFit/sst2':
            encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['label_text'])
        if dataset == 'dbpedia_14':
            encoded_sim_val_dataset = encoded_sim_val_dataset.remove_columns(['title'])
        encoded_sim_val_dataset = encoded_sim_val_dataset[:len(encoded_sim_val_dataset)]

    # Encode concepts
    encoded_c = tokenizer_sim(concept_set, padding=True, truncation=True, max_length=max_length)

    # Create data loaders
    class SimDataset(torch.utils.data.Dataset):
        def __init__(self, encode_sim):
            self.encode_sim = encode_sim

        def __getitem__(self, idx):
            return {key: torch.tensor(values[idx]) for key, values in self.encode_sim.items()}

        def __len__(self):
            return len(self.encode_sim['input_ids'])

    batch_size = 256 if concept_text_sim_model == 'mpnet' else 8
    train_sim_loader = torch.utils.data.DataLoader(
        SimDataset(encoded_sim_train_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    if val_dataset:
        val_sim_loader = torch.utils.data.DataLoader(
            SimDataset(encoded_sim_val_dataset),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

    # Get concept features
    print("Getting concept features...")
    encoded_c = {k: torch.tensor(v).to(device) for k, v in encoded_c.items()}

    with torch.no_grad():
        if concept_text_sim_model == 'mpnet':
            concept_features = sim_model(input_ids=encoded_c["input_ids"],
                                        attention_mask=encoded_c["attention_mask"])
            concept_features = mean_pooling(concept_features, encoded_c["attention_mask"])
        elif concept_text_sim_model == 'simcse':
            concept_features = sim_model(input_ids=encoded_c["input_ids"],
                                        attention_mask=encoded_c["attention_mask"],
                                        output_hidden_states=True,
                                        return_dict=True).pooler_output

        concept_features = F.normalize(concept_features, p=2, dim=1)

    # Compute similarities for training set
    print("Computing concept similarities for training set...")
    start_time = time.time()
    train_sim = []

    for i, batch_sim in enumerate(train_sim_loader):
        print(f"Processing batch {i+1}/{len(train_sim_loader)}", end="\r")
        batch_sim = {k: v.to(device) for k, v in batch_sim.items()}

        with torch.no_grad():
            if concept_text_sim_model == 'mpnet':
                text_features = sim_model(input_ids=batch_sim["input_ids"],
                                         attention_mask=batch_sim["attention_mask"])
                text_features = mean_pooling(text_features, batch_sim["attention_mask"])
            elif concept_text_sim_model == 'simcse':
                text_features = sim_model(input_ids=batch_sim["input_ids"],
                                         attention_mask=batch_sim["attention_mask"],
                                         output_hidden_states=True,
                                         return_dict=True).pooler_output

            text_features = F.normalize(text_features, p=2, dim=1)

        train_sim.append(text_features @ concept_features.T)

    train_similarity = torch.cat(train_sim, dim=0).cpu().detach().numpy()
    elapsed_time = (time.time() - start_time) / 3600
    print(f"\nConcept scoring completed in {elapsed_time:.4f} hours")

    # Compute similarities for validation set
    val_similarity = None
    if val_dataset:
        print("Computing concept similarities for validation set...")
        val_sim = []
        for batch_sim in val_sim_loader:
            batch_sim = {k: v.to(device) for k, v in batch_sim.items()}

            with torch.no_grad():
                if concept_text_sim_model == 'mpnet':
                    text_features = sim_model(input_ids=batch_sim["input_ids"],
                                             attention_mask=batch_sim["attention_mask"])
                    text_features = mean_pooling(text_features, batch_sim["attention_mask"])
                elif concept_text_sim_model == 'simcse':
                    text_features = sim_model(input_ids=batch_sim["input_ids"],
                                             attention_mask=batch_sim["attention_mask"],
                                             output_hidden_states=True,
                                             return_dict=True).pooler_output

                text_features = F.normalize(text_features, p=2, dim=1)

            val_sim.append(text_features @ concept_features.T)

        val_similarity = torch.cat(val_sim, dim=0).cpu().detach().numpy()

    # Save to disk
    d_name = dataset.replace('/', '_')
    prefix = f"{concept_text_sim_model}_acs/{d_name}"
    os.makedirs(prefix, exist_ok=True)

    np.save(f"{prefix}/concept_labels_train.npy", train_similarity)
    if val_similarity is not None:
        np.save(f"{prefix}/concept_labels_val.npy", val_similarity)

    print(f"Saved concept labels to {prefix}")

    # Compute metrics
    metrics = {
        'num_concepts': len(concept_set),
        'train_similarity_mean': float(train_similarity.mean()),
        'train_similarity_std': float(train_similarity.std()),
        'train_similarity_min': float(train_similarity.min()),
        'train_similarity_max': float(train_similarity.max()),
    }

    if val_similarity is not None:
        metrics['val_similarity_mean'] = float(val_similarity.mean())
        metrics['val_similarity_std'] = float(val_similarity.std())
        metrics['val_similarity_min'] = float(val_similarity.min())
        metrics['val_similarity_max'] = float(val_similarity.max())

    return metrics
