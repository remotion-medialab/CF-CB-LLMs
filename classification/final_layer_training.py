"""
Modularized final layer training for W&B integration.
Refactored from train_FL.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from utils import normalize, eos_pooling


def train_final_layer(
    cbl_path,
    saga_epoch=500,
    saga_batch_size=256,
    batch_size=128,
    max_length=512,
    num_workers=0,
    dropout=0.1,
):
    """
    Train final linear layer on top of CBL using GLM-SAGA.

    Args:
        cbl_path: Path to trained CBL checkpoint
        saga_epoch: Number of SAGA iterations
        saga_batch_size: Batch size for SAGA
        batch_size: Batch size for feature extraction
        max_length: Max sequence length
        num_workers: Number of workers
        dropout: Dropout rate

    Returns:
        tuple: (metrics_dict, final_layer_checkpoint_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Training final linear layer...")

    # Parse CBL path to get configuration
    path_parts = cbl_path.split("/")
    acs = path_parts[0]
    dataset = path_parts[1]
    if 'sst2' in dataset:
        dataset = dataset.replace('_', '/')
    backbone = path_parts[2]
    cbl_name = path_parts[-1]

    print(f"Dataset: {dataset}, Backbone: {backbone}, ACS: {acs}")

    # Load datasets
    print("Loading data...")
    train_dataset = load_dataset(dataset, split='train')
    val_dataset = None
    if dataset == 'SetFit/sst2':
        val_dataset = load_dataset(dataset, split='validation')
    test_dataset = load_dataset(dataset, split='test')

    print(f"Training data len: {len(train_dataset)}")
    if val_dataset:
        print(f"Val data len: {len(val_dataset)}")
    print(f"Test data len: {len(test_dataset)}")

    # Tokenize
    print("Tokenizing...")
    if 'roberta' in backbone:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif 'gpt2' in backbone:
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

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True,
                           max_length=max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    # Create dataloaders
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __getitem__(self, idx):
            return {key: torch.tensor(values[idx]) for key, values in self.texts.items()}

        def __len__(self):
            return len(self.texts['input_ids'])

    train_loader = torch.utils.data.DataLoader(
        ClassificationDataset(encoded_train_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            ClassificationDataset(encoded_val_dataset),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

    test_loader = torch.utils.data.DataLoader(
        ClassificationDataset(encoded_test_dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Load CBL model
    concept_set = CFG.concept_set[dataset]

    print("Loading CBL model...")
    if 'roberta' in backbone:
        if 'no_backbone' in cbl_name:
            cbl = CBL(len(concept_set), dropout).to(device)
            cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            backbone_cbl = RobertaCBL(len(concept_set), dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            backbone_cbl.eval()
    elif 'gpt2' in backbone:
        if 'no_backbone' in cbl_name:
            cbl = CBL(len(concept_set), dropout).to(device)
            cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            backbone_cbl = GPT2CBL(len(concept_set), dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            backbone_cbl.eval()

    # Extract concept features
    print("Extracting concept features from training set...")
    FL_train_features = []
    FL_train_labels = []

    with torch.no_grad():
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if 'no_backbone' in cbl_name:
                features = preLM(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]).last_hidden_state
                if 'roberta' in backbone:
                    features = features[:, 0, :]
                elif 'gpt2' in backbone:
                    features = eos_pooling(features, batch["attention_mask"])
                concept_features = cbl(features)
            else:
                concept_features = backbone_cbl(batch)

            FL_train_features.append(concept_features.cpu())

    FL_train_features = torch.cat(FL_train_features, dim=0)
    FL_train_labels = torch.LongTensor(encoded_train_dataset['label'])

    if val_dataset:
        print("Extracting concept features from validation set...")
        FL_val_features = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if 'no_backbone' in cbl_name:
                    features = preLM(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).last_hidden_state
                    if 'roberta' in backbone:
                        features = features[:, 0, :]
                    elif 'gpt2' in backbone:
                        features = eos_pooling(features, batch["attention_mask"])
                    concept_features = cbl(features)
                else:
                    concept_features = backbone_cbl(batch)

                FL_val_features.append(concept_features.cpu())

        FL_val_features = torch.cat(FL_val_features, dim=0)
        FL_val_labels = torch.LongTensor(encoded_val_dataset['label'])

    print("Extracting concept features from test set...")
    FL_test_features = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if 'no_backbone' in cbl_name:
                features = preLM(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]).last_hidden_state
                if 'roberta' in backbone:
                    features = features[:, 0, :]
                elif 'gpt2' in backbone:
                    features = eos_pooling(features, batch["attention_mask"])
                concept_features = cbl(features)
            else:
                concept_features = backbone_cbl(batch)

            FL_test_features.append(concept_features.cpu())

    FL_test_features = torch.cat(FL_test_features, dim=0)
    FL_test_labels = torch.LongTensor(encoded_test_dataset['label'])

    # Compute feature statistics
    print("Computing feature statistics...")
    metrics = {
        'fl_train_features_mean': float(FL_train_features.mean()),
        'fl_train_features_std': float(FL_train_features.std()),
        'fl_train_features_min': float(FL_train_features.min()),
        'fl_train_features_max': float(FL_train_features.max()),
    }

    # Create dataloaders for SAGA
    indexed_train_ds = IndexedTensorDataset(FL_train_features, FL_train_labels)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=saga_batch_size, shuffle=True)

    if val_dataset:
        val_ds = TensorDataset(FL_val_features, FL_val_labels)
        val_loader = DataLoader(val_ds, batch_size=saga_batch_size, shuffle=False)

    test_ds = TensorDataset(FL_test_features, FL_test_labels)
    test_loader = DataLoader(test_ds, batch_size=saga_batch_size, shuffle=False)

    # Get number of classes
    n_classes = CFG.class_num[dataset]

    # Initialize linear layer
    print(f"Initializing linear layer: {FL_train_features.shape[1]} -> {n_classes}")
    linear = torch.nn.Linear(FL_train_features.shape[1], n_classes)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    # SAGA parameters (from original implementation)
    STEP_SIZE = 0.05
    ALPHA = 0.99

    # Train linear layer using GLM-SAGA
    print(f"Training linear layer with GLM-SAGA ({saga_epoch} iterations)...")
    if val_dataset:
        output_proj = glm_saga(
            linear,
            indexed_train_loader,
            STEP_SIZE,
            saga_epoch,
            ALPHA,
            k=10,
            val_loader=val_loader,
            test_loader=test_loader,
            do_zero=True,
            n_classes=n_classes
        )
    else:
        output_proj = glm_saga(
            linear,
            indexed_train_loader,
            STEP_SIZE,
            saga_epoch,
            ALPHA,
            k=10,
            test_loader=test_loader,
            do_zero=True,
            n_classes=n_classes
        )

    print(f"Training completed. Test accuracy: {output_proj['path'][-1]['metrics']['acc_test']:.4f}")

    # Extract trained weights
    W_g = output_proj['path'][-1]['weight']
    b_g = output_proj['path'][-1]['bias']

    # Save linear layer as state dict
    linear_layer_path = os.path.join(os.path.dirname(cbl_path), "linear_layer.pt")
    linear_state = {
        'weight': W_g,
        'bias': b_g
    }
    torch.save(linear_state, linear_layer_path)
    print(f"Saved linear layer to {linear_layer_path}")

    metrics['saga_iterations'] = saga_epoch
    metrics['fl_test_accuracy'] = float(output_proj['path'][-1]['metrics']['acc_test'])

    return metrics, linear_layer_path
