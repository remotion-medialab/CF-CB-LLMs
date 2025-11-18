"""
Modularized evaluation for W&B integration.
Refactored from test_CBLLM.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model
import evaluate
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, eos_pooling
from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate_model(
    cbl_path,
    batch_size=256,
    max_length=512,
    num_workers=0,
    dropout=0.1,
    sparse=False,
):
    """
    Evaluate trained CB-LLM model.

    Args:
        cbl_path: Path to trained CBL checkpoint
        batch_size: Batch size for evaluation
        max_length: Max sequence length
        num_workers: Number of workers
        dropout: Dropout rate
        sparse: Whether to use sparse final layer

    Returns:
        dict: Evaluation metrics including accuracy, confusion matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Evaluating CB-LLM model...")

    # Parse CBL path to get configuration
    path_parts = cbl_path.split("/")
    acs = path_parts[0]
    dataset = path_parts[1]
    if 'sst2' in dataset:
        dataset = dataset.replace('_', '/')
    backbone = path_parts[2]
    cbl_name = path_parts[-1]

    print(f"Dataset: {dataset}, Backbone: {backbone}, ACS: {acs}")
    print(f"Sparse: {sparse}")

    # Load test dataset
    print("Loading test data...")
    test_dataset = load_dataset(dataset, split='test')
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

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True,
                           max_length=max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    # Create dataloader
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __getitem__(self, idx):
            return {key: torch.tensor(values[idx]) for key, values in self.texts.items()}

        def __len__(self):
            return len(self.texts['input_ids'])

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

    # Load linear layer
    linear_layer_path = os.path.join(os.path.dirname(cbl_path), "linear_layer.pt")
    if not os.path.exists(linear_layer_path):
        raise FileNotFoundError(f"Linear layer not found at {linear_layer_path}. "
                              "Please train the final layer first.")

    print("Loading linear layer...")
    linear_layer = torch.load(linear_layer_path, map_location=device)

    # Apply sparsity if requested
    if sparse:
        print("Applying sparsity to linear layer...")
        with torch.no_grad():
            # Get top k most important weights per class
            weights = linear_layer['weight']
            k = max(1, int(0.1 * weights.shape[1]))  # Keep top 10%

            # Zero out small weights
            for i in range(weights.shape[0]):
                threshold = torch.topk(torch.abs(weights[i]), k).values[-1]
                mask = torch.abs(weights[i]) < threshold
                weights[i][mask] = 0.0

            linear_layer['weight'] = weights

    # Extract concept features and make predictions
    print("Extracting features and making predictions...")
    all_preds = []
    all_labels = []
    all_concept_features = []

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

            # Apply linear layer
            logits = F.linear(concept_features, linear_layer['weight'], linear_layer.get('bias'))
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_concept_features.append(concept_features.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_concept_features = torch.cat(all_concept_features, dim=0)
    all_labels = np.array(encoded_test_dataset['label'])

    # Compute metrics
    print("Computing metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    n_classes = CFG.class_num[dataset]
    per_class_acc = []
    for i in range(n_classes):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            per_class_acc.append(float(class_acc))
        else:
            per_class_acc.append(0.0)

    # Concept activation statistics
    concept_mean = float(all_concept_features.mean())
    concept_std = float(all_concept_features.std())

    # Count active concepts (non-zero after ReLU-like activation)
    active_concepts = (all_concept_features > 0).float().mean().item()

    # Count non-zero weights in linear layer
    num_active_weights = (linear_layer['weight'] != 0).sum().item()
    total_weights = linear_layer['weight'].numel()
    sparsity = 1.0 - (num_active_weights / total_weights)

    metrics = {
        'test_accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'concept_features_mean': concept_mean,
        'concept_features_std': concept_std,
        'active_concepts_ratio': active_concepts,
        'linear_layer_sparsity': float(sparsity),
        'num_active_weights': num_active_weights,
        'total_weights': total_weights,
    }

    # Add per-class accuracy
    for i, acc in enumerate(per_class_acc):
        metrics[f'test_accuracy_class_{i}'] = acc

    print(f"Concept features mean: {concept_mean:.4f}, std: {concept_std:.4f}")
    print(f"Active concepts ratio: {active_concepts:.4f}")
    print(f"Linear layer sparsity: {sparsity:.4f}")

    return metrics
