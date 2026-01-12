# coding: utf-8
"""
Bipolar-encoder baseline training script.

Based on paper: "Revisiting the Role of Similarity and Dissimilarity in Best Counter Argument Retrieval" (2023)

Model:
  - Encoder: bert-base-uncased (shared parameters for point/counter)
  - Retrieval head: pooler_output -> Linear(768->128), used for TripletLoss
  - Classification head: [point_emb; counter_emb; |point_emb - counter_emb|] -> Linear(2304->2)

Training:
  - Optimizer: Adam, lr=3e-6
  - Batch size: 4
  - Epochs: 200
  - Negative sampling: random negative sampling (re-sample each epoch)
  - Loss: TripletMarginLoss(margin=1.0, p=2) + 2 * CrossEntropyLoss (joint training)
"""

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from config import training_dir, validation_dir, test_dir
from dataloader import DataLoader as OurDataLoader
from bert.bertdataloader import ArgumentDataSet, trans_to_pairs


class BipolarEncoder(nn.Module):
    """Bipolar-encoder: BERT with retrieval head + classification head."""

    def __init__(self, pretrained_model="bert-base-uncased", retrieval_dim=128):
        super(BipolarEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        # Retrieval head: 768 -> 128 for TripletLoss
        self.retrieval_proj = nn.Linear(768, retrieval_dim)

        # Classification head: [point; counter; |point-counter|] -> 2
        # Input: 768*3 = 2304
        self.classification_head = nn.Linear(768 * 3, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """Encode input and return retrieval embedding + pooler output."""
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        pooler_output = outputs[1]  # [batch, 768]
        retrieval_emb = self.retrieval_proj(pooler_output)  # [batch, 128]
        return retrieval_emb, pooler_output

    def classify_pair(self, point_emb, counter_emb):
        """
        Classification head: [point; counter; |point - counter|] -> logits.

        Args:
            point_emb: [batch, 768]
            counter_emb: [batch, 768]

        Returns:
            logits: [batch, 2]
        """
        diff = torch.abs(point_emb - counter_emb)
        x = torch.cat([point_emb, counter_emb, diff], dim=1)  # [batch, 2304]
        logits = self.classification_head(x)
        return logits

    def get_tokenizer(self):
        return self.tokenizer


class CombinedLoss(nn.Module):
    """Joint loss: TripletLoss + 2 * CrossEntropyLoss."""

    def __init__(self, margin=1.0, p=2):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, anchor, positive, negative, positive_logits, negative_logits):
        """
        Args:
            anchor, positive, negative: [batch, retrieval_dim]
            positive_logits, negative_logits: [batch, 2]

        Returns:
            combined_loss: scalar
        """
        loss_triplet = self.triplet_loss(anchor, positive, negative)
        # Positive pair label = 1
        loss_ce_pos = self.ce_loss(
            positive_logits, torch.ones(positive_logits.size(0), dtype=torch.long, device=positive_logits.device)
        )
        # Negative pair label = 0
        loss_ce_neg = self.ce_loss(
            negative_logits, torch.zeros(negative_logits.size(0), dtype=torch.long, device=negative_logits.device)
        )
        return loss_triplet + loss_ce_pos + loss_ce_neg


def set_seed(seed):
    """Set random seed for reproducibility (optional)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets():
    """Load and transform data into (point, counter, negative) triples."""
    ourdataloader = OurDataLoader(training_dir, validation_dir, test_dir)
    training_df, validation_df, test_df = ourdataloader.to_dataframe()

    training_df = trans_to_pairs(training_df).dropna().reset_index(drop=True)
    validation_df = trans_to_pairs(validation_df).dropna().reset_index(drop=True)
    test_df = trans_to_pairs(test_df).dropna().reset_index(drop=True)

    # Initialize negative_text (will be overwritten during training)
    training_df["negative_text"] = training_df["counter_text"]
    validation_df["negative_text"] = validation_df["counter_text"]
    test_df["negative_text"] = test_df["counter_text"]

    training_dataset = ArgumentDataSet(training_df)
    validation_dataset = ArgumentDataSet(validation_df)
    test_dataset = ArgumentDataSet(test_df)

    return training_df, validation_df, test_df, training_dataset, validation_dataset, test_dataset


def random_negative_sampling(df):
    """
    Random negative sampling: for each sample, randomly pick a counter_text from the dataset.

    Args:
        df: DataFrame with 'counter_text' column.

    Returns:
        df with updated 'negative_text' column.
    """
    n = len(df)
    # Simple random sampling (may pick same as positive, which is acceptable in random sampling)
    neg_indices = np.random.randint(0, n, size=n)
    df["negative_text"] = df["counter_text"].values[neg_indices]
    return df


def evaluate_split(model, tokenizer, dataset, batch_size, max_length, device):
    """
    Simple evaluation: compute accuracy of positive pairs.

    (In real implementation, you may want to compute retrieval metrics like top-k accuracy.)
    Here we just return classification accuracy on positive pairs for simplicity.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for point, counter, negative in dataloader:
            point_tok = tokenizer(point, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            counter_tok = tokenizer(counter, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

            point_ret, point_emb = model(
                input_ids=point_tok["input_ids"].to(device),
                token_type_ids=point_tok["token_type_ids"].to(device),
                attention_mask=point_tok["attention_mask"].to(device),
            )
            counter_ret, counter_emb = model(
                input_ids=counter_tok["input_ids"].to(device),
                token_type_ids=counter_tok["token_type_ids"].to(device),
                attention_mask=counter_tok["attention_mask"].to(device),
            )

            logits = model.classify_pair(point_emb, counter_emb)
            preds = torch.argmax(logits, dim=1)
            labels = torch.ones(logits.size(0), dtype=torch.long, device=device)
            correct += (preds == labels).sum().item()
            total += logits.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def train(args):
    """Main training loop for Bipolar-encoder."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optional: set seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # Build datasets
    print("Loading datasets...")
    training_df, validation_df, test_df, training_dataset, validation_dataset, test_dataset = build_datasets()
    print(f"Train: {len(training_df)}, Val: {len(validation_df)}, Test: {len(test_df)}")

    # Build model
    print("Building model...")
    model = BipolarEncoder(pretrained_model=args.pretrained_model, retrieval_dim=args.retrieval_dim)
    tokenizer = model.get_tokenizer()
    model = model.to(device)

    # Multi-GPU support (optional)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(margin=args.triplet_margin, p=args.triplet_p)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()

        # Re-sample negative examples at the beginning of each epoch
        training_df = random_negative_sampling(training_df)

        train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        epoch_loss = 0.0
        for batch_idx, (point, counter, negative) in enumerate(train_dataloader):
            # Tokenize
            point_tok = tokenizer(point, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
            counter_tok = tokenizer(counter, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
            negative_tok = tokenizer(negative, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")

            # Forward: point
            point_ret, point_emb = model(
                input_ids=point_tok["input_ids"].to(device),
                token_type_ids=point_tok["token_type_ids"].to(device),
                attention_mask=point_tok["attention_mask"].to(device),
            )
            # Forward: counter (positive)
            counter_ret, counter_emb = model(
                input_ids=counter_tok["input_ids"].to(device),
                token_type_ids=counter_tok["token_type_ids"].to(device),
                attention_mask=counter_tok["attention_mask"].to(device),
            )
            # Forward: negative
            negative_ret, negative_emb = model(
                input_ids=negative_tok["input_ids"].to(device),
                token_type_ids=negative_tok["token_type_ids"].to(device),
                attention_mask=negative_tok["attention_mask"].to(device),
            )

            # Classification logits
            if isinstance(model, nn.DataParallel):
                positive_logits = model.module.classify_pair(point_emb, counter_emb)
                negative_logits = model.module.classify_pair(point_emb, negative_emb)
            else:
                positive_logits = model.classify_pair(point_emb, counter_emb)
                negative_logits = model.classify_pair(point_emb, negative_emb)

            # Compute loss
            loss = criterion(point_ret, counter_ret, negative_ret, positive_logits, negative_logits)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            print("Evaluating...")
            train_acc = evaluate_split(model, tokenizer, training_dataset, args.batch_size, args.max_length, device)
            val_acc = evaluate_split(model, tokenizer, validation_dataset, args.batch_size, args.max_length, device)
            test_acc = evaluate_split(model, tokenizer, test_dataset, args.batch_size, args.max_length, device)
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Save checkpoint
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"bipolar_epoch_{epoch+1}.pt")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training completed.")


def main():
    parser = argparse.ArgumentParser(description="Train Bipolar-encoder baseline")

    # Model config
    parser.add_argument("--pretrained-model", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
    parser.add_argument("--retrieval-dim", type=int, default=128, help="Dimension of retrieval projection")

    # Training config
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length for tokenizer")

    # Loss config
    parser.add_argument("--triplet-margin", type=float, default=1.0, help="Margin for TripletMarginLoss")
    parser.add_argument("--triplet-p", type=int, default=2, help="p-norm for TripletMarginLoss")

    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints (optional)")

    # Random seed
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
