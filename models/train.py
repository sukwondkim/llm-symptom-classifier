import argparse

import pandas as pd
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from utils.preprocessing import categories, hpo_data
from models.hpo_dataset import HPODataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="outputs")
    parser.add_argument("--output_dir", type=str, default=outputs)
    parser.add_argument("--max_token_len", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--base_model", type=str, default="microsoft/biogpt")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj")
    parser.add_argument("--lora_task_type", type=str, default="SEQ_CLS")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(categories)
    )
    peft_config = LoraConfig(
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.lora_target_modules.split(","),
        "task_type": args.lora_task_type,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # Load data
    category_name2idx = {val: idx for idx, val in enumerate(categories)}
    df = pd.read_csv(args.train_file, sep="\t", header=None, names=["hpo_ids", "category"])
    def preprocess_hpo_ids(hpo_ids_str):
        hpo_ids = [h.strip() for h in hpo_ids_str.split(",") if h.strip() in hpo_data]
        if not hpo_ids:
            raise ValueError(f"No valid HPO IDs in {hpo_ids}")
        
        return sorted(hpo_ids)

    df["hpo_ids_list"] = df["hpo_ids"].apply(preprocess_hpo_ids)
    df["label"] = df["category"].map(category_name2idx)

    train_hpo_ids, val_hpo_ids, train_labels, val_labels = train_test_split(
        df["hpo_ids_list"].tolist(),
        df["label"].tolist(),
        stratify=df["label"].tolist(),
        test_size=args.validation_split,
    )
    train_dataset = HPODataset(
        train_hpo_ids,
        train_labels,
        tokenizer,
        args.max_token_len
    )
    val_dataset = HPODataset(
        val_hpo_ids,
        val_labels,
        tokenizer,
        args.max_token_len
    )

    # Set up methods for training
    def compute_accuracy(p: EvalPrediction):
        return {
            "accuracy": accuracy_score(
                p.label_ids, p.predictions.argmax(axis=-1)
            )
        }

    def use_only_logit(logits, _labels):
        return logits[0]

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=1,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
        preprocess_logits_for_metrics=use_only_logit,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
        )]
    )
    if args.checkpoint_path:
        trainer.train(resume_from_checkpoint=args.checkpoint_path)
    else:
        trainer.train()
    print("Training complete!")
    
    trainer.model.save_state()
    trainer.model.save_model(args.output_dir)
    trainer.tokenizer.save_tokenizer(args.output_dir)
    print(f"Trained model saved on {args.output_dir}")
