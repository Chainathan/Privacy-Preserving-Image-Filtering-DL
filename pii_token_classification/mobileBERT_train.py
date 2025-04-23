import os
import numpy as np
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate
import wandb
import datasets

dataset = load_dataset("ai4privacy/pii-masking-200k")
# Split the train dataset into train and validation
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Create a new DatasetDict with the splits
dataset = datasets.DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]  # The "test" split from train_test_split becomes our validation
})

print(f"Train size: {len(dataset['train'])}")
print(f"Validation size: {len(dataset['validation'])}")

# Create label mappings from the dataset
# First, extract all unique labels from the privacy_mask field
print("Extracting unique labels...")
entity_labels = set()
for example in dataset["train"]:
    for mask in example["privacy_mask"]:
        entity_labels.add(mask["label"])

# Create BIO tags for each entity type
label_list = ["O"]  # Outside tag
for entity in sorted(entity_labels):
    label_list.append(f"B-{entity}")  # Beginning tag
    label_list.append(f"I-{entity}")  # Inside tag

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print(f"Number of labels: {len(label_list)}")
print("Labels:", label_list[:10], "..." if len(label_list) > 10 else "")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MobileBERT tokenizer and model
model_name = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# Function to process dataset with MobileBERT tokenizer
def prepare_with_mobilebert_tokenizer(examples, tokenizer=tokenizer, max_length=256, label2id=label2id):
    """
    Tokenize text with MobileBERT's tokenizer and align labels based on character offsets.
    
    This function:
    1. Takes the raw text from source_text
    2. Uses the privacy_mask information to build a character-level entity map
    3. Tokenizes with MobileBERT's tokenizer and gets character offsets
    4. Aligns entity labels with the new tokens
    """
    # Get the original text
    texts = examples["source_text"]
    
    # Extract character-level entity information from privacy_mask
    entity_maps = []
    for example_masks in examples["privacy_mask"]:
        entity_map = {}
        for mask in example_masks:
            for pos in range(mask["start"], mask["end"]):
                entity_map[pos] = mask["label"]
        entity_maps.append(entity_map)
    
    # Tokenize with MobileBERT's tokenizer and get character offsets
    tokenized = tokenizer(
        texts, 
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True
    )
    
    # Align labels with new tokens
    labels = []
    for i, offset_mapping in enumerate(tokenized.pop("offset_mapping")):
        label_ids = []
        entity_map = entity_maps[i]
        
        previous_entity = None
        for j, (start, end) in enumerate(offset_mapping):
            # Skip special tokens which have empty offsets (0,0)
            if start == end == 0:
                label_ids.append(-100)
                continue
                
            # Find if this token overlaps with any entity
            current_entity = None
            for pos in range(start, end):
                if pos in entity_map:
                    current_entity = entity_map[pos]
                    break
            
            # Determine if this is a beginning or inside token
            if current_entity is None:
                # Not an entity
                label_ids.append(label2id["O"])
                previous_entity = None
            elif previous_entity != current_entity:
                # Beginning of entity or new entity
                label_ids.append(label2id[f"B-{current_entity}"])
                previous_entity = current_entity
            else:
                # Continuation of the entity
                label_ids.append(label2id[f"I-{current_entity}"])
        
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized

# Apply tokenization and label alignment to dataset
print("Processing dataset with MobileBERT tokenizer...")
tokenized_datasets = dataset.map(
    prepare_with_mobilebert_tokenizer,
    batched=True,
    batch_size=32,  # Process in smaller batches to avoid memory issues
    remove_columns=dataset["train"].column_names,
    num_proc=4,  # Use multiple processes for faster processing
)

def convert_to_binary_classification(examples):
    labels = examples["labels"]
    binary_labels = []
    
    for example_labels in labels:
        example_binary_labels = []
        for label in example_labels:
            if label == -100:  # Keep special token labels intact
                example_binary_labels.append(-100)
            elif label == label2id["O"]:  # Keep "O" label intact
                example_binary_labels.append(0)  # 0 for non-PII
            else:  # All PII types become a single class
                example_binary_labels.append(1)  # 1 for PII
                
        binary_labels.append(example_binary_labels)
    
    examples["labels"] = binary_labels
    return examples

# Create a binary version of the dataset
binary_id2label = {0: "O", 1: "PII"}
binary_label2id = {"O": 0, "PII": 1}

# Apply the conversion
binary_tokenized_datasets = tokenized_datasets.map(
    convert_to_binary_classification,
    batched=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Update model configuration for binary classification
binary_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=binary_id2label,
    label2id=binary_label2id
)
binary_model.to(device);


seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [binary_id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [binary_id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

import wandb
import os
wandb.login()

os.environ["WANDB_PROJECT"] = "pii_NER"  # Name of your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"       # Options: 'checkpoint', 'end', or 'false'
os.environ["WANDB_WATCH"] = "all"                  # Options: 'gradients', 'all', or 'false'
# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding="longest")

run_name = "mobilebert-pii-binary3"

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./results/{run_name}",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to=["wandb"],
    push_to_hub=False,
    run_name=run_name,
    # gradient_checkpointing=True,
    fp16=True,  # Enable mixed precision training if supported
)

# Initialize Trainer
trainer = Trainer(
    model=binary_model,
    args=training_args,
    train_dataset=binary_tokenized_datasets["train"],
    eval_dataset=binary_tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("Training model...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
evaluation_results = trainer.evaluate()
print(f"Evaluation results: {evaluation_results}")

# Save model
model_save_path = f"./results/{run_name}/final_model"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")