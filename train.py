
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

def get_labels(data):
    labels = set()
    for item in data:
        for label in item['labels']:
            labels.add(label)
    return sorted(list(labels))

def main():
    # Load data
    with open('data/combined_final_tagged.json', 'r') as f:
        data = json.load(f)

    # Validate data to ensure token and label lengths match
    validated_data = []
    for i, item in enumerate(data):
        if 'tokens' not in item or 'labels' not in item or len(item['tokens']) != len(item['labels']):
            print(f"Warning: Mismatch or malformed data in sample index {i}. Skipping.")
            continue
        validated_data.append(item)
    
    print(f"Original data size: {len(data)}, Validated data size: {len(validated_data)}")
    data = validated_data

    # Get labels and create mappings
    labels = get_labels(data)
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

    # Preprocess data
    class NERDataset(Dataset):
        def __init__(self, data, tokenizer, label_to_id):
            self.data = data
            self.tokenizer = tokenizer
            self.label_to_id = label_to_id

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            original_tokens = item['tokens']
            original_labels = item['labels']

            # Reconstruct sentence and tokenize with KoBERT to ensure consistency
            sentence = " ".join(original_tokens)
            tokenized_inputs = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=128)

            word_ids = tokenized_inputs.word_ids()

            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get a -100 label so they are ignored by the loss function
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # If it's the first token of a new word, assign the label
                    label_ids.append(self.label_to_id[original_labels[word_idx]])
                else:
                    # For subsequent tokens of the same word, also assign -100
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            tokenized_inputs["labels"] = label_ids
            return {key: torch.tensor(val) for key, val in tokenized_inputs.items()}

    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = NERDataset(train_data, tokenizer, label_to_id)
    val_dataset = NERDataset(val_data, tokenizer, label_to_id)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained("skt/kobert-base-v1", num_labels=num_labels)
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    # Training arguments
    # NOTE: Using older arguments for compatibility. Evaluating and saving every epoch.
    # Estimated steps per epoch = (Total samples * 0.8) / batch_size approx. 215
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        do_eval=True,
        eval_steps=215,
        save_steps=215,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model("./kobert_ner_model")
    tokenizer.save_pretrained("./kobert_ner_model")
    
    print("Training complete. Model saved to ./kobert_ner_model")

if __name__ == "__main__":
    main()
