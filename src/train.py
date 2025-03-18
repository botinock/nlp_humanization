import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from evaluation import evaluate_model  # Import evaluation function
from dataset import TextDataset
from datasets import Dataset as HFDataset
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Load dataset (replace with your own)
df = pd.read_csv('data/dataset.csv')
dataset = HFDataset.from_pandas(df[['human', 'LLM']])

tokenizer = T5Tokenizer.from_pretrained("t5-small")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# Freeze all layers except the final ones
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
for param in model.encoder.parameters():
    param.requires_grad = False
# TensorBoard writer (for logging metrics)
writer = SummaryWriter(log_dir="runs/evaluation_t5_1")
# Create train and eval datasets
train_size = int(0.9 * len(dataset))
train_dataset = TextDataset(dataset.select(range(train_size)), tokenizer)
eval_dataset = TextDataset(dataset.select(range(train_size, len(dataset))), tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

    for batch in progress_bar:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items() if k not in ("human_text", "LLM_text")}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}: Average Training Loss = {avg_train_loss:.4f}")

    # Log training loss
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)

    print(f"Epoch {epoch} - Training Loss: {loss.item():.4f}")

    # Evaluate model
    evaluate_model(model, eval_loader, tokenizer, device, epoch, writer)
