from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get LLM-generated and human text
        llm_text = self.dataset["LLM"][idx]
        human_text = self.dataset["human"][idx]

        # Tokenize inputs and targets
        input_encoding = self.tokenizer(llm_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        target_encoding = self.tokenizer(human_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # T5 expects padding tokens (-100) instead of actual padding token ID
        labels[labels == self.tokenizer.pad_token_id] = -100  

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "human_text": human_text,
            "LLM_text": human_text,
        }
