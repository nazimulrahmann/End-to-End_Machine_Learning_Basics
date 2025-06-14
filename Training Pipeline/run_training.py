from torch.utils.data import DataLoader, TensorDataset
import torch

# Prepare Data
# Example: Random data (replace with your actual data)
# For classification (10 classes)
X_train = torch.randn(1000, 3, 32, 32)  # 1000 samples, 3 channels, 32x32
y_train = torch.randint(0, 10, (1000,))  # 1000 labels (0-9)

X_val = torch.randn(200, 3, 32, 32)
y_val = torch.randint(0, 10, (200,))

X_test = torch.randn(200, 3, 32, 32)
y_test = torch.randint(0, 10, (200,))

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Defining the model
# model = ANN, CNN, RNN, LSTM, GRU, Transformer, Diffusion Models etc.
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    task_type='classification',  # or 'regression' for regression tasks
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
# Run Training
history = trainer.train(epochs=10)
# Evaluate Model
results = trainer.evaluate()  # Evaluates on test_loader

 """
   Example Usage 
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


model = BertClassifier(num_classes=2)
trainer = Trainer(model, train_loader, val_loader, test_loader, task_type='classification')
history = trainer.train(epochs=5)
"""