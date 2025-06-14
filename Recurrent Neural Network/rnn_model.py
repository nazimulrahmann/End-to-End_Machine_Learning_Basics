import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive attention mechanism for sequence data.
    Supports multiple attention types: 'additive' or 'dot'.
    """
    def __init__(self, hidden_size, attention_type='additive'):
        super(Attention, self).__init__()
        self.attention_type = attention_type

        if attention_type == 'additive':
            self.attn = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, rnn_output):
        # rnn_output: (batch, seq_len, hidden*2)
        if self.attention_type == 'additive':
            energy = self.attn(rnn_output)  # (batch, seq_len, 1)
            weights = F.softmax(energy, dim=1)
        elif self.attention_type == 'dot':
            weights = F.softmax(torch.sum(rnn_output, dim=2, keepdim=True), dim=1)
        else:
            raise ValueError("Unsupported attention type")

        context = torch.sum(weights * rnn_output, dim=1)  # (batch, hidden*2)
        return context, weights


class RecurrentNN(nn.Module):
    """
    Flexible and reusable RNN with attention and classifier.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        rnn_type='rnn',  # 'rnn', 'lstm', 'gru'
        bidirectional=True,
        dropout=0.3,
        attention_type='additive'
    ):
        super(RecurrentNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        rnn_cls = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }.get(self.rnn_type)

        if rnn_cls is None:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = Attention(hidden_size, attention_type=attention_type)

        final_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(final_dim // 2, num_classes)
        )

    def forward(self, x):
        # Expected x shape: (batch, channels, height, width)
        x = x.squeeze(1)  # remove channel if present: (B, H, W)
        x = x.permute(0, 2, 1)  # (B, seq_len, features)

        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_size,
            device=x.device
        )

        if self.rnn_type == 'lstm':
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        context, _ = self.attention(out)
        return self.classifier(context)



# Usage of Recurrent Neural Network
model = RecurrentNN(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    num_classes=10,
    rnn_type='lstm',
    attention_type='additive',
    bidirectional=True,
    dropout=0.4
)