import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# base = https://torch.classcat.com/2022/03/20/pytorch-ignite-0-4-8-notebooks-text-cnn/
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, d_prob, num_classes) -> None:
        super(TextCNN, self).__init__()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x.T).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()