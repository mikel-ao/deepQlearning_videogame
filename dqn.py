import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Red de política: RAM de Demon Attack (128 uint8) -> un Q-valor por acción (6)."""

    def __init__(self, state_dim=128, action_dim=6, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.float() / 255.0          # RAM uint8 [0, 255] -> [0, 1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)             # sin activación: los Q-valores pueden ser cualquier real


if __name__ == "__main__":
    net = DQN()
    q = net(torch.randint(0, 256, (10, 128), dtype=torch.uint8))
    print("Salida:", q.shape)          # torch.Size([10, 6])
