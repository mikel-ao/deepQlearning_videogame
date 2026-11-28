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


class DuelingDQN(nn.Module):
    """Dueling DQN (Wang et al., 2016): separa cuánto vale el estado (V) de cuánto mejor es cada acción (A).
    Q(s, a) = V(s) + A(s, a) - media_a A(s, a)."""

    def __init__(self, state_dim=128, action_dim=6, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.adv = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.adv(x)
        return self.value(x) + a - a.mean(dim=1, keepdim=True)


def build_dqn(state_dim, action_dim, hidden_dim=256, dueling=False):
    """La red que corresponde a la configuración (dueling: true en el YAML -> DuelingDQN)."""
    return (DuelingDQN if dueling else DQN)(state_dim, action_dim, hidden_dim)


if __name__ == "__main__":
    net = DQN()
    q = net(torch.randint(0, 256, (10, 128), dtype=torch.uint8))
    print("Salida:", q.shape)          # torch.Size([10, 6])
