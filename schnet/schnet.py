import argparse
import os.path as osp
import math
import copy

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple

import torch_geometric
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

OptTensor = Optional[Tensor]
PI = math.pi


# --- 1. EMA (指数移动平均) 实现 ---
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# --- 2. SchNet 模型定义 ---
class SchNet(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int = 64,
            num_filters: int = 64,
            num_interactions: int = 6,
            num_gaussians: int = 50,
            cutoff: float = 10.0,
            interaction_graph: Optional[Callable] = None,
            max_num_neighbors: int = 32,
            readout: str = 'add',
            dipole: bool = False,
            mean: Optional[float] = None,
            std: Optional[float] = None,
            atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase.data
            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out.float(), dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out


class RadiusInteractionGraph(torch.nn.Module):
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_filters: int,
            nn: Sequential,
            cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


# --- 3. 训练脚本设置 ---

parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args(args=[])

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9') if '__file__' in locals() else './data/QM9'

# Target 7 is U0 (Energy)
target = 7
print(f"Training on Target {target} (Expectation for U0: ~0.31 kcal/mol)")

# 1. 加载原始数据集 (暂时不应用 transform)
dataset = QM9(path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === 修正点 1: 手动计算残差 (Total Energy - Atomrefs) 并覆盖 dataset ===
print("Preprocessing: Subtracting atomrefs and standardizing...")
atom_refs = dataset.atomref(target)
z = dataset.data.z
slices = dataset.slices['z']
y_total = dataset.data.y[:, target]

# 计算每个分子的 atomref 之和
shifts = torch.zeros(len(dataset), device=y_total.device)
# 遍历计算 shifts (这比 vectorization 更安全易读，QM9 13万数据约需几秒)
for i in range(len(dataset)):
    atoms_idx = z[slices[i]:slices[i + 1]]
    shifts[i] = atom_refs[atoms_idx].sum()

# 计算残差能量 (Interaction Energy)
y_residual = y_total - shifts

# === 修正点 2: 对残差进行标准化 ===
mean = y_residual.mean()
std = y_residual.std()
dataset.data.y[:, target] = (y_residual - mean) / std

# 记录 mean 和 std 供后续还原使用
mean_val = mean.item()
std_val = std.item()
print(f"Residual Mean: {mean_val:.4f}, Residual Std: {std_val:.4f}")


# 定义 transform 只选取目标列
class MyTransform:
    def __call__(self, data):
        data.y = data.y[:, target]
        return data


# 应用 transform 并打乱
dataset.transform = MyTransform()
dataset = dataset.shuffle()

# === 修正点 3: 初始化模型时 atomref=None ===
# 因为我们已经手动减去了 atomref，模型只需要学习标准化后的残差
model = SchNet(
    hidden_channels=64,
    num_filters=64,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,
    dipole=False,
    atomref=None,  # 【关键】这里必须为 None，防止重复加和
    mean=None,  # 【关键】不传均值
    std=None  # 【关键】不传标准差
)

# 数据集划分 (论文标准: 110k Train)
N_train = 110000
N_val = 10000
train_dataset = dataset[:N_train]
val_dataset = dataset[N_train:N_train + N_val]
test_dataset = dataset[N_train + N_val:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = model.to(device)

# 优化器设置
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
steps_per_epoch = len(train_loader)
decay_steps = 100000
decay_rate = 0.96
lr_lambda = lambda step: decay_rate ** (step // decay_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# EMA 初始化
ema = EMA(model, decay=0.99)
ema.register()


def train(epoch, global_step):
    model.train()
    loss_all = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        data = data.to(device)
        optimizer.zero_grad()

        # 模型输出的是标准化的残差
        pred = model(data.z, data.pos, data.batch)
        # 标签也是标准化的残差
        loss = F.mse_loss(pred, data.y.view(-1, 1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        ema.update()
        scheduler.step()
        global_step += 1

        loss_all += loss.item() * data.num_graphs

    return loss_all / len(train_loader.dataset), global_step


def test(loader, desc="Testing", use_ema=False):
    if use_ema:
        ema.apply_shadow()

    model.eval()
    mae = []

    for data in tqdm(loader, desc=desc):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
            # === 修正点 4: MAE 计算反归一化 ===
            # Pred (标准残差) * Std = 物理单位残差
            # Target (标准残差) * Std = 物理单位残差
            # 误差 = |Pred - Target| * Std
            error = (pred - data.y.view(-1, 1)).abs() * std_val
            mae.append(error)

    if use_ema:
        ema.restore()

    return torch.cat(mae, dim=0)


best_val_error = float('inf')
global_step = 0

for epoch in range(1, 501):
    lr = optimizer.param_groups[0]['lr']
    loss, global_step = train(epoch, global_step)

    val_mae = test(val_loader, desc="Validation", use_ema=True)
    val_error = val_mae.mean()

    if val_error <= best_val_error:
        test_mae = test(test_loader, desc="Test Set", use_ema=True)
        best_val_error = val_error
        torch.save(model.state_dict(), 'schnet_qm9_best_ema.pt')
        print(f"Saved best model (EMA). Test MAE: {test_mae.mean():.4f}")

    print(f'Epoch: {epoch:03d}, LR: {lr:.6f}, Loss: {loss:.6f}, '
          f'Val MAE: {val_error:.4f}, Test MAE: {test_mae.mean():.4f}')