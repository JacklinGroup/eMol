import torch
import torch.nn as nn

from emol_core.model.geometry import act_class_mapping


class GatedEquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super().__init__()
        intermediate_channels = intermediate_channels or hidden_channels
        self.out_channels = out_channels
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class_mapping[activation](),
            nn.Linear(intermediate_channels, out_channels * 2),
        )
        self.act = act_class_mapping[activation]() if scalar_activation else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, vec):
        vec1 = torch.norm(self.vec1_proj(vec), dim=-2)
        vec2 = self.vec2_proj(vec)
        x, gate = torch.split(
            self.update_net(torch.cat([x, vec1], dim=-1)),
            self.out_channels,
            dim=-1,
        )
        vec = gate.unsqueeze(1) * vec2
        if self.act is not None:
            x = self.act(x)
        return x, vec


class EquivariantScalar(nn.Module):
    allow_prior_model = False

    def __init__(self, hidden_channels, activation="silu"):
        super().__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    activation=activation,
                ),
            ]
        )

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, vec, z, pos, batch):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return x + vec.sum() * 0

    @staticmethod
    def post_reduce(x):
        return x
