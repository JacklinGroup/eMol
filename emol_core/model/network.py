"""Electron-aware equivariant molecular representation network."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from emol_core.model.geometry import (
    CosineCutoff,
    Sphere,
    VecLayerNorm,
    rbf_class_mapping
)
from emol_core.model.sampling import batch_farthest_point_sample
from emol_core.model.density_features import SampledElectronEmbedding
from emol_core.model.graph import HybridGraphBuilder
from emol_core.model.interaction import HybridMessagePassing


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # 移除自环
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        cutoff_weight = self.cutoff(edge_weight)
        weight = self.distance_proj(edge_attr) * cutoff_weight.view(-1, 1)

        neighbor_x = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        neighbor_x = self.propagate(edge_index, x=neighbor_x, W=weight, size=None)
        neighbor_x = self.combine(torch.cat([x, neighbor_x], dim=1))
        return neighbor_x

    def message(self, x_j, W):
        return x_j * W


class EMolRepresentation(nn.Module):
    """Represent atoms and sampled electron-density tokens on a hybrid graph."""

    def __init__(
        self,
        lmax=2,
        vecnorm_type='none',
        trainable_vecnorm=False,
        num_heads=8,
        num_layers=9,
        hidden_channels=256,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=100,
        cutoff=5.0,
        max_num_neighbors=32,
        ee_cutoff=None,
        ee_max_num_neighbors=None,
        vertex_type=None,  # 兼容性参数，不使用
        # Electron density parameters
        electron_radius=2.0,
        learnable_radius=True,
        # Sampling parameters
        num_sample_points=3,  # 每个原子采样的点数
        atom_token_extra_neighbors=1,
        # Electron gate parameters
        electron_gate=0.25,
        electron_gate_mode="fixed",  # fixed | schedule | trainable
        aeea_five_body_scale=0.0,
        aeea_five_body_use_gate=True,
        ee_scalar_scale=0.1,
        ee_vector_scale=0.05,
    ):
        super().__init__()
        self.lmax = lmax
        self.vecnorm_type = vecnorm_type
        self.trainable_vecnorm = trainable_vecnorm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.max_z = max_z
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.ee_cutoff = cutoff if ee_cutoff is None else ee_cutoff
        self.ee_max_num_neighbors = max_num_neighbors if ee_max_num_neighbors is None else ee_max_num_neighbors
        self.electron_radius = electron_radius
        self.learnable_radius = learnable_radius
        self.num_sample_points = num_sample_points
        self.atom_token_extra_neighbors = int(atom_token_extra_neighbors)
        # 电子分支梯度门控
        self.electron_gate = float(electron_gate)
        self.electron_gate_mode = electron_gate_mode
        self.aeea_five_body_scale = float(aeea_five_body_scale)
        self.aeea_five_body_use_gate = bool(aeea_five_body_use_gate)
        if self.electron_gate_mode == "trainable":
            init_gate = torch.tensor(self._clamp_gate(self.electron_gate))
            self.electron_gate_param = nn.Parameter(torch.logit(init_gate))
        else:
            self.electron_gate_param = None
        self.electron_detach = False

        # === 原子嵌入 (带电子特征增强) ===
        # self.atom_embedding = ElectronEnhancedEmbedding(
        #     max_z=max_z,
        #     hidden_channels=hidden_channels,
        #     initial_radius=electron_radius,
        #     learnable_radius=learnable_radius
        # )

        self.atom_embedding = nn.Embedding(max_z, hidden_channels)

        # 将增强嵌入投影回hidden_channels
        # self.atom_embed_projection = nn.Linear(self.atom_embedding.output_dim, hidden_channels)

        # === 电子(采样点)嵌入 ===
        self.electron_embedding = SampledElectronEmbedding(
            hidden_channels=hidden_channels,
            initial_radius=electron_radius,
            learnable_radius=learnable_radius
        )
        self.electron_confidence_floor = 0.1
        self.electron_vec_init_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_owner_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2 + num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.electron_confidence_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2 + num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        # === 球谐函数 ===
        self.sphere = Sphere(l=lmax)

        # === 径向基函数 ===
        self.distance_expansion = rbf_class_mapping[rbf_type](cutoff, num_rbf, trainable_rbf)

        # === 混合图构建器 ===
        self.graph_builder = HybridGraphBuilder(
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            ee_cutoff=self.ee_cutoff,
            ee_max_num_neighbors=self.ee_max_num_neighbors,
        )

        # === 原子邻居嵌入 ===
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            cutoff=cutoff,
            max_z=max_z
        )

        # === 边嵌入 ===
        # 为不同类型的边创建边嵌入层
        self.aa_edge_embedding = nn.Linear(num_rbf, hidden_channels)
        self.ae_edge_embedding = nn.Linear(num_rbf, hidden_channels)
        self.ea_edge_embedding = nn.Linear(num_rbf, hidden_channels)
        self.ee_edge_embedding = nn.Linear(num_rbf, hidden_channels)

        # === 混合消息传递层 ===
        self.hybrid_mp_layers = nn.ModuleList()
        mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            activation=activation,
            attn_activation=attn_activation,
            cutoff=cutoff,
            ee_cutoff=self.ee_cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            aeea_five_body_scale=self.aeea_five_body_scale,
            aeea_five_body_use_gate=self.aeea_five_body_use_gate,
            ee_scalar_scale=ee_scalar_scale,
            ee_vector_scale=ee_vector_scale,
        )

        for _ in range(num_layers - 1):
            layer = HybridMessagePassing(last_layer=False, **mp_kwargs)
            self.hybrid_mp_layers.append(layer)
        self.hybrid_mp_layers.append(HybridMessagePassing(last_layer=True, **mp_kwargs))

        # === 输出归一化 ===
        self.atom_out_norm = nn.LayerNorm(hidden_channels)
        self.atom_vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        self.reset_parameters()

    def reset_parameters(self):

        self.electron_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.electron_vec_init_proj.weight)
        self.electron_vec_init_proj.bias.data.fill_(0)
        for module in self.electron_owner_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
        for module in self.electron_confidence_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)

        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()

        nn.init.xavier_uniform_(self.aa_edge_embedding.weight)
        self.aa_edge_embedding.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ae_edge_embedding.weight)
        self.ae_edge_embedding.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ea_edge_embedding.weight)
        self.ea_edge_embedding.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ee_edge_embedding.weight)
        self.ee_edge_embedding.bias.data.fill_(0)

        for layer in self.hybrid_mp_layers:
            layer.reset_parameters()

        self.atom_out_norm.reset_parameters()
        self.atom_vec_out_norm.reset_parameters()

    def set_electron_gate(self, gate: float, detach: bool = False):
        """设置电子分支的权重与是否阻断梯度"""
        self.electron_gate = float(gate)
        self.electron_detach = bool(detach)
        if self.electron_gate_mode == "trainable" and self.electron_gate_param is not None:
            self.electron_gate_param.data = torch.logit(
                torch.tensor(self._clamp_gate(self.electron_gate), device=self.electron_gate_param.device)
            )

    @staticmethod
    def _clamp_gate(gate: float) -> float:
        return float(min(max(gate, 1e-6), 1 - 1e-6))

    def _get_electron_gate(self, device, dtype) -> Tensor:
        if self.electron_gate_mode == "trainable" and self.electron_gate_param is not None:
            gate = torch.sigmoid(self.electron_gate_param)
            return gate.to(device=device, dtype=dtype)
        return torch.tensor(self.electron_gate, device=device, dtype=dtype)

    def get_electron_gate_value(self) -> float:
        if self.electron_gate_mode == "trainable" and self.electron_gate_param is not None:
            return float(torch.sigmoid(self.electron_gate_param).detach().cpu().item())
        return float(self.electron_gate)

    def _compute_token_confidence(self, owner_context: Tensor) -> Tensor:
        confidence = torch.sigmoid(self.electron_confidence_projection(owner_context)).squeeze(-1)
        floor = float(self.electron_confidence_floor)
        return floor + (1.0 - floor) * confidence

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        前向传播。

        Args:
            data: PyG Data对象,包含:
                - 标准分子数据: z, pos, batch
                - 电子数据: elec_coords, density, num_electrons

        Returns:
            x: 原子节点特征 [N_atoms, hidden_channels]
            vec: 原子向量特征 [N_atoms, (lmax+1)^2-1, hidden_channels]
        """
        z, pos, batch = data.z, data.pos, data.batch
        elec_coords, density = data.elec_coords, data.density

        # 处理电子的batch索引
        if not hasattr(data, 'elec_batch'):
            elec_batch_list = []
            for i, num in enumerate(data.num_electrons):
                elec_batch_list.append(
                    torch.full((num,), i, dtype=torch.long, device=elec_coords.device)
                )
            elec_batch = torch.cat(elec_batch_list) if elec_batch_list else torch.empty(0, dtype=torch.long, device=elec_coords.device)
        else:
            elec_batch = data.elec_batch

        # === 1. 原子嵌入 (带电子特征增强) ===
        # [N_atoms, hidden_channels]
        x_atom = self.atom_embedding(z)
        # 投影到hidden_channels
        # [N_atoms, hidden_channels]
        # x_atom = self.atom_embed_projection(enhanced_atom_x)

        # 初始化原子向量特征
        vec_atom = torch.zeros(x_atom.size(0), ((self.lmax + 1) ** 2) - 1,
                               x_atom.size(1), device=x_atom.device)

        # === 2. 采样电子云中的代表性点 ===
        # [N_atoms*num_sample_points, 3], [N_atoms*num_sample_points]
        sampling_radius = self.electron_radius
        sampled_pos, sampled_batch = batch_farthest_point_sample(
            pos, batch, elec_coords, elec_batch, self.num_sample_points, sampling_radius
        )
        sample_owner = torch.arange(pos.size(0), device=pos.device).repeat_interleave(self.num_sample_points)
        # print("Sampled points:", sampled_pos)

        # === 3. 为采样点计算电子嵌入 ===
        # [N_sampled, hidden_channels]
        x_electron = self.electron_embedding(
            sampled_pos,
            sampled_batch,
            elec_coords,
            density,
            elec_batch,
        )
        owner_pos = pos[sample_owner]
        owner_x = x_atom[sample_owner]
        owner_dist = torch.norm(sampled_pos - owner_pos, dim=-1).clamp(min=1e-4)
        owner_rbf = torch.clamp(torch.nan_to_num(self.distance_expansion(owner_dist)), -5.0, 5.0)
        owner_context = torch.cat([x_electron, owner_x, owner_rbf], dim=1)
        x_electron = x_electron + self.electron_owner_projection(owner_context)
        token_confidence = self._compute_token_confidence(owner_context)
        owner_dir = (sampled_pos - owner_pos) / owner_dist.unsqueeze(1)
        owner_dir_sph = self.sphere(owner_dir)
        vec_amp = torch.tanh(self.electron_vec_init_proj(x_electron))
        vec_electron = owner_dir_sph.unsqueeze(-1) * vec_amp.unsqueeze(1)
        # 梯度门控：可放大/缩小/阻断电子分支
        gate = self._get_electron_gate(device=x_electron.device, dtype=x_electron.dtype)
        token_gate = gate * token_confidence.to(device=x_electron.device, dtype=x_electron.dtype)
        x_electron = x_electron * token_gate.unsqueeze(-1)
        vec_electron = vec_electron * token_gate.view(-1, 1, 1)


        # === 4. 构建混合图 ===
        graph_dict = self.graph_builder(
            pos,
            batch,
            sampled_pos,
            sampled_batch,
            overlap_only=False,
            include_ee=False,
            electron_owner=sample_owner,
            extra_owner_neighbors=self.atom_token_extra_neighbors,
        )

        # === 5. 计算边特征 ===
        # 解包边向量并归一化
        aa_edge_index, aa_edge_weight, aa_edge_vec = graph_dict['aa']
        ae_edge_index, ae_edge_weight, ae_edge_vec = graph_dict['ae']
        ea_edge_index, ea_edge_weight, ea_edge_vec = graph_dict['ea']
        ee_edge_index, ee_edge_weight, ee_edge_vec = graph_dict['ee']

        #print(aa_edge_index.shape, ae_edge_index.shape, ea_edge_index.shape, ee_edge_index.shape)

        # 边长度下限，避免严格为 0 导致极端 RBF
        aa_edge_weight = aa_edge_weight.clamp(min=1e-4)
        ae_edge_weight = ae_edge_weight.clamp(min=1e-4) if ae_edge_weight.numel() > 0 else ae_edge_weight
        ea_edge_weight = ea_edge_weight.clamp(min=1e-4) if ea_edge_weight.numel() > 0 else ea_edge_weight
        ee_edge_weight = ee_edge_weight.clamp(min=1e-4) if ee_edge_weight.numel() > 0 else ee_edge_weight

        # 归一化边向量(已去掉自环,直接归一化即可)
        def safe_normalize(vec: Tensor) -> Tensor:
            if vec.numel() == 0:
                return vec
            norm = torch.norm(vec, dim=1, keepdim=True)
            # Larger epsilon improves force-gradient stability when edge vectors are near zero.
            norm = norm.clamp(min=1e-4)
            return vec / norm

        if aa_edge_weight.numel() > 0:
            aa_edge_vec = safe_normalize(aa_edge_vec)

        if ae_edge_weight.numel() > 0:
            ae_edge_vec = safe_normalize(ae_edge_vec)

        if ea_edge_weight.numel() > 0:
            ea_edge_vec = safe_normalize(ea_edge_vec)

        if ee_edge_weight.numel() > 0:
            ee_edge_vec = safe_normalize(ee_edge_vec)

        # 应用球谐函数到原子相关的边向量
        aa_edge_vec_sph = self.sphere(aa_edge_vec)
        ae_edge_vec_sph = self.sphere(ae_edge_vec) if ae_edge_weight.numel() > 0 else ae_edge_vec
        ea_edge_vec_sph = self.sphere(ea_edge_vec) if ea_edge_weight.numel() > 0 else ea_edge_vec
        ee_edge_vec_sph = self.sphere(ee_edge_vec) if ee_edge_weight.numel() > 0 else ee_edge_vec

        # 径向基函数展开
        aa_rbf = torch.clamp(torch.nan_to_num(self.distance_expansion(aa_edge_weight)), -5.0, 5.0)
        ae_rbf = torch.clamp(torch.nan_to_num(self.distance_expansion(ae_edge_weight)), -5.0, 5.0) if ae_edge_weight.numel() > 0 else torch.empty(0, self.num_rbf, device=pos.device)
        ea_rbf = torch.clamp(torch.nan_to_num(self.distance_expansion(ea_edge_weight)), -5.0, 5.0) if ea_edge_weight.numel() > 0 else torch.empty(0, self.num_rbf, device=pos.device)
        ee_rbf = (
            torch.clamp(torch.nan_to_num(self.distance_expansion(ee_edge_weight)), -5.0, 5.0)
            if ee_edge_weight.numel() > 0
            else torch.empty(0, self.num_rbf, device=pos.device)
        )

        # 边嵌入
        aa_edge_attr = self.aa_edge_embedding(aa_rbf)
        ae_edge_attr = self.ae_edge_embedding(ae_rbf) if ae_rbf.numel() > 0 else torch.empty(0, self.hidden_channels, device=pos.device)
        ea_edge_attr = self.ea_edge_embedding(ea_rbf) if ea_rbf.numel() > 0 else torch.empty(0, self.hidden_channels, device=pos.device)
        ee_edge_attr = (
            self.ee_edge_embedding(ee_rbf)
            if ee_rbf.numel() > 0
            else torch.empty(0, self.hidden_channels, device=pos.device)
        )

        # 原子邻居初始化
        x_atom = self.neighbor_embedding(z, x_atom, aa_edge_index, aa_edge_weight, aa_rbf)

        # 重新打包graph_dict(使用球谐边向量)
        graph_dict_with_sph = {
            'aa': (aa_edge_index, aa_edge_weight, aa_edge_vec_sph),
            'ae': (ae_edge_index, ae_edge_weight, ae_edge_vec_sph),
            'ea': (ea_edge_index, ea_edge_weight, ea_edge_vec_sph),
            'ee': (ee_edge_index, ee_edge_weight, ee_edge_vec_sph),
        }

        edge_attr_dict = {
            'aa': torch.clamp(torch.nan_to_num(aa_edge_attr), -10.0, 10.0),
            'ae': torch.clamp(torch.nan_to_num(ae_edge_attr), -10.0, 10.0),
            'ea': torch.clamp(torch.nan_to_num(ea_edge_attr), -10.0, 10.0),
            'ee': torch.clamp(torch.nan_to_num(ee_edge_attr), -10.0, 10.0),
        }

        # === 6. 混合消息传递层 ===
        for i, layer in enumerate(self.hybrid_mp_layers[:-1]):
            dx_atom, dvec_atom, dx_electron, dvec_electron, new_edge_attr_dict = layer(
                x_atom, vec_atom, x_electron, vec_electron, graph_dict_with_sph, edge_attr_dict,
                sample_owner=sample_owner, electron_gate=gate, electron_confidence=token_confidence
            )
            x_atom = x_atom + dx_atom
            vec_atom = vec_atom + dvec_atom
            x_electron = x_electron + dx_electron
            vec_electron = vec_electron + dvec_electron
            # 更新边特征
            if new_edge_attr_dict is not None:
                edge_attr_dict['aa'] = torch.clamp(edge_attr_dict['aa'] + new_edge_attr_dict['aa'], -10.0, 10.0)
                if new_edge_attr_dict['ae'].numel() > 0:
                    edge_attr_dict['ae'] = torch.clamp(edge_attr_dict['ae'] + new_edge_attr_dict['ae'], -10.0, 10.0)
                if new_edge_attr_dict['ea'].numel() > 0:
                    edge_attr_dict['ea'] = torch.clamp(edge_attr_dict['ea'] + new_edge_attr_dict['ea'], -10.0, 10.0)
                edge_attr_dict['ee'] = torch.clamp(edge_attr_dict['ee'] + new_edge_attr_dict['ee'], -10.0, 10.0)

        # 最后一层
        dx_atom, dvec_atom, dx_electron, dvec_electron, _ = self.hybrid_mp_layers[-1](
            x_atom, vec_atom, x_electron, vec_electron, graph_dict_with_sph, edge_attr_dict,
            sample_owner=sample_owner, electron_gate=gate, electron_confidence=token_confidence
        )
        x_atom = x_atom + dx_atom
        vec_atom = vec_atom + dvec_atom

        # === 7. 输出归一化 ===
        x_atom = self.atom_out_norm(x_atom)
        vec_atom = self.atom_vec_out_norm(vec_atom)

        return x_atom, vec_atom


class EMolModel(nn.Module):
    """Predict molecular properties from the electron-aware representation."""
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super().__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            print(
                "Prior model was given but the output model does "
                "not allow prior models. Dropping the prior model."
            )

        self.reduce_op = reduce_op
        self.derivative = derivative

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播。

        Args:
            data: PyG Data对象,包含分子和电子数据

        Returns:
            out: 预测张量
            dy: 力张量 (如果derivative=True)
        """

        if self.derivative:
            data.pos.requires_grad_(True)

        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, data.z)

        out = scatter(x, data.batch, dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)

        out = out + self.mean

        # 计算相对于坐标的梯度
        if self.derivative:
            grad_outputs = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data.pos],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, -dy
        return out, None
