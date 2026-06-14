import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from typing import Optional

from emol_core.model.geometry import CosineCutoff, VecLayerNorm, act_class_mapping


class HybridMessagePassing(nn.Module):
    """Hybrid message passing for atom/electron heterogeneous graphs."""

    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        ee_cutoff=None,
        aeea_five_body_scale=0.0,
        aeea_five_body_use_gate=True,
        ee_scalar_scale=0.1,
        ee_vector_scale=0.05,
        last_layer=False,
    ):
        super(HybridMessagePassing, self).__init__()

        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer
        self.ee_cutoff = cutoff if ee_cutoff is None else ee_cutoff

        # Reuse existing config knobs as cross-branch blending controls.
        self.aeea_five_body_scale = float(aeea_five_body_scale)
        self.aeea_five_body_use_gate = bool(aeea_five_body_use_gate)
        self.ee_scalar_scale = float(ee_scalar_scale)
        self.ee_vector_scale = float(ee_vector_scale)

        self.atom_layernorm = nn.LayerNorm(hidden_channels)
        self.atom_vec_layernorm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        )
        self.electron_layernorm = nn.LayerNorm(hidden_channels)
        self.electron_vec_layernorm = VecLayerNorm(
            hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type
        )

        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff)
        self.ee_cutoff_fn = CosineCutoff(self.ee_cutoff)

        # Atom branch (existing).
        self.atom_vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.atom_q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.atom_k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.atom_v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.atom_o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.atom_s_proj = nn.Linear(hidden_channels, hidden_channels * 2)

        # Electron scalar branch (existing EE path).
        self.electron_vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        self.electron_q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.electron_s_proj = nn.Linear(hidden_channels, hidden_channels * 2)

        # Distance-conditioned transforms for AA/EE (existing).
        self.atom_dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.atom_dv_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.electron_dv_proj = nn.Linear(hidden_channels, hidden_channels)

        # Cross branches upgraded to QuinNet-like full formulation.
        self._cross_prefixes = ("ae", "ea")
        for prefix in self._cross_prefixes:
            self._init_cross_branch(prefix)

        if not self.last_layer:
            self.aa_f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.aa_w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.aa_w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.ee_f_proj = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def _init_cross_branch(self, prefix: str):
        setattr(self, f"{prefix}_q_proj", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_k_proj", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_v_proj", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_o_proj", nn.Linear(self.hidden_channels, self.hidden_channels * 3))
        setattr(self, f"{prefix}_s_proj", nn.Linear(self.hidden_channels, self.hidden_channels * 2))
        setattr(self, f"{prefix}_dk_proj", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_dv_proj", nn.Linear(self.hidden_channels, self.hidden_channels))

        setattr(self, f"{prefix}_f_proj1", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_f_proj2", nn.Linear(self.hidden_channels, self.hidden_channels))
        setattr(self, f"{prefix}_f_proj3", nn.Linear(self.hidden_channels, self.hidden_channels))

        setattr(self, f"{prefix}_src_proj", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_trg_proj", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))

        setattr(self, f"{prefix}_vec_proj", nn.Linear(self.hidden_channels, self.hidden_channels * 3, bias=False))
        setattr(self, f"{prefix}_vec_proj_improper1", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_vec_proj_improper2", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_vec_proj_improper3", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_vec_proj_improper4", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_vec_proj_improper5", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))

        setattr(self, f"{prefix}_src_proj1", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_src_proj2", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        setattr(self, f"{prefix}_dst_proj1", nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))

        setattr(self, f"{prefix}_ratio", nn.Parameter(torch.ones(6, )))

    @staticmethod
    def vector_rejection(vec: Tensor, direction: Tensor) -> Tensor:
        """Remove the projection of vec along direction."""
        vec_proj = (vec * direction).sum(dim=1, keepdim=True)
        return vec - vec_proj * direction

    def _cross_quinnet_update(
        self,
        prefix: str,
        x_tgt: Tensor,
        vec_tgt: Tensor,
        x_src: Tensor,
        vec_src: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
        edge_gate: Optional[Tensor] = None,
    ):
        q_proj = getattr(self, f"{prefix}_q_proj")
        k_proj = getattr(self, f"{prefix}_k_proj")
        v_proj = getattr(self, f"{prefix}_v_proj")
        o_proj = getattr(self, f"{prefix}_o_proj")
        s_proj = getattr(self, f"{prefix}_s_proj")
        dk_proj = getattr(self, f"{prefix}_dk_proj")
        dv_proj = getattr(self, f"{prefix}_dv_proj")

        f_proj1 = getattr(self, f"{prefix}_f_proj1")
        f_proj2 = getattr(self, f"{prefix}_f_proj2")
        f_proj3 = getattr(self, f"{prefix}_f_proj3")

        src_proj = getattr(self, f"{prefix}_src_proj")
        trg_proj = getattr(self, f"{prefix}_trg_proj")

        vec_proj = getattr(self, f"{prefix}_vec_proj")
        vec_proj_improper1 = getattr(self, f"{prefix}_vec_proj_improper1")
        vec_proj_improper2 = getattr(self, f"{prefix}_vec_proj_improper2")
        vec_proj_improper3 = getattr(self, f"{prefix}_vec_proj_improper3")
        vec_proj_improper4 = getattr(self, f"{prefix}_vec_proj_improper4")
        vec_proj_improper5 = getattr(self, f"{prefix}_vec_proj_improper5")

        src_proj1 = getattr(self, f"{prefix}_src_proj1")
        src_proj2 = getattr(self, f"{prefix}_src_proj2")
        dst_proj1 = getattr(self, f"{prefix}_dst_proj1")
        ratio = getattr(self, f"{prefix}_ratio")

        q = q_proj(x_tgt).reshape(-1, self.num_heads, self.head_dim)
        k = k_proj(x_src).reshape(-1, self.num_heads, self.head_dim)
        v = v_proj(x_src).reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(vec_proj(vec_tgt), self.hidden_channels, dim=-1)
        vec_dot1 = (vec1 * vec2).sum(dim=1)

        normal_vec1_tgt = self.vector_rejection(vec_proj_improper1(vec_tgt), vec_proj_improper2(vec_tgt))
        normal_vec1_src = self.vector_rejection(vec_proj_improper1(vec_src), vec_proj_improper2(vec_src))
        vec_dot2 = (normal_vec1_tgt ** 2).sum(dim=1)

        normal_vec2_tgt = self.vector_rejection(vec_proj_improper3(vec_tgt), vec_proj_improper4(vec_tgt))
        vec_dot6 = (normal_vec2_tgt * vec_proj_improper5(vec_tgt)).sum(dim=1)
        vec_dot1 = torch.clamp(torch.nan_to_num(vec_dot1), -50.0, 50.0)
        vec_dot2 = torch.clamp(torch.nan_to_num(vec_dot2), -50.0, 50.0)
        vec_dot6 = torch.clamp(torch.nan_to_num(vec_dot6), -50.0, 50.0)

        if edge_index.numel() > 0:
            dst, src = edge_index
            edge_gate_scalar = None if edge_gate is None else edge_gate.to(device=x_tgt.device, dtype=x_tgt.dtype)

            vec_dot3_ = self.act(f_proj2(f_ij)) * (
                dst_proj1(normal_vec1_tgt[dst]) * src_proj1(normal_vec1_src[src])
            ).sum(dim=1)
            if edge_gate_scalar is not None:
                vec_dot3_ = vec_dot3_ * edge_gate_scalar.unsqueeze(1)
            vec_dot3_ = torch.clamp(torch.nan_to_num(vec_dot3_), -50.0, 50.0)
            vec_dot3 = scatter(vec_dot3_, dst, dim=0, dim_size=x_tgt.size(0), reduce="add")

            vec_dot4_ = self.act(f_proj3(f_ij)) * (src_proj2(normal_vec1_src[src]) ** 2).sum(dim=1)
            if edge_gate_scalar is not None:
                vec_dot4_ = vec_dot4_ * edge_gate_scalar.unsqueeze(1)
            vec_dot4_ = torch.clamp(torch.nan_to_num(vec_dot4_), -50.0, 50.0)
            vec_dot4 = scatter(vec_dot4_, dst, dim=0, dim_size=x_tgt.size(0), reduce="add")

            dk = self.act(dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            dv = self.act(dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)

            attn = (q[dst] * k[src] * dk).sum(dim=-1)
            attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
            if edge_gate_scalar is not None:
                attn = attn * edge_gate_scalar.unsqueeze(1)
            attn = torch.clamp(torch.nan_to_num(attn), -50.0, 50.0)

            v_j = v[src] * dv
            v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)
            v_j = torch.clamp(torch.nan_to_num(v_j), -100.0, 100.0)

            s1, s2 = torch.split(self.act(s_proj(v_j)), self.hidden_channels, dim=1)
            vec_msg = vec_src[src] * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

            x_agg = scatter(v_j, dst, dim=0, dim_size=x_tgt.size(0), reduce="add")
            vec_out = scatter(vec_msg, dst, dim=0, dim_size=vec_tgt.size(0), reduce="add")
            vec_out = torch.clamp(torch.nan_to_num(vec_out), -100.0, 100.0)

            w1 = self.vector_rejection(trg_proj(vec_tgt[dst]), d_ij.unsqueeze(2))
            w2 = self.vector_rejection(src_proj(vec_src[src]), -d_ij.unsqueeze(2))
            vec_dot5_ = self.act(f_proj1(f_ij)) * (w1 * w2).sum(dim=1)
            if edge_gate_scalar is not None:
                vec_dot5_ = vec_dot5_ * edge_gate_scalar.unsqueeze(1)
            vec_dot5_ = torch.clamp(torch.nan_to_num(vec_dot5_), -50.0, 50.0)
            vec_dot5 = scatter(vec_dot5_, dst, dim=0, dim_size=x_tgt.size(0), reduce="add")

            dedge_attr = vec_dot5_ + vec_dot3_ + vec_dot4_
            dedge_attr = torch.clamp(torch.nan_to_num(dedge_attr), -20.0, 20.0)
        else:
            x_agg = torch.zeros_like(x_tgt)
            vec_out = torch.zeros_like(vec_tgt)
            vec_dot3 = torch.zeros_like(vec_dot1)
            vec_dot4 = torch.zeros_like(vec_dot1)
            vec_dot5 = torch.zeros_like(vec_dot1)
            dedge_attr = torch.zeros_like(f_ij)

        o1, o2, o3 = torch.split(o_proj(x_agg), self.hidden_channels, dim=1)
        dx = (
            torch.stack([vec_dot1, vec_dot2, vec_dot3, vec_dot4, vec_dot5, vec_dot6], dim=0)
            * ratio[:, None, None]
        ).sum(dim=0) * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        dx = torch.clamp(torch.nan_to_num(dx), -100.0, 100.0)
        dvec = torch.clamp(torch.nan_to_num(dvec), -100.0, 100.0)

        return dx, dvec, dedge_attr

    def reset_parameters(self):
        self.atom_layernorm.reset_parameters()
        self.atom_vec_layernorm.reset_parameters()
        self.electron_layernorm.reset_parameters()
        self.electron_vec_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.atom_q_proj.weight)
        self.atom_q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_k_proj.weight)
        self.atom_k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_v_proj.weight)
        self.atom_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_o_proj.weight)
        self.atom_o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_s_proj.weight)
        self.atom_s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_vec_proj.weight)

        nn.init.xavier_uniform_(self.atom_dk_proj.weight)
        self.atom_dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_dv_proj.weight)
        self.atom_dv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.electron_q_proj.weight)
        self.electron_q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_k_proj.weight)
        self.electron_k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_v_proj.weight)
        self.electron_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_o_proj.weight)
        self.electron_o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_s_proj.weight)
        self.electron_s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_vec_proj.weight)

        nn.init.xavier_uniform_(self.electron_dk_proj.weight)
        self.electron_dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.electron_dv_proj.weight)
        self.electron_dv_proj.bias.data.fill_(0)

        for prefix in self._cross_prefixes:
            for name in [
                "q_proj", "k_proj", "v_proj", "o_proj", "s_proj", "dk_proj", "dv_proj",
                "f_proj1", "f_proj2", "f_proj3",
            ]:
                layer = getattr(self, f"{prefix}_{name}")
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

            getattr(self, f"{prefix}_ratio").data.fill_(1.0)

            for name in [
                "src_proj", "trg_proj", "vec_proj",
                "vec_proj_improper1", "vec_proj_improper2", "vec_proj_improper3",
                "vec_proj_improper4", "vec_proj_improper5",
                "src_proj1", "src_proj2", "dst_proj1",
            ]:
                layer = getattr(self, f"{prefix}_{name}")
                nn.init.xavier_uniform_(layer.weight)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.aa_f_proj.weight)
            self.aa_f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.aa_w_src_proj.weight)
            nn.init.xavier_uniform_(self.aa_w_trg_proj.weight)

            nn.init.xavier_uniform_(self.ee_f_proj.weight)
            self.ee_f_proj.bias.data.fill_(0)

    def message_passing_aa(self, x_atom, vec_atom, edge_index, r_ij, f_ij, d_ij):
        q = self.atom_q_proj(x_atom).reshape(-1, self.num_heads, self.head_dim)
        k = self.atom_k_proj(x_atom).reshape(-1, self.num_heads, self.head_dim)
        v = self.atom_v_proj(x_atom).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.atom_dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.atom_dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)

        row, col = edge_index
        attn = (q[row] * k[col] * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        v_j = v[col] * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.atom_s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_atom[col] * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        x_agg = scatter(v_j, row, dim=0, dim_size=x_atom.size(0), reduce="add")
        vec_agg = scatter(vec_j, row, dim=0, dim_size=vec_atom.size(0), reduce="add")
        return x_agg, vec_agg

    def message_passing_ee(self, x_electron, vec_electron, edge_index, r_ij, f_ij, d_ij):
        q = self.electron_q_proj(x_electron).reshape(-1, self.num_heads, self.head_dim)
        k = self.electron_k_proj(x_electron).reshape(-1, self.num_heads, self.head_dim)
        v = self.electron_v_proj(x_electron).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.electron_dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.electron_dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)

        row, col = edge_index
        attn = (q[row] * k[col] * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.ee_cutoff_fn(r_ij).unsqueeze(1)

        v_j = v[col] * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)
        s1, s2 = torch.split(self.act(self.electron_s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_electron[col] * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        x_agg = scatter(v_j, row, dim=0, dim_size=x_electron.size(0), reduce="mean")
        vec_agg = scatter(vec_j, row, dim=0, dim_size=vec_electron.size(0), reduce="mean")
        return x_agg, vec_agg

    def _get_cross_scale(self, device, dtype) -> Tensor:
        # Use only the configured residual scale here. Electron reliability is
        # already handled upstream by token gating and edge-level confidence.
        base_scale = 1.0 if self.aeea_five_body_scale <= 0 else float(self.aeea_five_body_scale)
        return torch.tensor(base_scale, device=device, dtype=dtype)

    def forward(
        self,
        x_atom,
        vec_atom,
        x_electron,
        vec_electron,
        graph_dict,
        edge_attr_dict,
        sample_owner: Optional[Tensor] = None,
        electron_gate: Optional[Tensor] = None,
        electron_confidence: Optional[Tensor] = None,
    ):
        x_atom = torch.nan_to_num(x_atom)
        vec_atom = torch.nan_to_num(vec_atom)
        x_electron = torch.nan_to_num(x_electron)
        vec_electron = torch.nan_to_num(vec_electron)

        x_atom_norm = self.atom_layernorm(x_atom)
        vec_atom_norm = self.atom_vec_layernorm(vec_atom)
        x_electron_norm = self.electron_layernorm(x_electron)
        vec_electron_norm = self.electron_vec_layernorm(vec_electron)

        aa_edge_index, aa_r_ij, aa_d_ij = graph_dict["aa"]
        ea_edge_index, ea_r_ij, ea_d_ij = graph_dict["ea"]

        # Stored cross-edge indices are [src, dst], while _cross_quinnet_update expects [dst, src].
        # ea stored: [electron, atom] -> want [atom, electron]
        ea_edge_index_ds = torch.stack([ea_edge_index[1], ea_edge_index[0]], dim=0)
        ea_edge_gate = None
        if electron_confidence is not None and ea_edge_index.numel() > 0:
            ea_edge_gate = electron_confidence[ea_edge_index[0]]

        aa_f_ij = edge_attr_dict["aa"]
        ea_f_ij = edge_attr_dict["ea"]

        # Atom backbone + static electron-token context.
        vec1, vec2, vec3 = torch.split(self.atom_vec_proj(vec_atom_norm), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        x_aa, vec_aa = self.message_passing_aa(
            x_atom_norm, vec_atom_norm, aa_edge_index, aa_r_ij, aa_f_ij, aa_d_ij
        )

        o1_aa, o2_aa, o3_aa = torch.split(self.atom_o_proj(x_aa), self.hidden_channels, dim=1)
        dx_atom_aa = vec_dot * o2_aa + o3_aa
        dvec_atom_aa = vec3 * o1_aa.unsqueeze(1) + vec_aa

        dx_atom_ea, dvec_atom_ea, dea_quinn = self._cross_quinnet_update(
            "ea",
            x_atom_norm,
            vec_atom_norm,
            x_electron_norm,
            vec_electron_norm,
            ea_edge_index_ds,
            ea_r_ij,
            ea_f_ij,
            ea_d_ij,
            edge_gate=ea_edge_gate,
        )

        cross_scale = self._get_cross_scale(x_atom.device, x_atom.dtype)
        dx_atom = dx_atom_aa + cross_scale * dx_atom_ea
        dvec_atom = dvec_atom_aa + cross_scale * dvec_atom_ea

        # Electron probe tokens keep fixed node states; only EA edge states evolve.
        dx_electron = torch.zeros_like(x_electron)
        dvec_electron = torch.zeros_like(vec_electron)

        new_edge_attr_dict = None
        if not self.last_layer:
            row, col = aa_edge_index
            w1 = self.vector_rejection(self.aa_w_trg_proj(vec_atom_norm[row]), aa_d_ij.unsqueeze(2))
            w2 = self.vector_rejection(self.aa_w_src_proj(vec_atom_norm[col]), -aa_d_ij.unsqueeze(2))
            w_dot = (w1 * w2).sum(dim=1)
            daa_f_ij = self.act(self.aa_f_proj(aa_f_ij)) * w_dot

            new_edge_attr_dict = {
                "aa": daa_f_ij,
                "ae": torch.zeros_like(edge_attr_dict["ae"]),
                "ea": cross_scale * dea_quinn,
                "ee": torch.zeros_like(edge_attr_dict["ee"]),
            }

        return dx_atom, dvec_atom, dx_electron, dvec_electron, new_edge_attr_dict
