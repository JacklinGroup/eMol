import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import radius_graph
from typing import Tuple


class HybridGraphBuilder(nn.Module):
    """Build atom-atom, atom-electron, electron-atom, and optional electron-electron edges."""

    def __init__(self, cutoff=5.0, max_num_neighbors=32, ee_cutoff=None, ee_max_num_neighbors=None):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.ee_cutoff = cutoff if ee_cutoff is None else ee_cutoff
        self.ee_max_num_neighbors = max_num_neighbors if ee_max_num_neighbors is None else ee_max_num_neighbors

    def build_atom_atom_edges(self, atom_pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = radius_graph(
            atom_pos,
            r=self.cutoff,
            batch=batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )

        row, col = edge_index
        edge_vec = atom_pos[row] - atom_pos[col]
        edge_weight = torch.norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec

    def build_atom_electron_edges(
        self,
        atom_pos: Tensor,
        atom_batch: Tensor,
        electron_pos: Tensor,
        electron_batch: Tensor,
        overlap_only: bool = False,
        electron_owner: Tensor = None,
        extra_owner_neighbors: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = atom_pos.device
        num_atoms = atom_pos.size(0)
        num_electrons = electron_pos.size(0)

        if overlap_only:
            if num_atoms != num_electrons:
                raise ValueError("overlap_only requires the same number of atoms and electron tokens.")
            if not torch.equal(atom_batch, electron_batch):
                raise ValueError("overlap_only requires matching atom and electron-token batches.")

            indices = torch.arange(num_atoms, device=device)
            edge_index = torch.stack([indices, indices], dim=0)
            edge_vec = electron_pos - atom_pos
            edge_weight = torch.norm(edge_vec, dim=-1)
            return edge_index, edge_weight, edge_vec

        if electron_owner is not None:
            edge_index_parts = []
            edge_weight_parts = []
            edge_vec_parts = []

            for electron_idx in range(num_electrons):
                owner_idx = int(electron_owner[electron_idx].item())
                token_pos = electron_pos[electron_idx]

                owner_edge_vec = token_pos - atom_pos[owner_idx]
                owner_edge_weight = torch.norm(owner_edge_vec, dim=-1, keepdim=True)
                owner_edge_index = torch.tensor(
                    [[owner_idx], [electron_idx]],
                    dtype=torch.long,
                    device=device,
                )

                edge_index_parts.append(owner_edge_index)
                edge_weight_parts.append(owner_edge_weight.view(1))
                edge_vec_parts.append(owner_edge_vec.view(1, 3))

                if extra_owner_neighbors <= 0:
                    continue

                same_batch_mask = atom_batch == electron_batch[electron_idx]
                same_batch_mask[owner_idx] = False
                candidate_indices = torch.nonzero(same_batch_mask, as_tuple=False).view(-1)
                if candidate_indices.numel() == 0:
                    continue

                candidate_pos = atom_pos[candidate_indices]
                candidate_vec = token_pos.unsqueeze(0) - candidate_pos
                candidate_dist = torch.norm(candidate_vec, dim=-1)
                within_cutoff = candidate_dist <= self.cutoff
                if not within_cutoff.any():
                    continue

                candidate_indices = candidate_indices[within_cutoff]
                candidate_vec = candidate_vec[within_cutoff]
                candidate_dist = candidate_dist[within_cutoff]

                k = min(int(extra_owner_neighbors), candidate_indices.numel())
                nearest = torch.topk(candidate_dist, k=k, largest=False).indices
                chosen_indices = candidate_indices[nearest]
                chosen_vec = candidate_vec[nearest]
                chosen_dist = candidate_dist[nearest]

                edge_index_parts.append(
                    torch.stack(
                        [chosen_indices, torch.full_like(chosen_indices, electron_idx)],
                        dim=0,
                    )
                )
                edge_weight_parts.append(chosen_dist)
                edge_vec_parts.append(chosen_vec)

            if len(edge_index_parts) == 0:
                empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
                empty_weight = torch.empty((0,), dtype=atom_pos.dtype, device=device)
                empty_vec = torch.empty((0, 3), dtype=atom_pos.dtype, device=device)
                return empty_index, empty_weight, empty_vec

            edge_index = torch.cat(edge_index_parts, dim=1)
            edge_weight = torch.cat(edge_weight_parts, dim=0)
            edge_vec = torch.cat(edge_vec_parts, dim=0)
            return edge_index, edge_weight, edge_vec

        edge_index_parts = []
        edge_weight_parts = []
        edge_vec_parts = []

        for batch_id in torch.unique(atom_batch).tolist():
            atom_mask = atom_batch == batch_id
            electron_mask = electron_batch == batch_id
            if not atom_mask.any() or not electron_mask.any():
                continue

            local_atom_indices = torch.nonzero(atom_mask, as_tuple=False).view(-1)
            local_electron_indices = torch.nonzero(electron_mask, as_tuple=False).view(-1)
            local_atom_pos = atom_pos[local_atom_indices]
            local_electron_pos = electron_pos[local_electron_indices]

            edge_vec_matrix = local_electron_pos.unsqueeze(0) - local_atom_pos.unsqueeze(1)
            distance_matrix = torch.norm(edge_vec_matrix, dim=-1)
            valid_mask = distance_matrix <= self.cutoff
            if not valid_mask.any():
                continue

            atom_indices_local, electron_indices_local = torch.where(valid_mask)
            atom_indices = local_atom_indices[atom_indices_local]
            electron_indices = local_electron_indices[electron_indices_local]

            edge_index_parts.append(torch.stack([atom_indices, electron_indices], dim=0))
            edge_weight_parts.append(distance_matrix[atom_indices_local, electron_indices_local])
            edge_vec_parts.append(edge_vec_matrix[atom_indices_local, electron_indices_local])

        if len(edge_index_parts) == 0:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_weight = torch.empty((0,), dtype=atom_pos.dtype, device=device)
            empty_vec = torch.empty((0, 3), dtype=atom_pos.dtype, device=device)
            return empty_index, empty_weight, empty_vec

        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_weight = torch.cat(edge_weight_parts, dim=0)
        edge_vec = torch.cat(edge_vec_parts, dim=0)
        return edge_index, edge_weight, edge_vec

    def build_electron_electron_edges(self, electron_pos: Tensor, electron_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        edge_index = radius_graph(
            electron_pos,
            r=self.ee_cutoff,
            batch=electron_batch,
            loop=False,
            max_num_neighbors=self.ee_max_num_neighbors,
        )

        row, col = edge_index
        edge_vec = electron_pos[row] - electron_pos[col]
        edge_weight = torch.norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec

    def forward(
        self,
        atom_pos: Tensor,
        atom_batch: Tensor,
        electron_pos: Tensor,
        electron_batch: Tensor,
        overlap_only: bool = False,
        include_ee: bool = True,
        electron_owner: Tensor = None,
        extra_owner_neighbors: int = 0,
    ) -> dict:
        aa_edge_index, aa_edge_weight, aa_edge_vec = self.build_atom_atom_edges(atom_pos, atom_batch)
        ae_edge_index, ae_edge_weight, ae_edge_vec = self.build_atom_electron_edges(
            atom_pos,
            atom_batch,
            electron_pos,
            electron_batch,
            overlap_only=overlap_only,
            electron_owner=electron_owner,
            extra_owner_neighbors=extra_owner_neighbors,
        )

        ea_edge_index = torch.stack([ae_edge_index[1], ae_edge_index[0]], dim=0)
        ea_edge_weight = ae_edge_weight
        ea_edge_vec = -ae_edge_vec

        if include_ee:
            ee_edge_index, ee_edge_weight, ee_edge_vec = self.build_electron_electron_edges(
                electron_pos,
                electron_batch,
            )
        else:
            ee_edge_index = torch.empty((2, 0), dtype=torch.long, device=atom_pos.device)
            ee_edge_weight = torch.empty((0,), dtype=atom_pos.dtype, device=atom_pos.device)
            ee_edge_vec = torch.empty((0, 3), dtype=atom_pos.dtype, device=atom_pos.device)

        return {
            "aa": (aa_edge_index, aa_edge_weight, aa_edge_vec),
            "ae": (ae_edge_index, ae_edge_weight, ae_edge_vec),
            "ea": (ea_edge_index, ea_edge_weight, ea_edge_vec),
            "ee": (ee_edge_index, ee_edge_weight, ee_edge_vec),
        }
