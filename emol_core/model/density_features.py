"""
Differentiable electron statistics for sampled points.

Compared with hard-threshold counting, this module uses smooth kernel weights
to keep energy/force gradients stable with respect to sampled coordinates.
"""

import torch
import torch.nn as nn
from torch import Tensor


class SampledElectronFeatures(nn.Module):
    """
    Extract smooth local electron descriptors for sampled points.

    Features (17 total):
    1  weighted mean density
    2  weighted std density
    3  weighted L1 density
    4  weighted L2 density
    5  weighted density entropy
    6  effective sample size
    7  log(1 + kernel mass)
    8  weighted density sum
    9  weighted mean distance
    10 weighted std distance
    11 density-distance covariance
    12-14 weighted centroid offset (x,y,z)
    15 weighted second radial moment
    16 weighted skewness of density
    17 weighted kurtosis of density
    """

    def __init__(self, initial_radius=2.0, learnable_radius=True):
        super().__init__()

        radius = torch.tensor(float(initial_radius), dtype=torch.float32)
        if learnable_radius:
            self.radius = nn.Parameter(radius)
        else:
            self.register_buffer("radius", radius)
        self.num_features = 17
        self.eps = 1e-8
        # Temperature for smooth cutoff around radius.
        self.smooth_temp = 0.35

    def _kernel_weights(self, distances: Tensor, batch_mask: Tensor) -> Tensor:
        radius = self.radius.to(device=distances.device, dtype=distances.dtype)
        sigma = torch.clamp(radius, min=1e-4)
        # Gaussian kernel for smooth locality.
        gaussian = torch.exp(-0.5 * (distances / sigma) ** 2)
        # Smooth cutoff near radius to reduce long-tail noise.
        temp = torch.clamp(self.smooth_temp * sigma, min=1e-4)
        soft_cutoff = torch.sigmoid((radius - distances) / temp)
        return gaussian * soft_cutoff * batch_mask.to(distances.dtype)

    def _compute_features_for_chunk(
        self,
        sampled_pos_chunk: Tensor,
        elec_coords: Tensor,
        density: Tensor,
    ) -> Tensor:
        num_sampled = sampled_pos_chunk.size(0)
        if num_sampled == 0:
            return sampled_pos_chunk.new_zeros((0, self.num_features))
        if elec_coords.numel() == 0:
            return sampled_pos_chunk.new_zeros((num_sampled, self.num_features))

        sampled_pos_expanded = sampled_pos_chunk.unsqueeze(1)  # [N_sampled_chunk, 1, 3]
        elec_pos_expanded = elec_coords.unsqueeze(0)  # [1, N_electrons_local, 3]
        delta = elec_pos_expanded - sampled_pos_expanded  # [N_sampled_chunk, N_electrons_local, 3]
        distances = torch.norm(delta, dim=-1)  # [N_sampled_chunk, N_electrons_local]

        batch_mask = torch.ones_like(distances, dtype=torch.bool)
        weights = self._kernel_weights(distances, batch_mask)

        w_sum_raw = weights.sum(dim=1, keepdim=True)
        w_sum = w_sum_raw + self.eps
        w_norm = weights / w_sum
        has_mass = w_sum_raw > 1e-6

        density_matrix = density.unsqueeze(0).expand_as(distances)
        density_pos = density_matrix.clamp(min=0.0)

        mean_density = (w_norm * density_matrix).sum(dim=1, keepdim=True)
        centered_density = density_matrix - mean_density
        var_density = (w_norm * centered_density.pow(2)).sum(dim=1, keepdim=True)
        std_density = torch.sqrt(var_density + self.eps)

        m3_density = (w_norm * centered_density.pow(3)).sum(dim=1, keepdim=True)
        m4_density = (w_norm * centered_density.pow(4)).sum(dim=1, keepdim=True)
        skew_density = m3_density / (std_density.pow(3) + self.eps)
        kurt_density = m4_density / (var_density.pow(2) + self.eps)

        l1_density = (w_norm * density_matrix.abs()).sum(dim=1, keepdim=True)
        l2_density = torch.sqrt((w_norm * density_matrix.pow(2)).sum(dim=1, keepdim=True) + self.eps)

        weighted_prob = weights * density_pos
        prob_norm = weighted_prob / (weighted_prob.sum(dim=1, keepdim=True) + self.eps)
        density_entropy = -(prob_norm * torch.log(prob_norm + self.eps)).sum(dim=1, keepdim=True)

        effective_n = 1.0 / (w_norm.pow(2).sum(dim=1, keepdim=True) + self.eps)
        log_kernel_mass = torch.log1p(w_sum_raw)
        weighted_density_sum = (weights * density_matrix).sum(dim=1, keepdim=True)

        mean_distance = (w_norm * distances).sum(dim=1, keepdim=True)
        centered_distance = distances - mean_distance
        var_distance = (w_norm * centered_distance.pow(2)).sum(dim=1, keepdim=True)
        std_distance = torch.sqrt(var_distance + self.eps)
        density_distance_cov = (w_norm * centered_density * centered_distance).sum(dim=1, keepdim=True)
        centroid_offset = (w_norm.unsqueeze(-1) * delta).sum(dim=1)
        radial_second_moment = (w_norm * distances.pow(2)).sum(dim=1, keepdim=True)

        features = torch.cat(
            [
                mean_density,
                std_density,
                l1_density,
                l2_density,
                density_entropy,
                effective_n,
                log_kernel_mass,
                weighted_density_sum,
                mean_distance,
                std_distance,
                density_distance_cov,
                centroid_offset,
                radial_second_moment,
                skew_density,
                kurt_density,
            ],
            dim=1,
        )
        return torch.where(has_mass.expand_as(features), features, torch.zeros_like(features))

    def forward(self, sampled_pos: Tensor, sampled_batch: Tensor,
                elec_coords: Tensor, density: Tensor, elec_batch: Tensor) -> Tensor:
        """
        Build differentiable local electron descriptors for sampled points.
        """
        features = sampled_pos.new_zeros((sampled_pos.size(0), self.num_features))
        if sampled_pos.numel() == 0:
            return features

        chunk_size = max(1, 128 // max(1, sampled_pos.size(1)))
        unique_batches = torch.unique(sampled_batch)

        for batch_id in unique_batches.tolist():
            sampled_mask = sampled_batch == batch_id
            elec_mask = elec_batch == batch_id

            local_sampled_pos = sampled_pos[sampled_mask]
            local_elec_coords = elec_coords[elec_mask]
            local_density = density[elec_mask]

            local_features = []
            for start in range(0, local_sampled_pos.size(0), chunk_size):
                end = start + chunk_size
                local_features.append(
                    self._compute_features_for_chunk(
                        local_sampled_pos[start:end],
                        local_elec_coords,
                        local_density,
                    )
                )

            if local_features:
                features[sampled_mask] = torch.cat(local_features, dim=0)

        return features


class SampledElectronEmbedding(nn.Module):
    """
    Project sampled electron descriptors into hidden channels.
    """

    def __init__(self, hidden_channels, initial_radius=2.0, learnable_radius=True):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.electron_features = SampledElectronFeatures(
            initial_radius=initial_radius,
            learnable_radius=learnable_radius,
        )

        self.feature_projection = nn.Sequential(
            nn.Linear(self.electron_features.num_features, hidden_channels),
            nn.SiLU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        for m in self.feature_projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sampled_pos: Tensor, sampled_batch: Tensor,
                elec_coords: Tensor, density: Tensor, elec_batch: Tensor) -> Tensor:
        """
        Build sampled electron embeddings.
        """
        electron_features = self.electron_features(
            sampled_pos,
            sampled_batch,
            elec_coords,
            density,
            elec_batch,
        )

        embedding = self.feature_projection(electron_features)

        return embedding
