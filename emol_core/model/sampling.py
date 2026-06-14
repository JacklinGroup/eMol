"""
Sampling utilities for electron cloud representation.

包含最远点采样(FPS)和相关功能。
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    最远点采样算法 (Farthest Point Sampling, FPS).

    从点云中选择npoint个点,使得这些点尽可能分散。

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud indices, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def batch_farthest_point_sample(
    atom_pos: Tensor,
    atom_batch: Tensor,
    electron_pos: Tensor,
    electron_batch: Tensor,
    npoint: int,
    radius: Union[float, Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    对每个原子在其局部邻域内执行最远点采样。

    Args:
        atom_pos: [N_atoms, 3] 原子坐标
        atom_batch: [N_atoms] 原子batch索引
        electron_pos: [N_electrons, 3] 电子坐标
        electron_batch: [N_electrons] 电子batch索引
        npoint: 每个原子采样的点数
        radius: 局部邻域半径(单位Å)

    Returns:
        sampled_pos: [N_atoms * npoint, 3] 采样点的坐标
        sampled_batch: [N_atoms * npoint] 采样点的batch索引
    """
    device = atom_pos.device
    num_atoms = atom_pos.size(0)

    sampled_pos_list = []
    sampled_batch_list = []

    for atom_idx in range(num_atoms):
        batch_id = atom_batch[atom_idx]
        # 同一分子的电子
        same_batch_mask = (electron_batch == batch_id)
        local_electrons = electron_pos[same_batch_mask]

        if local_electrons.numel() == 0:
            # 该分子没有电子数据,退化为使用原子坐标本身
            fallback_pos = atom_pos[atom_idx].unsqueeze(0).repeat(npoint, 1)
            sampled_pos_list.append(fallback_pos)
            sampled_batch_list.append(torch.full((npoint,), batch_id, dtype=torch.long, device=device))
            continue

        # 计算与当前原子的距离并筛选半径内的电子
        distances = torch.norm(local_electrons - atom_pos[atom_idx], dim=-1)
        within_radius_mask = distances <= radius

        if within_radius_mask.any():
            candidates = local_electrons[within_radius_mask]
        else:
            # 若半径内没有电子,选择最近的电子作为候选
            nearest_k = min(npoint, local_electrons.size(0))
            nearest_indices = torch.topk(distances, k=nearest_k, largest=False).indices
            candidates = local_electrons[nearest_indices]

        num_candidates = candidates.size(0)

        if num_candidates >= npoint:
            # 在局部电子上执行FPS
            fps_indices = farthest_point_sample(candidates.unsqueeze(0), npoint).squeeze(0)
            sampled = candidates[fps_indices]
        else:
            # 候选不足时循环重复以保持固定采样数
            repeat_count = (npoint + num_candidates - 1) // num_candidates
            tiled_indices = torch.arange(num_candidates, device=device).repeat(repeat_count)[:npoint]
            sampled = candidates[tiled_indices]

        sampled_pos_list.append(sampled)
        sampled_batch_list.append(torch.full((npoint,), batch_id, dtype=torch.long, device=device))

    sampled_pos = torch.cat(sampled_pos_list, dim=0)
    sampled_batch = torch.cat(sampled_batch_list, dim=0)

    return sampled_pos, sampled_batch
