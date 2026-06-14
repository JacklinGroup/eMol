import re

import torch

from emol_core.model.network import EMolModel, EMolRepresentation
from emol_core.model.readout import EquivariantScalar


def create_model(args, mean=None, std=None):
    representation = EMolRepresentation(
        lmax=args["lmax"],
        vecnorm_type=args["vecnorm_type"],
        trainable_vecnorm=args["trainable_vecnorm"],
        num_heads=args["num_heads"],
        num_layers=args["num_layers"],
        hidden_channels=args["embedding_dimension"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        attn_activation=args["attn_activation"],
        max_z=args["max_z"],
        cutoff=args["cutoff"],
        max_num_neighbors=args["max_num_neighbors"],
        vertex_type=args["vertex_type"],
        electron_radius=args["electron_radius"],
        learnable_radius=args["learnable_radius"],
        num_sample_points=args["num_sample_points"],
        atom_token_extra_neighbors=args["atom_token_extra_neighbors"],
        electron_gate=args["electron_gate"],
        electron_gate_mode=args["electron_gate_mode"],
        aeea_five_body_scale=args["aeea_five_body_scale"],
        aeea_five_body_use_gate=args["aeea_five_body_use_gate"],
        ee_cutoff=args["ee_cutoff"],
        ee_max_num_neighbors=args["ee_max_num_neighbors"],
        ee_scalar_scale=args["ee_scalar_scale"],
        ee_vector_scale=args["ee_vector_scale"],
    )
    output_model = EquivariantScalar(args["embedding_dimension"], args["activation"])
    return EMolModel(
        representation,
        output_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
    )


def load_model(filepath, args=None, device="cpu"):
    checkpoint = torch.load(filepath, map_location="cpu")
    config = checkpoint["hyper_parameters"] if args is None else args
    model = create_model(config)
    state_dict = {
        re.sub(r"^model\.", "", key): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model.to(device)
