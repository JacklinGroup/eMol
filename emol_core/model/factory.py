import re

import torch

from emol_core.model.network import EMolModel, EMolRepresentation
from emol_core.model.readout import EquivariantScalar


def create_model(args, mean=None, std=None):
    # RAGEDSampledBlock is an alias for EMolRepresentation (same architecture)
    model_name = args.get("model", "EMolRepresentation")

    if model_name not in {"EMolRepresentation", "RAGEDSampledBlock"}:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Available: EMolRepresentation, RAGEDSampledBlock.")

    # Use .get() for all electron-specific args so YAML keys from RAGED configs work
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
        vertex_type=args.get("vertex_type", None),
        electron_radius=args.get("electron_radius", 2.0),
        learnable_radius=args.get("learnable_radius", True),
        num_sample_points=args.get("num_sample_points", 3),
        atom_token_extra_neighbors=args.get("atom_token_extra_neighbors", 1),
        electron_gate=args.get("electron_gate", 0.25),
        electron_gate_mode=args.get("electron_gate_mode", "fixed"),
        aeea_five_body_scale=args.get("aeea_five_body_scale", 0.0),
        aeea_five_body_use_gate=args.get("aeea_five_body_use_gate", True),
        ee_cutoff=args.get("ee_cutoff", None),
        ee_max_num_neighbors=args.get("ee_max_num_neighbors", None),
        ee_scalar_scale=args.get("ee_scalar_scale", 0.1),
        ee_vector_scale=args.get("ee_vector_scale", 0.05),
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
