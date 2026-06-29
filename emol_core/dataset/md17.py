"""MD17 dataset for molecular dynamics."""

import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class MD17(InMemoryDataset):
    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="md17_aspirin.npz",
        ethanol="md17_ethanol.npz",
        malonaldehyde="md17_malonaldehyde.npz",
        naphthalene="md17_naphthalene.npz",
        salicylic_acid="md17_salicylic.npz",
        toluene="md17_toluene.npz",
        uracil="md17_uracil.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, dataset_arg=None, transform=None, pre_transform=None,
                 electron_db_path=None, electron_topk=5):
        assert dataset_arg is not None, (
            "Please provide the desired molecule through 'dataset_arg'. "
            f"Available: {', '.join(self.available_molecules)}"
        )
        self.dataset_arg = dataset_arg
        self._source_root = osp.abspath(root)
        super().__init__(osp.join(root, dataset_arg), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self._resolve_raw_file(self.dataset_arg)]

    @property
    def processed_file_names(self):
        return [f"md17_{self.dataset_arg}.pt"]

    def download(self):
        download_url(self.raw_url + self.molecule_files[self.dataset_arg], self.raw_dir)

    def _resolve_raw_file(self, molecule):
        file_name = self.molecule_files[molecule]
        for c in [osp.join(self._source_root, f"after_{file_name}"),
                  osp.join(self._source_root, file_name)]:
            if osp.isfile(c):
                return osp.abspath(c)
        return file_name

    @staticmethod
    def _load_augmented_fields(data_npz):
        if "elec_coords" not in data_npz or "density" not in data_npz:
            return None, None, None
        elec_coords = torch.from_numpy(data_npz["elec_coords"]).float()
        density = torch.from_numpy(data_npz["density"]).float()
        num_electrons = (torch.from_numpy(data_npz["num_electrons"]).long()
                         if "num_electrons" in data_npz
                         else torch.full((elec_coords.size(0),), elec_coords.size(1), dtype=torch.long))
        return elec_coords, density, num_electrons

    def process(self):
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()
            elec_coords, density, num_electrons = self._load_augmented_fields(data_npz)

            samples = []
            iterator = zip(positions, energies, forces)
            if elec_coords is not None:
                iterator = zip(positions, energies, forces, elec_coords, density, num_electrons)

            for frame in tqdm(iterator, total=energies.size(0)):
                if elec_coords is None:
                    pos, y, dy = frame
                    data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy)
                else:
                    pos, y, dy, frame_ec, frame_d, frame_ne = frame
                    data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy,
                                elec_coords=frame_ec, density=frame_d, num_electrons=frame_ne.view(1))
                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)
