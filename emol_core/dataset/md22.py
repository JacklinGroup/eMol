"""MD22 dataset for molecular dynamics."""

import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class MD22(InMemoryDataset):
    def __init__(self, root, dataset_arg=None, transform=None, pre_transform=None):
        assert dataset_arg is not None, "Please provide the desired molecule name via 'dataset_arg'."
        self.dataset_arg = dataset_arg
        self._source_root = osp.abspath(root)
        super().__init__(osp.join(root, dataset_arg), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def molecule_names(self):
        return dict(
            Ac_Ala3_NHMe="md22_Ac-Ala3-NHMe.npz",
            DHA="md22_DHA.npz",
            stachyose="md22_stachyose.npz",
            AT_AT="md22_AT-AT.npz",
            AT_AT_CG_CG="md22_AT-AT-CG-CG.npz",
            buckyball_catcher="md22_buckyball-catcher.npz",
            double_walled_nanotube="md22_dw_nanotube.npz",
        )

    @property
    def molecule_splits(self):
        return dict(
            Ac_Ala3_NHMe=6000, DHA=8000, stachyose=8000,
            AT_AT=3000, AT_AT_CG_CG=2000,
            buckyball_catcher=600, double_walled_nanotube=800,
        )

    @property
    def raw_file_names(self):
        return [self._resolve_raw_file(self.dataset_arg)]

    @property
    def processed_file_names(self):
        return [f"md22_{self.dataset_arg}.pt"]

    @property
    def base_url(self):
        return "http://www.quantum-machine.org/gdml/data/npz/"

    def download(self):
        download_url(self.base_url + self.molecule_names[self.dataset_arg], self.raw_dir)

    def _resolve_raw_file(self, molecule):
        file_name = self.molecule_names[molecule]
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
