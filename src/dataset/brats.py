import pathlib

import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

from src.config import get_brats_folder
from src.dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, debug=False, data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3


def get_datasets(seed, debug, no_seg=False, on="train", full=False,
                 fold_number=0, normalisation="minmax", exclude_ids=None):
    """exclude_ids: optional iterable of patient-dir names (e.g. "BraTS20_Training_141")
    to drop from the *training* split only -- for the "filtered/cleaned dataset"
    Pipeline-A variant (cases flagged as high-training-loss by
    src/find_noisy_patients.py). Val/benchmark sets for the fold are left
    untouched on purpose, so a filtered-vs-unfiltered comparison for the same
    fold is evaluated on exactly the same held-out patients.
    """
    base_folder = pathlib.Path(get_brats_folder(on)).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    exclude_ids = set(exclude_ids) if exclude_ids else set()
    if full:
        train_patients = [p for p in patients_dir if p.name not in exclude_ids]
        if exclude_ids:
            print(f"Excluded {len(patients_dir) - len(train_patients)} patient(s) from "
                  f"training (of {len(exclude_ids)} requested): {sorted(exclude_ids)}")
        train_dataset = Brats(train_patients, training=True, debug=debug,
                              normalisation=normalisation)
        bench_dataset = Brats(patients_dir, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset
    if no_seg:
        return Brats(patients_dir, training=False, debug=debug,
                     no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    print("first idx of train", train_idx[0])
    print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    if exclude_ids:
        before = len(train)
        train = [p for p in train if p.name not in exclude_ids]
        print(f"Excluded {before - len(train)} patient(s) from fold {fold_number}'s training "
              f"split (of {len(exclude_ids)} requested -- some may belong to this fold's val "
              f"split instead, where they are deliberately left in): {sorted(exclude_ids)}")
    # return patients_dir
    train_dataset = Brats(train, training=True,  debug=debug,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,  debug=debug,
                        normalisation=normalisation)
    bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset
