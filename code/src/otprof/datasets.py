from __future__ import annotations

from typing import List, Literal

import numpy as np
import torch
import xarray as xr


def _validate_ds(ds: xr.Dataset):
    # Check stacked sample dimension
    if "sample" not in ds.dims:
        raise ValueError("Dataset must have a 'sample' dimension.")

    # Check dimension
    if len(ds.sizes) > 2:
        raise ValueError("Dataset must be at most 2D (sample + vertical dimension).")

    # Check NaNs
    is_nan = ds.isnull().any()
    vars_nan = [v for v in is_nan if is_nan[v]]
    if vars_nan:
        raise ValueError(f"Dataset contains NaN values in variables: {vars_nan}")

    # Check Infs
    is_inf = ds.isin([np.inf, -np.inf]).any()
    vars_inf = [v for v in is_inf if is_inf[v]]
    if vars_inf:
        raise ValueError(f"Dataset contains Inf values in variables: {vars_inf}")


class ProfileDataset(torch.utils.data.Dataset):
    """Dataset with profiles outputs as matrix for conv-based models.
    Surface features get repeated along vertical dimension.
    """

    def __init__(
        self,
        ds_X: xr.Dataset,
        vars_X: List[str] | None,
        ds_y: xr.Dataset,
        vars_y: List[str] | None,
        vars_y_frc: List[str] | None,
        device=None,
    ):
        # Select vars
        if vars_X is None:
            vars_X = list(ds_X.data_vars)
        if vars_y is None:
            vars_y = list(ds_y.data_vars)

        # Validate datasets
        _validate_ds(ds_X[vars_X])
        # Allow empty target lists: only validate if there are variables to validate
        vars_to_validate = []
        if vars_y:
            vars_to_validate += vars_y
        if vars_y_frc:
            vars_to_validate += vars_y_frc
        if vars_to_validate:
            _validate_ds(ds_y[vars_to_validate])

        dim_s, dim_z_X = list(ds_X.dims)
        if dim_s != "sample":
            dim_s, dim_z_X = dim_z_X, dim_s  # Swap if necessary
            assert dim_s == "sample"  # validation should take care of this

        dim_s, dim_z_y = list(ds_y.dims)
        if dim_s != "sample":
            dim_s, dim_z_y = dim_z_y, dim_s  # Swap if necessary
            assert dim_s == "sample"  # validation should take care of this

        # Get input surface variables and expand them along vertical dimension
        vars_X_sfc = [v for v in vars_X if dim_z_X not in ds_X[v].dims]
        ds_X_sfc = ds_X[vars_X_sfc].expand_dims({dim_z_X: ds_X.sizes[dim_z_X]}, axis=1)

        vars_X_vert = [v for v in vars_X if v not in vars_X_sfc]
        ds_X_vert = ds_X[vars_X_vert]

        # Concat inputs into DataArray and convert to tensor
        if len(vars_X_sfc) > 0:
            da_X = xr.concat([ds_X_sfc.to_array(), ds_X_vert.to_array()], dim="variable")
        else:
            da_X = ds_X_vert.to_array()
        da_X = da_X.transpose("sample", dim_z_X, ...)
        self.X = torch.tensor(da_X.values, dtype=torch.float32, device=device)
        self.vars_X: List[str] = da_X.coords["variable"].values.tolist()
        self.n_vars_X: int = len(self.vars_X)
        self.n_levels_X: int = self.X.shape[1]

        # Forcing variables from y dataset
        self.y_frc = None
        self.vars_y_frc: List[str] = []
        self.n_vars_y_frc: int = 0
        if vars_y_frc:
            da_y_forcing = ds_y[vars_y_frc].to_array().transpose("sample", dim_z_y, ...)
            self.y_frc = torch.tensor(da_y_forcing.values, dtype=torch.float32, device=device)
            self.vars_y_frc = da_y_forcing.coords["variable"].values.tolist()
            self.n_vars_y_frc = len(self.vars_y_frc)

        # Create target tensor if variables were provided, else set to None and keep vars empty
        if vars_y:
            da_y = ds_y[vars_y].to_array().transpose("sample", dim_z_y, ...)
            self.y = torch.tensor(da_y.values, dtype=torch.float32, device=device)
            self.vars_y = da_y.coords["variable"].values.tolist()
            self.n_vars_y: int = len(self.vars_y)
            self.n_levels_y: int = self.y.shape[1]
            self._sample_coord = da_y.coords["sample"]  # Save for unstacking later
        else:
            self.y = None
            self.vars_y = []
            self.n_vars_y = 0
            self.n_levels_y = 0
            self._sample_coord = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y_frc is not None:
            X_frc = self.y_frc[idx]
        else:
            X_frc = [torch.nan]

        if self.y is None:
            y_out = [torch.nan]
        else:
            y_out = self.y[idx]

        return self.X[idx], X_frc, y_out

    def unstack_y(self, y: torch.Tensor, sample_coord: None | Literal["self"] | xr.DataArray) -> xr.Dataset:
        """Unstack y tensor to xarray Dataset here each variable has shape (sample, )."""
        # Create dataarray and convert to dataset
        da = xr.DataArray(
            y.cpu().numpy(),
            dims=["sample", "level", "variable"],
            coords={
                "variable": self.vars_y,
            },
        )
        ds = da.to_dataset(dim="variable")

        # Assign coord if passed, but don't unstack. Discarded NaNs can yield unexpected results.
        if sample_coord is not None:
            if isinstance(sample_coord, xr.DataArray):
                ds = ds.assign_coords(sample=sample_coord)
            elif sample_coord == "self":
                ds = ds.assign_coords(sample=self._sample_coord)

        return ds
