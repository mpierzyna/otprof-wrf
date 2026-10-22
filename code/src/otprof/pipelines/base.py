from __future__ import annotations

import abc
import pathlib
from typing import Any, Callable, Dict, List, Protocol, Self, Tuple, TypeVar, runtime_checkable

import joblib
import pydantic
import xarray as xr
from research_tools.misc_tools.yaml_config import dict_to_yaml

from otprof import DATA_ROOT
from otprof.logging import LogContext, get_logger
from otprof.transform import DatasetScaler
from otprof.types import TScalerConfig, TStage

logger = get_logger(context=LogContext.data)

# optional xarray.Dataset
TDatasetOptX = TypeVar("TDatasetOptX", xr.Dataset, None)
TDatasetOpty = TypeVar("TDatasetOpty", xr.Dataset, None)


@runtime_checkable
class FeatureEngineeringFn(Protocol):
    def __call__(self, ds_X: xr.Dataset, ds_y: xr.Dataset) -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
        """Takes X and y datasets as input and returns dicts of new features for X and y."""


class BaseDataPipeline(pydantic.BaseModel, abc.ABC):
    """Implement all steps to prepare dataset for training.
    Most importantly, implements how data is split and scaled.
    """

    # WRF domain (geographic region) to read from. Selects the data/<domain>/ subtree.
    domain: str

    # Train/val/test split selectors for xarray.sel() method
    # If callable, it receives the pipeline and has to return a dict with selectors.
    sel_train: Dict[str, Any] | Callable[[BaseDataPipeline], Dict[str, Any]]
    sel_val: Dict[str, Any] | Callable[[BaseDataPipeline], Dict[str, Any]]
    sel_test: Dict[str, Any] | Callable[[BaseDataPipeline], Dict[str, Any]]

    # Variables for scalers
    vars_X: List[str]  # low-res vars for low-res scaler
    vars_y: List[str]  # high-res vars for high-res scaler

    # Feature engineering functions
    fe_fns: List[FeatureEngineeringFn] = []

    # Scaler configs
    scaler_config_X: Dict[str, TScalerConfig]
    scaler_config_y: Dict[str, TScalerConfig]

    # Optional transformer configs
    tf_config_X: Dict[str, Tuple[Callable, Callable]] | None = None
    tf_config_y: Dict[str, Tuple[Callable, Callable]] | None = None

    # Datasets, scalers, and transformers are set up through methods. Don't pass them.
    ds_X: xr.Dataset | None = None
    ds_y: xr.Dataset | None = None
    scaler_X: DatasetScaler | None = None  # for X and y_frc variables
    scaler_y: DatasetScaler | None = None  # for y variables only

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @property
    def data_root(self) -> pathlib.Path:
        """Data tree of the selected domain. Derived from `domain` so the two cannot drift."""
        return DATA_ROOT / self.domain

    @property
    def is_setup(self) -> bool:
        """Check if all necessary steps have been run"""
        return not any([self.ds_X is None, self.ds_y is None, self.scaler_X is None, self.scaler_y is None])

    @abc.abstractmethod
    def load_data(self) -> None:
        """Implement how dataset is loaded from disk."""

    def fill_gaps(self) -> None:
        """Implement how missing values are filled. Return gap-filled datasets."""

    def make_features(self) -> None:
        """Create additional features in ds_X and ds_y using feature engineering functions."""

        def _check_nan(fe_dict: Dict[str, xr.DataArray], ds_name: str) -> None:
            for v, da in fe_dict.items():
                if da.isnull().any():
                    logger.warning(f"Feature engineering produced NaNs in {ds_name} variable '{v}'")

        if self.ds_X is None or self.ds_y is None:
            raise ValueError("Datasets not loaded. Cannot make features.")

        for fe_fn in self.fe_fns:
            logger.info(f"Applying feature engineering function: {fe_fn.__name__}")
            X_fe, y_fe = fe_fn(self.ds_X, self.ds_y)

            _check_nan(X_fe, "X")
            _check_nan(y_fe, "y")

            X_fe_vars = list(X_fe.keys())
            if X_fe_vars:
                logger.info(f"Adding X features: {X_fe_vars}.")
                self.ds_X = xr.merge([self.ds_X, xr.Dataset(X_fe)], compat="override")
                # Rebind (not in-place `+=`) and dedup so make_features stays idempotent and never
                # mutates an aliased list: `model_copy(update={"vars_X": ...})` stores the passed
                # object without copying (would corrupt a module-level vars constant), and a pipeline
                # rebuilt from a saved config already lists these features in vars_X.
                self.vars_X = self.vars_X + [v for v in X_fe_vars if v not in self.vars_X]

            y_fe_vars = list(y_fe.keys())
            if y_fe_vars:
                logger.info(f"Adding y features: {y_fe_vars}.")
                self.ds_y = xr.merge([self.ds_y, xr.Dataset(y_fe)], compat="override")
                self.vars_y = self.vars_y + [v for v in y_fe_vars if v not in self.vars_y]

        # Only select specified variables
        self.ds_X = self.ds_X[self.vars_X]
        self.ds_y = self.ds_y[self.vars_y]

    def fit_scalers(self, force: bool = False) -> None:
        """Fit X and y scalers on training data"""
        if not force and (self.scaler_X is not None or self.scaler_y is not None):
            # OOD domains load the trained scalers (via load_ood_scalers) instead of fitting on the
            # new domain. Skip rather than refit so a loaded scaler is never silently discarded.
            if self.scaler_X is None or self.scaler_y is None:
                logger.warning(
                    "Only one of scaler_X/scaler_y is set; skipping fit to avoid discarding the "
                    "loaded scaler (the missing one stays None). Pass force=True to refit both."
                )
            else:
                logger.info("Scalers already set (e.g. loaded for OOD inference). Skipping fit.")
            return

        # Get training data for fitting
        # for legacy reasons, the transform flag of split only allows scaling and transforming together
        # so we first split without transforming, then transform without scaling, then fit the scalers
        logger.info("Fitting low-res and high-res scalers on training data.")
        ds_X_train, ds_y_train = self.split(stage="train", transform=False)
        ds_X_train, ds_y_train = self.transform(ds_X=ds_X_train, ds_y=ds_y_train, transform=True, scale=False)
        self.scaler_X = DatasetScaler(self.scaler_config_X).fit(ds_X_train[self.vars_X])
        self.scaler_y = DatasetScaler(self.scaler_config_y).fit(ds_y_train[self.vars_y])

    def setup(self) -> Self:
        """Run all setup steps"""
        logger.info(f"This is {self.__class__.__name__}. Starting setup.")
        self.load_data()
        self.fill_gaps()
        self.make_features()
        self.fit_scalers()
        logger.info("Data pipeline setup complete.")

        return self

    def split(self, stage: TStage, transform: bool) -> Tuple[xr.Dataset, xr.Dataset]:
        """Split low-res and high-res data into training, validation, test."""
        # Get selector based on stage
        if stage == "train":
            sel = self.sel_train(self) if callable(self.sel_train) else self.sel_train
        elif stage == "val":
            sel = self.sel_val(self) if callable(self.sel_val) else self.sel_val
        elif stage == "test":
            sel = self.sel_test(self) if callable(self.sel_test) else self.sel_test
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Select X and y data
        ds_X = self.ds_X.sel(**sel)
        ds_y = self.ds_y.sel(**sel)

        # Transform if needed (always scale and transform together)
        if transform:
            ds_X, ds_y = self.transform(ds_X=ds_X, ds_y=ds_y)

        return ds_X, ds_y

    @staticmethod
    def _apply_tf(ds: xr.Dataset, config: Dict[str, Tuple[Callable, Callable]], inverse: bool) -> xr.Dataset:
        """Apply transformations to dataset"""
        ds = ds.copy()
        for v, (fn, fn_inv) in config.items():
            if v in ds:
                logger.info(f"{'Inverting' if inverse else 'Applying'} transformation for variable '{v}'")
                fn = fn_inv if inverse else fn
                ds[v] = fn(ds[v])  # assume that fn works on xarray.DataArray
        return ds

    def transform(
        self,
        *,
        ds_X: TDatasetOptX = None,
        ds_y: TDatasetOpty = None,
        transform: bool = True,
        scale: bool = True,
    ) -> Tuple[TDatasetOptX, TDatasetOpty]:
        """Transform low-res and/or high-res data"""

        # Transform and scale X
        if transform and self.tf_config_X and (ds_X is not None):
            ds_X = self._apply_tf(ds_X, self.tf_config_X, inverse=False)
        if scale and (ds_X is not None):
            ds_X = self.scaler_X.transform(ds_X)

        # Transform and scale y
        if transform and self.tf_config_y and (ds_y is not None):
            ds_y = self._apply_tf(ds_y, self.tf_config_y, inverse=False)
        if scale and (ds_y is not None):
            ds_y = self.scaler_y.transform(ds_y)

        return ds_X, ds_y

    def inverse_transform(
        self,
        *,
        ds_X: TDatasetOptX = None,
        ds_y: TDatasetOpty = None,
        transform: bool = True,
        scale: bool = True,
    ) -> Tuple[TDatasetOptX, TDatasetOpty]:
        """Inverse transform data"""
        # Invert scaling of X
        if scale and (ds_X is not None):
            ds_X = self.scaler_X.inverse_transform(ds_X)

        # Invert transformations of X
        if transform and self.tf_config_X and (ds_X is not None):
            ds_X = self._apply_tf(ds_X, self.tf_config_X, inverse=True)
        elif not transform and self.tf_config_X and (ds_X is not None):
            logger.warning("Not inverting transformations of X even though transformers are defined.")

        # Invert scaling of y
        if scale and (ds_y is not None):
            ds_y = self.scaler_y.inverse_transform(ds_y)

        # Invert transformations of y
        if transform and self.tf_config_y and (ds_y is not None):
            ds_y = self._apply_tf(ds_y, self.tf_config_y, inverse=True)
        elif not transform and self.tf_config_y and (ds_y is not None):
            logger.warning("Not inverting transformations of y even though transformers are defined.")

        return ds_X, ds_y

    def to_disk(self, dp_dir: str | pathlib.Path) -> None:
        """Save data pipeline to disk, but without data"""
        dp_dir = pathlib.Path(dp_dir)
        dp_dir.mkdir(parents=True, exist_ok=True)

        # Record the pipeline class ("module:qualname") with the config
        class_path = f"{self.__class__.__module__}:{self.__class__.__qualname__}"
        (dp_dir / "type.txt").write_text(class_path)

        # Save config without data or scaler
        self_dict = self.model_dump(exclude={"ds_X": ..., "ds_y": ..., "scaler_X": ..., "scaler_y": ...})
        (dp_dir / "config.yml").write_text(dict_to_yaml(self_dict))

        # Save fitted scalers
        joblib.dump(self.scaler_X, dp_dir / "scaler_X.joblib")
        joblib.dump(self.scaler_y, dp_dir / "scaler_y.joblib")

    def load_ood_scalers(self, ood_dp_dir: str | pathlib.Path) -> Self:
        """Load out-of-distribution scalers (from another pipeline's `to_disk` method).

        Needed to run a trained model on a domain it was not trained on: the scalers have to stay the ones seen during
        training, otherwise inputs are normalized against the new domain's statistics.

        Run before calling setup(). Scaler fitting will be skipped.
        """
        ood_dp_dir = pathlib.Path(ood_dp_dir)
        self.scaler_X = joblib.load(ood_dp_dir / "scaler_X.joblib")
        self.scaler_y = joblib.load(ood_dp_dir / "scaler_y.joblib")
        return self
