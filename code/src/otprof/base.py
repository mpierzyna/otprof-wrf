from __future__ import annotations

import abc
import pathlib
from typing import List

import pydantic
import lightning as L
import torch.utils.data
from typing_extensions import Self

from otprof.logging import get_logger, LogContext
from otprof.pipelines.base import BaseDataPipeline
from otprof.types import TStage
from research_tools.misc_tools.yaml_config import dict_to_yaml

logger = get_logger()
logger_train = get_logger(context=LogContext.train)
logger_eval = get_logger(context=LogContext.eval)


class BaseExperiment(pydantic.BaseModel, abc.ABC):
    # Experiment settings
    name: str
    exp_dir: pathlib.Path | None = None

    # Data config
    dp: BaseDataPipeline
    features: List[str]  # features from X
    targets: List[str]  # targets from y
    forcings: List[str] = []  # forcing from y added to X

    # Training config
    batch_size: int
    max_epochs: int

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,  # Allow, e.g., numpy arrays or custom types
    )

    @pydantic.model_validator(mode="after")
    def validate_features_targets(self) -> Self:
        """Check that features and targets are in the data pipeline"""
        for f in self.features:
            if f not in self.dp.vars_X:
                raise ValueError(f"Feature {f} not found in data pipeline X variables.")
        for t in self.targets:
            if t not in self.dp.vars_y:
                raise ValueError(f"Target {t} not found in data pipeline y variables.")
        for f in self.forcings:
            if f not in self.dp.vars_y:
                raise ValueError(f"Forcing {f} not found in data pipeline y variables.")
        return self

    @pydantic.field_validator("dp")
    @classmethod
    def validate_dp_setup(cls, dp: BaseDataPipeline) -> BaseDataPipeline:
        """Check that data pipeline setup has been run"""
        if not dp.is_setup:
            dp.setup()
        return dp

    @abc.abstractmethod
    def get_torch_dataset(self, stage: TStage) -> torch.utils.data.Dataset:
        """Implement how to create a torch dataset for train/val/test"""

    @abc.abstractmethod
    def get_torch_loader(
        self,
        stage: TStage,
        data: torch.utils.data.Dataset | None = None,
    ) -> torch.utils.data.DataLoader:
        """Implement how torch data loaders for training, validation, testing are created.
        Optionally, preloaded dataset can be passed via `data` argument.
        """

    @abc.abstractmethod
    def get_lit_model(self, data_train: torch.utils.data.Dataset) -> L.LightningModule:
        """Implement how Lightning model is created"""

    def to_disk(self, exp_dir: str | pathlib.Path):
        logger.info(f"Saving experiment to {exp_dir}.")
        exp_dir = pathlib.Path(exp_dir)
        exp_dir.parent.mkdir(parents=True, exist_ok=True)
        self.exp_dir = exp_dir

        # Save data pipeline separately
        dp_path = exp_dir / "data_pipeline"
        self.dp.to_disk(dp_path)

        # Save experiment config
        self_dict = self.model_dump(exclude={"exp_dir": ..., "dp": ...})
        self_dict["dp_path"] = dp_path.name  # only relative path
        self_dict["dp_cls"] = f"{self.dp.__class__.__module__}:{self.dp.__class__.__qualname__}"

        # Record the experiment class ("module:qualname")
        self_dict["exp_cls"] = f"{self.__class__.__module__}:{self.__class__.__qualname__}"
        (exp_dir / "exp_config.yml").write_text(dict_to_yaml(self_dict))
