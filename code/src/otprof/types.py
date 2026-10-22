from __future__ import annotations

from typing import List, Literal, Optional, Tuple

# Tuple for dataset scaler configuration
# Expects (dims, scaler type, additional arguments)
TScalerConfig = Tuple[
    Optional[List[str]],
    Literal["standard", "robust", "minmax", "quantile"],
    Optional[Tuple],
]

TStage = Literal["train", "val", "test"]
