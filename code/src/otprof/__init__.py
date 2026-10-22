"""OTProf: optical-turbulence (Cn2) profiles from reanalysis with a height-aware Squeezeformer (SQF-H).

Paths assume the editable install of the project in `code/` (`uv sync`): datasets live in
`code/dataset/data/<domain>/` (written by the DVC pipeline in `code/dataset/` or downloaded from Zenodo),
trained models and predictions in `code/runs/`.
"""

import pathlib

PKG_ROOT = pathlib.Path(__file__).parent
PROJECT_ROOT = PKG_ROOT.parent.parent  # code/src/otprof -> code/
DATA_ROOT = PROJECT_ROOT / "dataset" / "data"  # per-domain trees data/<domain>/{1_raw, ..., 4_features_interp}
RUNS_ROOT = PROJECT_ROOT / "runs"  # models/<run>/ and preds/<domain>/
