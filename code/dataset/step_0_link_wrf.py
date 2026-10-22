"""Link the post-processed WRF output of a domain (`wrfout*_cn2.nc` in <wrf-dir>/sim_*/4_postproc/, see
data/<domain>/wrf/ of this repository) into 1_raw/wrf_cn2/. Run from inside dataset/data/<domain>/ (see
dvc.yaml). Existing links are replaced."""

import argparse
import pathlib

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf-dir", type=pathlib.Path, required=True)
    wrf_dir = ap.parse_args().wrf_dir.resolve()

    files = sorted(wrf_dir.glob("sim_*/4_postproc/wrfout*_cn2.nc"))
    if not files:
        raise FileNotFoundError(f"No post-processed WRF output (sim_*/4_postproc/wrfout*_cn2.nc) in {wrf_dir}")

    out = pathlib.Path("1_raw/wrf_cn2")
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        link = out / f.name
        link.unlink(missing_ok=True)
        link.symlink_to(f)
    print(f"Linked {len(files)} WRF files from {wrf_dir} into {out}")
