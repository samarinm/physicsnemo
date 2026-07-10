# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute per-variable z-score normalization statistics (mean/std), where
each variable is stored in its own set of netCDF files (e.g.
`EOBS/temperature/*.nc`, `MeteoSwiss/RhiresD_v2.0_swiss.lv95/*.nc`).

This script reads netCDF files directly via xarray and has no dependency on
any CorrDiff dataset loader. Each `--input`/`--output` argument is a path
(a directory, a single file, or a glob pattern) holding one variable's
netCDF file(s); the variable name is inferred automatically from the
dataset's data variables. `--input` and `--output` are each optional, but
at least one of them must be given; the corresponding "input"/"output"
group is only computed (and included in the output JSON) if paths for it
are provided.

Pass `--log-transform NAME [NAME ...]` to compute mean/std on log1p(x)
instead of x for the given (inferred) variable name(s) (matching, e.g., the
log1p/expm1 (de)normalization applied to precipitation, "tp", in the
CorrDiff dataset loaders).

Pass `--start-date` / `--end-date` (any format understood by
`xarray.DataArray.sel(time=slice(start, end))`, e.g. "1971-01-01") to
restrict the stats computation to a time window, e.g. the training period.
Either may be omitted for an open-ended slice. Datasets without a "time"
dimension are left untouched.

Pass `--output-coarsen-factor N` to average-pool the output maps over NxN
pixel windows (via `xarray.DataArray.coarsen(..., boundary="trim").mean()`
on every non-"time" dimension) before computing their stats, e.g. to obtain
stats for a coarsened target resolution. Incomplete windows at the edges
are dropped. Only applies to `--output`; `--input` is never coarsened.
Before coarsening, if the last two dimensions are (240, 370) (the MeteoSwiss
output grid), they are first NaN-padded to (240, 416) — 23 pixels on each
side of the last dimension only — so the width divides evenly.

Example
-------

input_path="/mydata/speed2zero/shared/DeepDown/EOBS/v33"
output_path="/mydata/speed2zero/shared/DeepDown/MeteoSwiss"

python compute_stats.py --input "$input_path/temperature" "$input_path/min_temperature" "$input_path/max_temperature" "$input_path/precipitation" --output "$output_path/TabsD_v2.0_swiss.lv95" "$output_path/TminD_v2.0_swiss.lv95" "$output_path/TmaxD_v2.0_swiss.lv95" "$output_path/RhiresD_v2.0_swiss.lv95" --log-transform tp pr RhiresD --start-date 1971-01-01 --end-date 2011-12-31 --out stats_coarse_eobs_mch_v33.json
"""

import argparse
import json
import os

import numpy as np
import xarray as xr

# Data variables that may tag along in a netCDF file besides the actual
# variable of interest, and should not be considered candidates when
# inferring the variable name.
_AUXILIARY_VARIABLES = {"crs", "time_bnds", "swiss_lv95_coordinates"}


def _resolve_files(path):
    """Return a path/glob pattern that `xr.open_mfdataset` can expand."""
    if os.path.isdir(path):
        return os.path.join(path, "*.nc")
    return path


def _infer_variable(ds, path):
    """Infer the single data variable of interest in `ds`."""
    candidates = [v for v in ds.data_vars if v not in _AUXILIARY_VARIABLES]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not infer a single variable for {path!r}; "
            f"found data variables {list(ds.data_vars)}"
        )
    return candidates[0]


def _coarsen(da, factor):
    """Average-pool `da` over `factor`x`factor` windows on every non-'time' dimension."""
    spatial_dims = [d for d in da.dims if d != "time"]
    print(f"  coarsening dims {spatial_dims} by a factor of {factor}")
    coarsen_dims = {d: factor for d in spatial_dims}
    return da.coarsen(coarsen_dims, boundary="trim").mean()


def _pad_last_dim(da, from_shape=(240, 370), pad=23):
    """If da's last two dims are `from_shape`, pad the last dim symmetrically with NaN.

    E.g. (240, 370) -> (240, 370 + 2 * pad), padding `pad` NaN pixels on each side of
    the last dimension only (matching the MeteoSwiss output grid, whose precipitation
    extent is otherwise smaller than the other variables').
    """
    if tuple(da.shape[-2:]) != tuple(from_shape):
        return da
    last_dim = da.dims[-1]
    print(f"  padding dim {last_dim!r} from {from_shape[-1]} to {from_shape[-1] + 2 * pad} with NaN")
    return da.pad({last_dim: (pad, pad)}, mode="constant", constant_values=np.nan)


def compute_variable_stats(
    path, log_transform_names, start_date=None, end_date=None, coarsen_factor=None
):
    """Load one variable's netCDF file(s) and return (name, {"mean", "std"})."""
    print(f"Loading {path!r}")
    with xr.open_mfdataset(_resolve_files(path), combine="by_coords") as ds:
        if (start_date is not None or end_date is not None) and "time" in ds.dims:
            print(f"  selecting time window [{start_date}, {end_date}]")
            ds = ds.sel(time=slice(start_date, end_date))
        variable = _infer_variable(ds, path)
        print(f"  inferred variable {variable!r}")
        da = ds[variable]

        if coarsen_factor is not None:
            da = _pad_last_dim(da)
            print(f"  coarsen (average pooling) by factor {coarsen_factor}")
            da = _coarsen(da, coarsen_factor)
        if variable in log_transform_names:
            print(f"  log1p transforming variable {variable!r} and providing mean/std in log-space")
            da = np.log1p(da)
        mean = float(da.mean(skipna=True).compute())
        std = float(da.std(skipna=True).compute())
    return variable, {"mean": mean, "std": std}


def compute_group_stats(
    paths, log_transform_names, start_date=None, end_date=None, coarsen_factor=None
):
    stats = {}
    for path in paths:
        variable, values = compute_variable_stats(
            path, log_transform_names, start_date, end_date, coarsen_factor
        )
        if variable in stats:
            raise ValueError(f"Variable {variable!r} inferred more than once (path {path!r})")
        stats[variable] = values
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute z-score normalization stats directly from per-variable netCDF files."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Paths (dirs/files/globs) to input variables' netCDF files, one per variable. "
        "Omit to skip computing input stats.",
    )
    parser.add_argument(
        "--output",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Paths (dirs/files/globs) to output variables' netCDF files, one per variable. "
        "Omit to skip computing output stats.",
    )
    parser.add_argument(
        "--log-transform",
        nargs="*",
        default=[],
        metavar="VARIABLE",
        help="Inferred variable name(s) to log1p-transform before computing stats.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        metavar="DATE",
        help="Start of the time window to select via ds.sel(time=slice(start, end)). "
        "Omit for an open start. Ignored for datasets without a 'time' dimension.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        metavar="DATE",
        help="End of the time window to select via ds.sel(time=slice(start, end)). "
        "Omit for an open end. Ignored for datasets without a 'time' dimension.",
    )
    parser.add_argument(
        "--output-coarsen-factor",
        type=int,
        default=None,
        metavar="N",
        help="Average-pool the output maps over NxN pixel windows (dropping incomplete edge "
        "windows) before computing output stats. Only applies to --output.",
    )
    parser.add_argument("--out", required=True, help="Path to write the stats JSON file to.")
    args = parser.parse_args()

    if not args.input and not args.output:
        parser.error("at least one of --input or --output must be given")

    return args


def main():
    args = parse_args()
    log_transform_names = set(args.log_transform)

    stats = {}
    if args.input:
        stats["input"] = compute_group_stats(
            args.input, log_transform_names, args.start_date, args.end_date
        )
    if args.output:
        stats["output"] = compute_group_stats(
            args.output,
            log_transform_names,
            args.start_date,
            args.end_date,
            args.output_coarsen_factor,
        )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=4)

    print(json.dumps(stats, indent=4))
    print(f"\nWrote stats to {args.out}")


if __name__ == "__main__":
    main()
