# Data loader for TWC MVP: GEFS and HRRR forecasts
# adapted from https://gitlab-master.nvidia.com/earth-2/corrdiff-internal/-/blob/dpruitt/hrrr/explore/dpruitt/hrrr/datasets/hrrr.py

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

import datetime
import math
from typing import List, Tuple, Union

import json
import numpy as np
from numba import jit, prange
import xarray as xr

from physicsnemo.utils.diffusion import convert_datetime_to_cftime

from datasets.base import ChannelMetadata, DownscalingDataset

import os
import cftime
import copy
import time
from IPython import embed

from physicsnemo.launch.logging import PythonLogger

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from deepdown_simple_data_loader import load_target_data, load_data

dataloader_logging = PythonLogger("dataloader")


class era5MchDataset(DownscalingDataset):
    """Custom reader for ERA5 - MeteoSwiss data pairs over Switzerland."""

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        period: str = 'train-period',
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = ("elev_mean", "lsm_mean"),
        tmp_path: str = './tmp'
    ):
        # load data
        (self.input, self.input_variables, self.times) = _load_dataset(
            data_path=data_path,
            group="input",
            variables=input_variables,
            period=period,
            stack_axis=1,
            tmp_path=tmp_path,
        )

        (self.output, self.output_variables, _) = _load_dataset(
            data_path=data_path,
            group="output",
            variables=output_variables,
            period=period,
            stack_axis=1,
            tmp_path=tmp_path,
        )

        dataloader_logging.info(f"\n{period}\n{self.input.shape = }\n{self.times.shape = }\n{self.output.shape = }\n")

        # Todo for invariant variables: Loading

        self.img_shape = self.output.shape[-2:]
        # self.upsample_factor = self.output.shape[-1] // self.input.shape[-1]
        # Choosing y dimension for convenience in the ERA5 setting.
        self.upsample_factor = self.output.shape[-2] // self.input.shape[-2]

        # load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)

        (self.input_mean, self.input_std) = _load_stats(stats, self.input_variables, "input")

        # Todo for invariant variables: Include for invariant variables
        # (input_mean, input_std) = _load_stats(stats, self.input_variables, "input")
        # (inv_mean, inv_std) = _load_stats(stats, self.invariant_variables, "invariant")
        # self.input_mean = np.concatenate([input_mean, inv_mean], axis=0)
        # self.input_std = np.concatenate([input_std, inv_std], axis=0)

        (self.output_mean, self.output_std) = _load_stats(
            stats, self.output_variables, "output"
        )

    def __getitem__(self, idx):
        """Return the data sample (output, input) at index idx."""
        x = self.upsample(self.input[idx].copy())

        # Cropping upsampled input by for pixels to have shape (240, 416) instead of (240, 420).
        x = x[..., 2:-2]

        # Todo for invariant variables: Add to input
        # (i, j) = self.coords[idx]
        # inv = self.invariants[:, i : i + self.img_shape[0], j : j + self.img_shape[1]]
        # x = np.concatenate([x, inv], axis=0)

        y = self.output[idx]

        x = self.normalize_input(x)
        y = self.normalize_output(y)
        return (y, x)

    def __len__(self):
        return self.input.shape[0]

    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        return np.full(self.img_shape, np.nan)

    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        return np.full(self.img_shape, np.nan)

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        inputs = [ChannelMetadata(name=v) for v in self.input_variables]
        return inputs

        # Todo for invariant variables: Add to input channels
        # invariants = [
        #     ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables
        # ]
        # return inputs + invariants

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        """Get time values from the dataset."""
        datetimes = (
            datetime.datetime.utcfromtimestamp(t.tolist() / 1e9) for t in self.times
        )
        return [convert_datetime_to_cftime(t) for t in datetimes]

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean

    def upsample(self, x):
        """Extend x around edges with linear extrapolation."""
        y_shape = (
            x.shape[0],
            x.shape[1] * self.upsample_factor,
            x.shape[2] * self.upsample_factor,
        )
        y = np.empty(y_shape, dtype=np.float32)
        _zoom_extrapolate(x, y, self.upsample_factor)
        return y


def _load_dataset(data_path, group, variables, period, stack_axis=1, tmp_path='./tmp'):

    dataloader_logging.info(f"{data_path = }, {group = }, {variables = }, {period = }, "
                            f"{stack_axis = }, {tmp_path = }\n")

    if period == 'train-period':
        years = ['1971', '2011']
    elif period == 'val-period':
        years = ['2012', '2017']
    elif period == 'test-period':
        years = ['2018', '2023']
    elif period == 'full-period':
        years = ['1971', '2023']
    else:
        raise ValueError(f"Period {period} not recognised")

    # High resolution coordinates to ensure a 240x370 matrix for the MCH dataset.
    # This is necessary because precipitation data can come with a larger extent (265x370).
    N_min = 1064500.0
    N_max = 1303500.0

    # Low resolution coordinates to ensure a 12x21 matrix for the ERA5 dataset
    # (necessary for current implementation of data loader; same as E-OBS)
    lon_min = 5.6
    lon_max = 10.8
    lat_min = 45.3
    lat_max = 48.3

    data = np.array([])
    times = np.array([])

    if group == 'input':
        # Load ERA5 data
        for t_enum, target in enumerate(variables):
            if target == 't':
                data_dir_LR = [os.path.join(data_path, 'ERA5', 'temperature')]
            elif target == 'tp':
                data_dir_LR = [os.path.join(data_path, 'ERA5', 'precipitation')]

            dataloader_logging.info(f"Loading {data_dir_LR}.")

            # Bounding box for selecting shape (12,21) in ERA5
            # (necessary for current implementation of data loader)
            lon_bnds = [lon_min, lon_max]
            lat_bnds = [lat_min, lat_max]

            data_era5 = load_data(paths=data_dir_LR, date_start=years[0], date_end=years[-1],
                                lon_bnds=lon_bnds, lat_bnds=lat_bnds, levels=[])

            data_LR = data_era5[target]

            # Flip latitude coordinates to range from large to small values
            # Todo: In first training runs, this flip was erroneously applied. Undo for next runs.
            data_LR = np.flip(data_LR, 1)

            if target == 't':
                data_LR -= 273.15
            elif target == 'tp':
                data_LR *= 86400

            if t_enum == 0:
                data = copy.deepcopy(data_LR.values)
                data = np.expand_dims(data, axis=stack_axis)

                # Store the different times in a separate array.
                # This has to be done only for the first variable, as it should
                # be the same for all variables.
                times = data_LR.indexes['time'].strftime('%Y-%m-%d')
                times = np.array(times, dtype=str)
                times = np.char.add(times, 'T12:00:00.000000000')
                times = times.astype('datetime64[ns]')
            else:
                data = np.concatenate([data,
                                       np.expand_dims(data_LR.values, axis=stack_axis)],
                                      axis=stack_axis)

            dataloader_logging.info(f"{t_enum = }, {data.shape = }\n")

    elif group == 'output':
        # Load MeteoSwiss data
        for t_enum, target in enumerate(variables):

            if target == 't':
                data_dir_HR = [os.path.join(data_path, 'MeteoSwiss', 'TabsD_v2.0_swiss.lv95')]
            elif target == 'tp':
                data_dir_HR = [os.path.join(data_path, 'MeteoSwiss', 'RhiresD_v2.0_swiss.lv95')]
            elif target == 't_min':
                data_dir_HR = [os.path.join(data_path, 'MeteoSwiss', 'TminD_v2.0_swiss.lv95')]
            elif target == 't_max':
                data_dir_HR = [os.path.join(data_path, 'MeteoSwiss', 'TmaxD_v2.0_swiss.lv95')]

            # Provide a print statement indicating the date and time
            dataloader_logging.info(f"Loading {data_dir_HR}.")

            data_mch = load_target_data(date_start=years[0], date_end=years[-1],
                                        paths=data_dir_HR, path_tmp=tmp_path)

            # Selecting consistent shape (240,370)
            data_HR = data_mch[target].sel(y=slice(N_min, N_max))

            # Flip latitude coordinates to range from large to small values
            data_HR = np.flip(data_HR, 1)

            if t_enum == 0:
                data = copy.deepcopy(data_HR.values)
                data = np.expand_dims(data, axis=stack_axis)
            else:
                data = np.concatenate([data,
                                       np.expand_dims(data_HR.values, axis=stack_axis)],
                                      axis=stack_axis)

            dataloader_logging.info(f"{t_enum = }, {data.shape = }\n")

        # Add padding to ensure the output data has the correct shape

        # Pad width (before, after) for each dimension, padding only the last dimension
        pad_width = [(0, 0), (0, 0), (0, 0), (23, 23)]

        data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=np.nan)

        dataloader_logging.info(f"After padding with np.nan: {data.shape = }\n")

    elif group == 'invariant':
        # Todo for invariant variables: Loading
        pass

    if np.any(np.isnan(data)):
        dataloader_logging.info(f"NaN values are set to 0"
                                f" if any pixel for any variables is NaN!\n")
        # Extract the mask of NaN values -> include if any pixel for any variables is NaN
        nanmask = np.any(np.isnan(data), axis=(0,1)).astype(bool)
        data[:,:,nanmask] = 0.0

        # dataloader_logging.info(f"NaN values in the {group} data are set to 0!")
        # data = np.nan_to_num(data, nan=0.0)

    return (data, variables, times)


def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return (mean, std)


@jit(nopython=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation.
    Use a numba function here because numpy/scipy options are rather slow.
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)