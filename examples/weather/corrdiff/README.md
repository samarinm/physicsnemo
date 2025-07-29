# Downscaling of climate variables

<p align="center">
  <img src="https://cdn.prod.website-files.com/63f1f58039379743bd96333e/67f53a91e36f87e221fe986a_SPEED2ZERO_Figure2.png" width="900" class="center-img" alt="Downscaling of climate variables">
</p>

This is a fork of the [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) library. We've made small adjustments to use NVIDIA's [CorrDiff model](https://arxiv.org/abs/2309.15214) in the [DeepDown](https://www.datascience.ch/projects/deepdown) and [SPEED2ZERO](https://www.datascience.ch/projects/speed2zero) projects of the [Swiss Data Science Center](https://www.datascience.ch/).

Find below a minimal instruction on how to adapt the code to your own datasets. Please check out the [official CorrDiff README](https://github.com/NVIDIA/physicsnemo/tree/main/examples/weather/corrdiff) for a more detailed documentation.

### Docker and Singularity images

The following images are available for running the code in this repository (incorporating the adjustments):

- [Docker image](https://hub.docker.com/repository/docker/samarinm/deepdown_corrdiff/general) 
- [Singularity image](https://polybox.ethz.ch/index.php/s/DatRR8onQyGbwDa)

### How to get started

1. Use the Docker or Singularity image (latest version) to run the code.
2. Copy this repository ([physicsnemo/examples/weather/corrdiff/](https://github.com/samarinm/physicsnemo/tree/main/examples/weather/corrdiff/)) to your local machine or a remote server where you want to run the code.
3. In [datasets](datasets), create your own custom dataloader class. You can use the provided examples like [datasets/eobs_mch.py](datasets/eobs_mch.py) as a template.
4. Add your custom dataset to `known_datasets` in [datasets/dataset.py](datasets/dataset.py#L31-L37).
5. Provide the mean and standard deviation required for standardisation of your dataset as a JSON file in e.g. [datasets/stats/](datasets/stats)`stats_YOUR_DATASET.json`.
6. Specify the details for training CorrDiff in the config files in [conf](conf). You can use files ending in `..._mch.yaml` as templates for your own configuration files. The most important configurations are the following:
   - [conf/base/](conf/base/)`config_training_YOUR_CONF.YAML` (and also `config_generate_....yaml`) specifies (general) configurations for training (or generation), e.g. which dataset to use, training parameters etc.
   - [conf/base/dataset/](conf/base/dataset)`YOUR_CONF.YAML` specifies configurations of the dataset, e.g. which input and output variables should be used.
   - [conf/base/model_size/](conf/base/model_size)`YOUR_CONF.YAML` specfies the model size, e.g. the number of initial channels (`model_channels`), the number of pooling steps in the U-Net and number of channels per level (`channel_mult`) and at which resolution to use self attention layers (`attn_resolutions`). For instance, in the original CorrDiff U-Net architecture (see below [Figure S1 of the CorrDiff paper](https://arxiv.org/pdf/2309.15214#page=22)), these values are set to `model_channels=128`, `channel_mult=[1,2,2,2,2]` and `attn_resolutions=[28]`. With an input image shape of `(448, 448)`and `len(channel_mult)-1=4`pooling steps, this results in a resolution of `(28, 28)` at the bottleneck representation. To use self attention at this resolution, the `attn_resolutions` parameter is required to be set to `[28]`.
     <p align="center">
     <a href="https://arxiv.org/pdf/2309.15214#page=22">
        <img src="https://polybox.ethz.ch/index.php/apps/files_sharing/publicpreview/cZ7boEHxcpWiaCG?file=/&fileId=4180813177&x=1920&y=1200&a=true&etag=b776cf462c183d942df42ffccca09119" width="800" class="center-img" alt="Figure S1 from CorrDiff paper">
     </a>
     </p>
    
     Find the provide examples in [conf/base/model_size/](conf/base/model_size) below.

     | Parameter        | CorrDiff ([normal.yaml](conf/base/model_size/normal.yaml)) | CorrDiff ([mini.yaml](conf/base/model_size/mini.yaml)) | Ours ([normal_MCH.yaml](conf/base/model_size/normal_MCH.yaml)) | Ours ([mini_MCH.yaml](conf/base/model_size/mini_MCH.yaml)) |
     |------------------|------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------|
     | img_shape_y      | 448                                                        | 64                                                     | 240                                                            | 240                                                        |
     | channel_mult     | [1,2,2,2,2]                                                | [1,2,2]                                                | [1,2,2,2,2]                                                    | [1,2,2]                                                    |
     | attn_resolutions | [28]                                                       | [16]                                                   | [15]                                                           | [60]                                                       |
    
     **Note**: The `attn_resolutions` parameter depends on the values of `img_shape_y` and `channel_mult`. It needs be set to the resolution at the bottleneck representation, otherwise self attention layers will not be used. If the representation is not of symmetric shape, use the y-shape of the feature maps in the bottleneck representation.
7. A training examples for the regression model is provided in [deepdown_train_reg.sh](deepdown_train_reg.sh). You will need to adjust the arguments to your config files, data paths etc. 
8. After training the regression model, you can run the diffusion model with an example provided in [deepdown_train_diff.sh](deepdown_train_diff.sh). Note that you need to specify the checkpoint path to the pretrained regression model.