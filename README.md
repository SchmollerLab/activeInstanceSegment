# ActiveCell-ACDC
A repository to benchmark active learning strategies on microscopy data.

## Active Learning
The following active learning strategies are implemented:
- `random:` randomly sample data points from unlabeled pool (used as benchmark)
- `mc_dropout:` sample data points based on uncertainty quantified using Monte Carlo dropout.
- `tta:` sample data points based on uncertainty quantified using test time augmentation.
- `hybrid:` sample data points based on uncertainty quantified using mc dropout and clustering over the datas latent space representation.

The al strategy can be specified in the cfg.yaml in /pipeline_configs. Benchmarking can be done with
```console
python ./src/al_runner.py
```

## Installation
intsallation can be done running
```console
$ ./shell_scripts/downloadData
```

## Data 
The data can be downloaded using the following command:
```console
$ ./shell_scripts/downloadData
```

First of all the data needs to be converted from Cell-ACDC format to [COCO format](https://cocodataset.org/#format-data). This can be done by running

```console
python ./src/data/data2coco.py
```


## Model Architecture
The active learning is built ontop of the [detectron2](https://github.com/facebookresearch/detectron2) implementation of Mask R-CNN. Training a model without active learning can be done by running
```console
python ./src/pipeline_runner -f default_acdc_large_full_ds.yaml
```

