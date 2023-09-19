# activeInstanceSegment 
A repository to benchmark active learning strategies on microscopy data.

## Active Learning
The following active learning strategies are implemented:
- `random:` randomly sample data points from unlabeled pool (used as benchmark)
- `mc_dropout:` sample data points based on uncertainty quantified using Monte Carlo dropout.
- `tta:` sample data points based on uncertainty quantified using test time augmentation.
- `hybrid:` sample data points based on uncertainty quantified using mc dropout and clustering over the datas latent space representation.

The al strategy can be specified in the cfg.yaml in ./pipeline_configs. Benchmarking can be done with
```console
python -m src.al_main -c <path_to_config>
```

## Installation
installation on a ubuntu 22.04 can be done with the following script
```console
$ ./shell_scripts/install.sh
```

## Data 
all used datasets need to follow the [COCO format](https://cocodataset.org/#format-data)
### How to Add a New Dataset
new datasets can be added using the Data2cocoConverter class in utils.datapreprocessing.data2coco 

## Model Architecture
The active learning is built ontop of the [detectron2](https://github.com/facebookresearch/detectron2) implementation of Mask R-CNN. Training a model without active learning can be done by running
```console
python -m src.pipeline_runner -c <path_to_config>
```


## Virtual Enviroment
this project uses a venv which is initailized by running
```console
$ python -m venv ac_acdc_env
```
the venv is activated using the following command

```console
$ source ac_acdc_env/bin/activate 
```
requirements can be installed by running 
```console
$ pip install -r requirements.txt 
```


## Configuration
hyperparameters used for model training, testing and during active learning are specified in configuration.yaml files in the pipeline_config directory. The configuration file containes [detectron2 configurations](https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references) and custom active learning configurations.
### Active Learning
The following active learning configs can be specified

- DATASETS.TRAIN_UNLABELED: name of training dataset used for active learning
- INCREMENT_SIZE: number of data points which are annotated each active learning iteration
- INIT_SIZE: size of initial training dataset
- MAX_LOOPS: maximal number of active learning loops
- NUM_MC_SAMPLES: number of monte carlo samples used for uncertianty estivation
- OBJECT_TO_IMG_AGG: aggregation of object uncertainties to an uncertainty value for the entire image (`mean`, `max`,`min`,`sum`)
- OUTPUT_DIR: path to output directory
- QUERY_STRATEGY: used active learning strategy (`random`, `mc_drop`, `tta`, `hybrid`)
- TTA_MAX_NOISE: max intensity of gaussian noise applied during test-time augmentation (only applied in QUERY_STRATEGY:tta)
- SAMPLE_EVERY: number of images which form the subset used for AL sampling
- MAX_TRAINING_EPOCHS: number of training epochs during active learning
- RETRAIN: flag if model should be retrained from scratch every active learning iteration (only `true` implemented)

## Model Training

## Run Active Learning

## Evaluation of Models
