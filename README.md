# segmentationMicroscopy
This repo contains the first steps towards my master thesis

## Data 
The data can be downloaded using the following command:
```console
$ ./downloadData

https://hmgubox2.helmholtz-muenchen.de/index.php/s/DdXYAam2mRwZn88
```

First of all the data needs to be converted from Cell-ACDC format to [COCO format](https://cocodataset.org/#format-data). This can be done by running

```python
python ./src/data/data2coco.py
```

## Links

- data2coco tutorial: https://patrickwasp.com/create-your-own-coco-style-dataset/
- detectron2
    - repo: https://github.com/facebookresearch/detectron2
    - docs: https://detectron2.readthedocs.io/en/latest/index.html 
