# CiteULike-PyTorch
Neural Network model for the CiteULike dataset using PyTorch. The model utilizes negative sampling, as the dataset only contains positive samples.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
A simple model that can be used with the CiteULike dataset. It achieves roughly 90% accuracy after 15 epochs. As the original dataset only contains positive samples, the data generation adds an equal amount of random negative samples for every user, meaning that if user A has interacted with 20 articles, 20 random articles are drawn and put in the dataset as negative samples. Validation set only contains positive samples for obvious reasons.

## Installation
To run this program you simply need a python 3.x (that works with PyTorch).

### Python
    pip install -r requirements.txt


### Dataset
The dataset is included in the ``dataset`` folder, however you need to generate the training and validation set first.
Do the following in a python console:

    import pandas as pd
    import numpy as np
    from random import randint
    import os
    dataset = os.path.join('dataset')
    from data import to_csv_citeulike
    to_csv_citeulike()

## Usage
    python main.py

You can play around with the hyperparameters in the ``main.py`` or the model in ``model.py``

## License
The package is Open Source Software released under the [MIT](LICENSE) license.
