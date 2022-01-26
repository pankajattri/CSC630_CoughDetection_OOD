Requirements:

```
python==3.8.5
pytorch==1.7.1
torchaudio==0.7.2
pandas==1.1.3
```

This branch is structured as follows:

* `data` contains all the data needed for training, validation, and testing. However, this data must be downloaded first. See `data/README.md` for details.
* `models` contains definitions of the (F)eature (E)xtraction (Net)work and the overall discriminator network. The FENet architecture is described in `models/FENet.py`, and in the paper [Knowledge Transfer from Weakly Labeled Audio using Convolutional Neural Network for Sound Events and Scenes](https://arxiv.org/abs/1711.01369).
* `parameters` contains `.pt` files that store the pre-trained parameters for the discriminator network.
* `results/test` contains testing results in `csv` format.
* `torch_datasets` contains definitions of PyTorch dataset classes.

To train the discriminator network, run `train.py`. To test the discriminator network, run `test.py`, which will generate a `csv` file with testing results in the `results/test` folder. See the `parameters/disc` folder for details on how to download a sample `.pt` file to run `test.py`.
