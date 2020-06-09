> 📋A template README.md for code accompanying a Machine Learning paper

# Path Integral Based Convolution and Pooling for Graph Neural Networks

This repository is the official implementation of [Path Integral Based Convolution and Pooling for Graph Neural Networks](https://arxiv.org/abs/). 

> 📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training and Evaluation

To train and test the model(s) in the paper, run the following command. We provide the codes for PAN on three benchmarks in Table 1 and the new dataset PointPattern. The dataset will be downloaded and preprocessed before training. 

```PANConv+PANPool on MUTAG
python train.py --input-data

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
