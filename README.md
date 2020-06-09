
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

```PAN on MUTAG
python pan_mutag.py
```
```PAN on PROTEINSF
python pan_proteinsf.py
```
```PAN on NCI1
python pan_nci1.py
```
```PAN on PointPattern
python pan_pointpattern_phi035.py
```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.


> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results

Our model PAN achieves the following performance on graph classification benchmark datasets MUTAG, PROTEINSF and NCI1, and our new graph classification dataset PointPattern (with phi=0.35). The table below shows the mean test accuracy with SD for 10 repetitions. Compared to existing methods such as GCN ... with the same network architecture, the PAN achieves top test accuracy on all these datasets. The results are obtained using the above .py programs.

### [PAN on Graph Classification Datasets]()

| Model name         |   MUTAG         |   PROTEINSF     |   NCI1       |   PointPattern, \phi=0.35 |
| ------------------ |---------------- | --------------- |--------------|---------------------------|
|     PAN            |     85% (0.5%)  |      95% (0.5%) |              |                           |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
