# Scientific Concept Discovery: Using Machine Learning to Advance Scientific Research

## Part 1: Loss Lanscape Visualization

### Overview

A common practice in machine learning is to use three data sets. One for training, one for hyper-parameter tuning, and one for measuring the final performance of the model. However, even following those best practices, we can overestimate how well the model will perform in the real life. The problem is that the world constantly evolves. A model that works on data collected yesterday might not work so well on today's data (cf. [Hand 2006](https://projecteuclid.org/journals/statistical-science/volume-21/issue-1/Classifier-Technology-and-the-Illusion-of-Progress/10.1214/088342306000000060.full)).

Mathematically speaking, the distribution of data shifts in time. We would like to measure how well models generalize to unfamiliar situations. We know that wide minima in the loss function often correspond to states of increased generalizability (cf. e.g., [Chaudhari et al. 2017](https://openreview.net/forum?id=B1YfAfcgl); though, the situation might be not as clear as some see it, cf. e.g., [Dinh 2017](http://proceedings.mlr.press/v70/dinh17b.html)).

Another challenge is that loss functions are highly-dimensional. We present here, how to visualize a 2-dimensional projection of that higher-dimensional space. Next, we examine how various changes to the model or to the training procedure influence the shape of the loss function landscape.

### Main Results

*A short summary of the main results*

### Detailed Reports

See detailed reports in the [loss_landscape](loss_landscape) folder.

## Part 2: Generalization Potential

*A short description.*

*A short summary of the main results*

See detailed reports in the [generalization](generalization) folder.

## Part 3: Neuron Importance

*A short description.*

*A short summary of the main results*

*Link to a detailed report*

## Part 4: Importance-Informed Pruning

*A short description.*

*A short summary of the main results*

*Link to a detailed report*
