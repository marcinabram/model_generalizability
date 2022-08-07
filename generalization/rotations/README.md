# Generalization to Rotations

By Saurabh Koshatwar et al. (2022)

## Motivation

Study the generalization of deep learning models to rotation and approaches to improve it.

## Overview

In this study, we used ResNet-50 to train on the CIFAR dataset in order to test generalization to rotations, with the source distribution being the original images and the target distribution being the modified dataset produced by rotating the images at specific angles. We compared the outcomes using the L1 regularized, L2 regularized, and dropout ResNet-50 model. performed some experiments to support the findings, and then compared the results with the state of the art Equivariant CNN model.

## Main Results

![Dummy Plot](pics/dummy_plot.png)

## Discussion

...

## Detailed Reports

 1. [Generalization to Rotations Experiment](reports/generalization_to_rotations.md) (Markdown Report).
 2. [Day-Night Experiment](reports/day_night_experiment.ipynb) (Jupyter Notebook).
 4. [Equivariant Neural Networks - Minimal Example](reports/equivariant_neural_networks.ipynb) (Jupyter Notebook).
