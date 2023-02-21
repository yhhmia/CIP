# Client-level Input Perturbation (CIP)
This is the code for paper 'Fortifying Federated Learning against Membership
Inference Attacks via Client-level Input Perturbation'


## Requirements
+ Python3.8
+ Tensorflow 2.6.0
+ Tensorflow Datasets
+ Tensorflow Privacy 0.5.1
+ Scikit-learn
+ tqdm
+ Numpy
+ Pillow
+ OpenCV

## Code Usage

dataLoader.py provides the data.

modelUtil.py provides utilities.

target.py is implementation of baseline one client scenario (target for external adversary and adaptive adversary.)

CIP.py is implementation of one client scenario with our defense (target for external adversary and adaptive adversary.)

federatedTrain.py is implementation of baseline Federated Learning (FedAVG) (target for internal adversary.)

federatedCIP.py is implementation of FedAVG with our defense (target for internal adversary.)

## References
Please refer to the following links for attacks/defenses evaluated in the paper:
+ Diferential Privacy: [code](https://github.com/tensorflow/privacy)
+ Adverserial Regularization: [code](https://github.com/NNToan-apcs/python-DP-DL)
+ MMD+Mix-up: [paper](https://arxiv.org/pdf/2002.12062.pdf)
+ Internal adversary: [paper](https://arxiv.org/pdf/1812.00910.pdf)
+ External adversary: [code](https://github.com/hyhmia/BlindMI)
