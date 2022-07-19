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
