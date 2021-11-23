# Topology and geometry of data manifold in deep learning
<br/>

to install the necessary modules copy to the command line
```
pip install -r path/to/requirements.txt
```
_______________________________________
### Changing the topology across all layers of the CNNs: ###
Basic experiments, tracking the transformation of topological descriptors (Lifespans) and Persistent Homological fractal dimension (PHdim) throughout the depth of convolutional neural networks. The calculations were performed using the [Ripser](https://github.com/scikit-tda/ripser.py) package for fast computation of persistent diagrams for TDA tasks.
+ --net: Resnet or VGG
+ --mode: Topology (Topological descriptors of data) or PHdim (Persistent Homological fractal dimension)
+ --path: 'path/to/tensorflow_model_name.h5'

```
python Experiments.py --net VGG --mode PHdim --path 'path/to/model.h5'
```

the experimental results in -net VGG case will be approximately as follows 

![plot_7](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_topology.png)
![plot_8](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_phdim.png)

_______________________________________

### Predicting generalization ability of CNNs: ###
Reproduction and verification of experiments on assessing the generalizing ability of CNN by extracting topological descriptors of the training dataset manifold from the internal representation of neural networks. Download the [Resnet models dataset](https://drive.google.com/file/d/1que2h8aQGg6sagtkEdm46vubhHIWDKPr/view?usp=sharing) and copy the following code into the command line.

+ --path: path to dataset dir
+ --homdim: dimension of homology group: 0 or 1

```
python generalization.py --path 'path/to/dir_models_dataset' --homdim 0
```

![plot_9](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/generalization_resnet_0.png)
![plot_10](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/generalization_resnet_1.png.png)

_______________________________________

### Adversarial manifold: ###
Adversarial examples were generated by the FGSM method using the [Cleverhans](https://github.com/cleverhans-lab/cleverhans) library. In the process of going through all the layers and transforming the data, the topology of the manifold of the dataset changed. It can be seen that the more successful the targeted attack is, the simpler the data topology (color palette from black to white), which is confirmed by experimental results. The experiments were carried out on a our [Dataset](https://drive.google.com/file/d/1epigNlWVSD2i8yIb7488OBIxT6CGj7We/view?usp=sharing) (download it from google drive) of generated images from all categories of buildings in the ImageNet, using the [MobileNetV2](https://drive.google.com/file/d/19GYB94xN_WWoRvsgJMPAK9mQPqZHSMWU/view?usp=sharing) model. Different attack success was achieved by changing epsilon. You can reproduce and check the results as follows:

+ --data: path to dataset file_name.pkl
+ --model: path to model_name.h5'
+ --homdim: dimension of homology group: 0 or 1

```
python Adversarial.py --data 'path/to/Buildings_dataset.pkl' --model 'path/to/MobileNetv2_model.h5' --homdim 1
```

![plot_11](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Adversarial_manifold_experiment.png)

The [Large Crowdcollected Facial Anti-Spoofing Dataset](https://github.com/IDRnD/LCC_FASD) was used to estimate the Persistent Homological fractal dimension of spoofing attacks and faces.
_______________________________________

### Models training: ###
+ --net: Resnet32, Resnet56, Resnet110 
+ --epochs: epochs for training
+ --path: path to save resnet_model.h5
```
python Resnet_train.py --net resnet32 --epochs 100 --path 'path/to/model.h5'
```
<hr style="border:0.5px solid gray"> </hr>

+ --net: CNN arhitecture: Resnet, VGG, MobileNetV2, SEResnet
+ --epochs: epochs for training
+ --path: path to save tf_model.h5
```
python CNN_architectures.py --net Resnet --epochs 50 --path 'Resnet_name.h5'
```
_______________________________________
# Files content: #

utils - helper and utility functions

Topological_descriptors_ID - implementation of methods for calculating the PHdim and topological descriptors

CNN_architectures - architectures of convolutional neural networks for experiments with tracking the dynamics of geometry and topology

Resnet_train - training resnet with different hyperparameters to predict generalizing ability through topological descriptors

Experiments - running experiments

Adversarial - tracing the topology of adversarial manifold

generalization - predicting the generalizing ability of neural networks using topological descriptors
