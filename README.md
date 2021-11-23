# Topology and geometry of data manifold in deep learning
<br/>

to install the necessary modules copy to the command line
```
pip install -r path/to/requirements.txt
```
_______________________________________
### Quick start experiments: ###
+ --net: Resnet or VGG
+ --mode: Topology (Topological descriptors of data) or PHdim (Persistent Homological fractal dimension)
+ --path 'path/to/tensorflow_model_name.h5'
<br/>

```
python Experiments.py --net VGG --mode PHdim --path 'path/to/model.h5'
```
<br/>

the experimental results in -net VGG case will be approximately as follows 

![plot_7](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_topology.png)
![plot_8](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_phdim.png)

_______________________________________

### Predicting generalization of CNN experiments: ###
Reproduction and verification of experiments on assessing the generalizing ability of CNN by extracting topological descriptors of the training dataset manifold from the internal representation of neural networks. Download the [Resnet models dataset](https://drive.google.com/file/d/1que2h8aQGg6sagtkEdm46vubhHIWDKPr/view?usp=sharing) and copy the following code into the command line.

+ --path: path to dataset dir
+ --homdim: dimension of homology group: 0 or 1

```
python generalization.py --path 'path/to/dir_models_dataset' --homdim 0
```

![plot_9](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Generalization_experiment.png)
![plot_10](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Generalization_experiment1.png)

_______________________________________

### Adversarial manifold experiments: ###
Adversarial examples were generated by the FGSM method using the [Cleverhans](https://github.com/cleverhans-lab/cleverhans) library. In the process of going through all the layers and transforming the data, the topology of the manifold of the dataset changed. It can be seen that the more successful the targeted attack is, the simpler the data topology (color palette from black to white), which is confirmed by experimental results. The experiments were carried out on a our [Dataset](https://drive.google.com/file/d/1epigNlWVSD2i8yIb7488OBIxT6CGj7We/view?usp=sharing) (download it from google drive) of generated images from all categories of buildings in the ImageNet, using the MobileNetV2 model. Different attack success was achieved by changing epsilon. You can reproduce and check the results as follows:

+ --data: path to dataset file_name.pkl
+ --model: path to model_name.h5'
+ --homdim: dimension of homology group: 0 or 1

```
python Adversarial.py --data 'path/to/Buildings_dataset.pkl' --model 'path/to/MobileNetv2_model.h5' --homdim 1
```

![plot_11](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Adversarial_manifold_experiment.png)

_______________________________________

<br/>

to train Resnet model
+ --net: ResNet32, ResNet56, ResNet110 
+ --epochs: epochs for training
+ --path: path to save resnet_model.h5

<br/>

```
python train_model.py --net resnet32 --epochs 100 --path 'path/to/model.h5'
```

_______________________________________
files content:

utils - helper and utility functions
<br/>
<br/>
Topological_descriptors_ID - implementation of methods for calculating the dimension and topological descriptors
<br/>
<br/>
CNN_architectures - architectures of convolutional neural networks for experiments with tracking the dynamics of geometry and topology
<br/>
<br/>
Resnet_train - training resnet with different hyperparameters to predict generalizing ability through topological descriptors
<br/>

------------------------------------------
<br/>
Changing the topological descriptors and PHdim inside ResNet  

![plot_5](https://user-images.githubusercontent.com/94429302/142766610-e1532d60-5985-49a7-8bab-9dad1b77c1d6.png)
<br/>
 
<br/>




