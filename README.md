# Topology and geometry of data manifold in deep learning
<br/>

to install libraris

```
pip install -r requirements.txt  
```
------------------------------------------
### Quick start experiments: ###
+ --net: Resnet or VGG
+ --mode: Topology (Topological descriptors of data) or PHdim (Persistent Homological fractal dimension)
+ --path 'path/to/tensorflow_model_name.h5'
<br/>

copy to command line this code

<br/>

```
python Experiments.py --net VGG --mode PHdim --path 'path/to/model.h5'
```
<br/>

the experimental results in -net VGG case will be approximately as follows 

<br/>

![plot_7](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_topology.png)
![plot_8](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_phdim.png)

------------------------------------------

<br/>
Reproduction and verification of experiments on assessing the generalizing ability of CNN by extracting topological descriptors of the training dataset manifold from the internal representation of neural networks. Download the [Resnet models dataset](https://drive.google.com/file/d/1que2h8aQGg6sagtkEdm46vubhHIWDKPr/view?usp=sharing) and copy the following code into the command line.
<br/>

### Predicting generalization of CNN experiments: ###
+ --path: path to dataset dir
+ --homdim: Dimension of homology group: 0 or 1

```
python generalization.py --path 'path/to/dir_models_dataset' --homdim 0
```

<br/>

![plot_9](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Generalization_experiment.png)
![plot_10](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Generalization_experiment1.png)

------------------------------------------

### Adversarial manifold experiments: ###
+ --data: path to dataset file_name.pkl
+ --model: path to model_name.h5'
+ --homdim: Dimension of homology group: 0 or 1

<br/>

```
python Adversarial.py --data 'path/to/Buildings_dataset.pkl' --model 'path/to/MobileNetv2_model.h5' --homdim 1
```

<br/>

![plot_11](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/Adversarial_manifold_experiment.png)

------------------------------------------

<br/>

to train Resnet model
+ --net: resnet32, resnet56, resnet110 
+ --epochs: epochs for training
+ --path: path to save resnet_model.h5

<br/>

```
python train_model.py --net resnet32 --epochs 100 --path 'path/to/model.h5'
```

------------------------------------------
files content:
-------------------------
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




