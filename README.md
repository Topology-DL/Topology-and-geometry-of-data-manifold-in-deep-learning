# Topology-and-geometry-of-data-manifold-in-deep-learning
<br/>

to install libraris

```
pip install -r requirements.txt  
```
### Quick start experiments ###
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
Changing the topological descriptors and PHdim inside ResNet 
![plot_7](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/figures/VGG_example_topology.png)
<br/>

adversarial manifold 

<br/>

```
python adversarial.py --path 'path/to/model.h5'
```

<br/>

to train model

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
Changing the topological descriptors and PHdim inside ResNet  
![plot_4](https://github.com/Topology-DL/Topology-and-geometry-of-data-manifold-in-deep-learning/blob/main/Topology%20and%20PHdim.png)
<br/>


