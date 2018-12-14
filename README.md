# How Real is Real?  Quantitative  and  Qualitative  comparison  of  GANs  andsupervised-learning classifiers.

CS229 Fall 2018 Project - Riccardo Verzeni (rverzeni), Jacqueline Yau (jyau)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Create a virtual environment to set up the prerequisites, and activate it

```
virtualenv -p python3 .env
source .env/bin/activate
```

### Installing

After activating the virtual environment, install the required software:

```
pip install -r requirements.txt 
```

## Running the tests

To run the classifiers, make sure the environment is activated. Then 

```
cd cs229_project
cd classifiers
python [classifier file to test]
```

To see the config for how to run the test:

```
python [classifier file] -h
```

## Authors

* **Riccardo Verzeni**
* **Jacqueline Yau** 

See also the list of [contributors](https://github.com/project-mnist-2018/cs229_project/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Keras-vis attention map tutorials 
* sklearn confusion matrix tutorials
