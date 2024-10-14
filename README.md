# UniPT
This is an official implementation of UniPT, full codes will be released soon
## Getting Started
### Installation

### Requirements
All the codes are tested in the following environment:
* Python = 3.7.16
* CUDA = 11.1
* torch = 1.8.1+cu111
* torch-scatter= 2.0.6
* torchvision = 0.9.1+cu111
* spconv-cu111 = 2.1.25


a. Clone this repository.
```shell
git clone https://github.com/microjie372/UniPT.git
```

b. Install the dependent libraries as follows:

* Install the python dependent libraries.
  ```shell
    pip install -r requirements.txt 
  ```

* Install the gcc library, we use the gcc-5.4 version

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * It is recommended that you should install the latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
    * Also, you should choice **the right version of spconv**, according to **your CUDA version**. For example, for CUDA 11.1, pip install spconv-cu111
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```
