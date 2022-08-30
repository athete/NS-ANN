# Inferring the Dense Matter Equation of State from Neutron Star Observations via Artificial Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2208.13163-b31b1b.svg?style=flat)](https://arxiv.org/abs/2208.13163)

<b> Ameya Thete, Kinjal Banerjee, and Tuhin Malik </b>

**Abstract:** The difficulty in describing the equation of state (EoS) for nuclear matter at densities above the saturation density $(\rho_0)$ has led to the emergence of a multitude of models based on different assumptions and techniques. These EoSs, when used to describe a neutron star (NS), lead to differing values of observables. An outstanding goal in astrophysics is to constrain the dense matter EoS by exploiting astrophysical and gravitational wave measurements. Nuclear matter parameters appear as Taylor coefficients in the expansion of the EoS around the saturation density of symmetric and asymmetric nuclear matter, and provide a physically-motivated representation of the EoS. In this paper, we introduce a deep learning-based methodology to predict key neutron star observables such as the NS mass, NS radius, and tidal deformability from a set of nuclear matter parameters. Using generated mock data, we confirm that the neural network model is able to accurately capture the underlying physics of finite nuclei and replicate inter-correlations between the symmetry energy slope, its curvature and the tidal deformability arising from a set of physical constraints. We also perform a systematic Bayesian estimation of NMPs in light of recent observational data with the trained neural network and study the effects of correlations among these NMPs. We show that by not considering inter-correlations arising from finite nuclei constraints, an intrinsic uncertainty of upto 10-30% can be observed on higher order NMPs.

## Requirements
- `Python>=3.7`, `NumPy>=1.20 `, and `TensorFlow>=2.5 `.
- Use the following command to install required packages.
    - ```pip install -r requirements.txt```
    - We recommend the use of virtual environments to avoid disturbing preexisting package installations on your computer. Two popular virtual environments managers are [`Anaconda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and Python's own [`venv`](https://docs.python.org/3/library/venv.html) module. 

## Dataset

We supply the dataset used to train the NS-ANN model in our paper. The dataset consists of seven key nuclear matter parameters that govern the equation of state and six target neutron star properties. The data path is [`./data/NS-EOS-Data.csv`](https://github.com/ameya1101/NS-ANN/blob/main/data/NS-EOS-Data.csv). 

We also supply the data points for both posterior distributions over the nuclear matter parameters following the Bayesian inference. These samples are used to generate the corner plot in Figure 5 from the paper. The data path is [`./data/posterior/`](https://github.com/ameya1101/NS-ANN/tree/main/data/posterior). The `.json` files are BILBY outputs, and the `.dat` files contain pure data samples read from the BILBY output.

## Pre-trained Model

Along with the dataset, the pre-trained NS-ANN model is offered in [`./pretrained/`](https://github.com/ameya1101/NS-ANN/tree/main/pretrained/NS-ANN). For a given set of nuclear matter parameters, the model predicts the six aforementioned neutron star observables corresponding to a star described by the input equation of state. To test the pre-trained model, run:
```
python -m predict.py --input=<path-to-input-file> --output=<path-to-output-file> \
                     --model=<path-to-pretrained-model>
```
The input file must follow the same format as the same input file [`./data/sample.csv`](https://github.com/ameya1101/NS-ANN/blob/main/data/sample.csv). The `utils/` directory contains pickle files for data scalers required in the prediction pipeline. Please ensure that it is located in the same working directory as the script. 

## Citation
If you find this work helpful, please cite our paper:
```
@article{thete2020nmp,
  title={Inferring the dense matter equation of state from neutron star observations via artificial neural networks},
  author={Thete, Ameya and Banerjee, Kinjal and Malik, Tuhin},
  archivePrefix = {arXiv},
  eprint = {2208.13163},
  primaryClass = {nucl-th},
  year={2022}
} 
```
