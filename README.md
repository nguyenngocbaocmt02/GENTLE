# [ICML 2024] GENERATIVE CONDITIONAL DISTRIBUTIONS BY NEURAL (ENTROPIC) OPTIMAL TRANSPORT

<div  align="center">

<a  href="https://nguyenngocbaocmt02.github.io/"  target="_blank">Bao&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://tbng.github.io/"  target="_blank">Binh&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://hieunt91.github.io/"  target="_blank">Hieu Trung&nbsp;Nguyen</a> &emsp; <b>&middot;</b> &emsp;
<a  href="https://www.vietanhnguyen.net/"  target="_blank">Viet Anh&nbsp;Nguyen</a>
<br>
<a  href="https://arxiv.org/abs/2406.02317">[Paper]</a> &emsp;&emsp; 
<a  href="https://openreview.net/forum?id=FoRqdsN4IA">[OpenReview]</a> &emsp;&emsp;

</div>

<br>
    


Learning conditional distributions is challenging because the desired outcome is not a single distribution but multiple distributions that correspond to multiple instances of the covariates. We introduce a novel neural entropic optimal transport method designed to effectively learn generative models of conditional distributions, particularly in scenarios characterized by limited sample sizes. Our method relies on the minimax training of two neural networks: a generative network parametrizing the inverse cumulative distribution functions of the conditional distributions and another network parametrizing the conditional Kantorovich potential. To prevent overfitting, we regularize the objective function by penalizing the Lipschitz constant of the network output. Our experiments on real-world datasets show the effectiveness of our algorithm compared to state-of-the-art conditional distribution learning techniques.


##### Table of contents

1. [Installation](#installation)

2. [Dataset](#dataset-preparation)

3. [How to run](#how-to-run)

4. [Acknowledgments](#acknowledgments)

5. [Contacts](#contacts)

  
  

Details regarding our methodology and the corresponding experimental results are available in [our following paper](https://arxiv.org/abs/2406.02317):

  

**Please CITE** our paper whenever utilizing this repository to generate published results or incorporating it into other software.

```
@misc{nguyen2024generative,
      title={Generative Conditional Distributions by Neural (Entropic) Optimal Transport}, 
      author={Bao Nguyen and Binh Nguyen and Hieu Trung Nguyen and Viet Anh Nguyen},
      year={2024},
      eprint={2406.02317},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
  

## Installation ##

Python `3.9` and Pytorch `1.11.0+cu113` are used in this implementation.

It is recommended to create `conda` env from our provided config files `./environment.yml` and `./requirements.txt`:
```
conda env create -f environment.yml
conda activate gentle
pip install -r requirements.txt
```

## Dataset ##

We use two datasets including the LDW-CPS dataset and the ECM dataset. The original LDW-CPS dataset is available at `datasets/cps/cps.csv` which is originally downloaded by following the guideline in [Using Wasserstein Generative Adversial Networks for the Design of Monte Carlo Simulations.](https://github.com/gsbDBI/ds-wgan). The train, test, validation sets and the script to obtain them are also in the same folder. The ECM dataset is available at `datasets/cancer`. The train, validation, test sets are in the same folder and can be regenerated by running `datasets/data_generation.py` for each set.

## How to run ##

**GPU allocation**: Our work is experimented on 4 NVIDIA A5000 GPUs.

Check out `./train_gentle.py`, `./train_mgan.py`, `./train_wgan_gp.py`, `./train_cdsb.py`, `./train_wgan.py` for training our method and other baselines. 

Check out the `./grid_search.py` for finding reasonable parameters for each algorithm.

## Acknowledgments ##

Thanks to Athey et al. and Shi et al. for releasing their official implementation of the [Using Wasserstein Generative Adversial Networks for the Design of Monte Carlo Simulations.](https://github.com/gsbDBI/ds-wgan), [Conditional Simulation Using Diffusion Schrödinger Bridges](https://github.com/vdeborto/cdsb). Thanks to Erik et al. for their awesome repository for GAN-based implementations [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN).

## Contacts ##

If you have any problems, please open an issue in this repository or ping an email to [nguyenngocbaocmt@gmail.com](mailto:nguyenngocbaocmt@gmail.com).
