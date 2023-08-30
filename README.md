<p align="center">
<img src="https://github.com/ci-ber/RA/assets/106509806/7843d7bc-65e6-420b-a6a8-af39c7897982" width="200" class="center">
</p>
<h1 align="center">
  <br>
Generalizing Unsupervised Anomaly Detection: Towards Unbiased Pathology Screening  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">MIDL 2023</h4>
<h4 align="center"><a href="https://ci.bercea.net/project/ra/">Project Website</a> • <a href="https://openreview.net/pdf?id=8ojx-Ld3yjR">Paper</a> </h4>

<p align="center">
<img src="https://github.com/ci-ber/RA/assets/106509806/03b9aa7f-3357-421e-b1e9-b0a80c4b43e0">
</p>

## Citation

If you find our work helpful, please cite our paper:
```
@inproceedings{
bercea2023generalizing,
title={Generalizing Unsupervised Anomaly Detection: Towards Unbiased Pathology Screening},
author={Cosmin I. Bercea and Benedikt Wiestler and Daniel Rueckert and Julia A Schnabel},
booktitle={Medical Imaging with Deep Learning},
year={2023},
url={https://openreview.net/forum?id=8ojx-Ld3yjR}
}
```

> **Abstract:** *The main benefit of unsupervised anomaly detection is the ability to identify arbitrary instances of pathologies even in the absence of training labels or sufficient examples of the rare class(es). Even though much work has been done on using auto-encoders (AE) for anomaly detection, there are still two critical challenges to overcome: First, learning compact and detailed representations of the healthy distribution is cumbersome. Second, the majority of unsupervised algorithms are tailored to detect hyperintense lesions on FLAIR brain MR scans. We found that even state-of-the-art (SOTA) AEs fail to detect several classes of non-hyperintense anomalies on T1w brain MRIs, such as brain atrophy, edema, or resections. In this work, we propose reversed AEs (RA) to generate pseudo-healthy reconstructions and localize various brain pathologies. Our method outperformed SOTA methods on T1w brain MRIs, detecting more global anomalies (AUROC increased from 73.1 to 89.4) and local pathologies (detection rate increased from 52.6% to 86.0%).*


## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview: 

<p align="center">
<img src="https://github.com/ci-ber/RA/assets/106509806/844b35fa-0e3e-4b1c-8b4c-1adecae6703a">
</p>

#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository

```bash
git clone https://github.com/ci-ber/RA.git
cd RA
```

#### 3). Create a virtual environment with the needed packages (use conda_environment-osx.yaml for macOS)

```
cd ${TARGET_DIR}/RA
conda env create -f ra_environment.yaml
conda activate ra_env *or* source activate ra_env
```

#### 4). Install PyTorch 

> Example installation:

* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 5). Download datasets 

<h4 align="center"><a href="https://brain-development.org/ixi-dataset/">IXI</a> • <a href="https://fastmri.org">FastMRI</a> • <a href="https://github.com/microsoft/fastmri-plus"> Labels for FastMRI</a> </h4>

> *Alternatively you can use your own mid-axial brain T1w slices with our pre-trained weights or train from scratch on other anatomies and modalities.*

> Move the datasets to the expected paths (listed in the data/splits csv files)

#### 6). Run the pipeline

> [Optional] set config 'task' to test and load model from ./weights/RA/best_model.pt

```
python core/Main.py --config_path projects/RA/configs/fast_mri/ra.yaml
```

> Refer to *.yaml files for experiment configurations.



# That's it, enjoy! :rocket:





