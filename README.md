## [EuroXR2025] Cross-System Virtual Reality (VR) Authentication Using Transformer-Based Trajectory Forecasting


<div align="center">
<img src="https://github.com/Terascale-All-sensing-Research-Studio/Cross_System_Authentication_using_Forecasting/blob/main/figs/teaser.png" height=100%>
</div>

Ubiquitous devices, such as smartphones and tablets, have pervaded critical applications in education, healthcare, and military, with users having systems for their home, office, clinic, or job site. As virtual reality (VR) systems become more affordable, they are likely to gain similar multi-system adoption in critical applications where sensitive user data must be protected from malicious users. Recent work has shown that deep learning techniques are usable for cross-system authentication where enrollment data is provided in one system, e.g. an HTC Vive, and use time data on another, e.g. a Meta Quest. However, these approaches require complete or near complete trajectories of user behavior and show lower performance when smaller portions of user behavior from the start of the activity are used. In prior work, motion forecasting has been used to predict future motion and use it for VR authentication. However, this work works on a single VR system, and requires a distinct authentication model per user. We present the __first__ generalized authentication framework for cross-system VR biometrics using a forecasting neural network based on the Transformer, and an authentication neural network based on a fully convolutional network. To validate our approach, we use a [publicly available dataset](https://github.com/Terascale-All-sensing-Research-Studio/MultiModal_VR_BallThrowing_Dataset)  that provides motion trajectories from a ball-throwing task using multiple VR systems. We show that our approach reduces the equal error rate (EER) by an average of __53.16%__ across all VR system combinations when compared to existing state-of-the-art approaches on cross-VR-system authentication. Our approach enables interoperability across VR systems that use lighthouse- and camera-based tracking and provides early authentication without requiring the full user trajectory.

If you find our work helpful please cite us:
```
@inproceedings{li2025cross,
  title={Cross-System Virtual Reality (VR) Authentication Using Transformer-Based Trajectory Forecasting},
  author={Li, Mingjun and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={International Conference on Virtual Reality and Mixed Reality},
  pages={},
  year={2025},
  organization={Springer}
}
```

## Installation

Code tested using Ubutnu __20.04__ and python __3.8__.

We recommend using virtualenv. The following snippet will create a new virtual environment, activate it, and install deps.
```bash
sudo apt-get install virtualenv && \
virtualenv -p python venv && \
source venv/bin/activate && \
git clone https://github.com/Terascale-All-sensing-Research-Studio/Cross_System_Authentication_using_Forecasting.git && \
pip install -r requirements.txt
```

## Training

### Train a general forecasting model
```
cd python
python forecast_train.py --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

### Finetune the pretrained general forecasting model
```
cd python
python finetune.py --train_for <enrollment_system> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

### Authentication using the finetuned forecasting model
```
cd python
python auth.py --train_for <enrollment_system> --test_for <use-time systemm> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```
Use the flag ```-l``` if save the log file for all the above runs .