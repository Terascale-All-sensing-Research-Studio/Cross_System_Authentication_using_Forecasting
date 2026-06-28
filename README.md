## Cross-System Forecasting-based User Authentication for Virtual Reality

<div align="center">
<img src="https://github.com/Terascale-All-sensing-Research-Studio/Cross_System_Authentication_using_Forecasting/blob/main/figs/teaser.png" height=500%>
</div>

As virtual reality (VR) systems gain acceptance into critical domains, such as healthcare, education, banking, and military, sensitive user data has to be protected from malicious users. Existing security measures such as usernames/passwords, PINs, or multi-factor approaches do not provide any protection when the attacker gains access to the credentials or when the user intentionally hands over their credentials to an ally, for instance to take an exam on their behalf. Recognizing these challenges, a large body of work has emerged over the past decade on behavioral biometrics for single VR systems. However, cross-system behavioral biometrics for VR remains at a nascent stage. Cross-system behavioral biometrics is necessary to enable users to seamlessly transition between their office, home, job site, or clinic issued system. Early work in cross-system behavioral biometrics for VR showed that while deep learning techniques such as Siamese neural networks were effective, they required near complete trajectories for high assurance user authentication or identification. The emergence of motion forecasting approaches for VR biometrics enabled the use of limited data from the start of the user’s action, thereby preventing the attacker from gaining access to the full user trajectory. However, such forecasting models were designed for a single VR system that did not enable cross-system authentication. In recent work, we showed that cross-system forecasting-based authentication for VR biometrics can be performed using an Informer-based model to train the forecasting component and a fully convolutional network to train the authenticator. Using a [publicly available dataset](https://github.com/Terascale-All-sensing-Research-Studio/MultiModal_VR_BallThrowing_Dataset) of 41 users performing a ball throwing task using the Meta Quest, HTC Vive, and HTC Vive Cosmos, we showed that in comparison to non-forecasted Siamese networks, our approach reduces the equal error rate (EER) by an average of 53.16% across all VR system combinations over prior cross-system authentication work. In this paper, we compare the performance of the Informer-based cross-system forecasting model, which operates on a point-wise input and treats each timestamp as a separate token, against a patching architecture, namely PatchTST, that that provides lower runtime by splitting the input time series into individual channels, operates on each channel independently, and encodes channel features into patches. Using the same ball throwing dataset, we show that PatchTST shows an average EER reduction of 30.30\% over prior cross-system authentication. Using PatchTST, we obtain a speedup of more than 3 times compared to Informer when using a GPU with half the core count.

If you find our work helpful please cite us:
```
@inproceedings{li2026cross,
  title={Cross-System Forecasting-based User Authentication for Virtual Reality},
  author={Li, Mingjun and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={Computers & Graphics},
  year={2026}
}
```

```
@inproceedings{li2025cross,
  title={Cross-System Virtual Reality (VR) Authentication Using Transformer-Based Trajectory Forecasting},
  author={Li, Mingjun and Banerjee, Natasha Kholgade and Banerjee, Sean},
  booktitle={International Conference on Virtual Reality and Mixed Reality},
  pages={240--264},
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

## Informer Based Training

### Train a general forecasting model
```
cd Informer_code/python
python forecast_train.py --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

### Finetune the pretrained general forecasting model
```
cd Informer_code/python
python finetune.py --train_for <enrollment_system> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

### Authentication using the finetuned forecasting model
```
cd Informer_code/python
python auth.py --train_for <enrollment_system> --test_for <use-time systemm> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```


## PatchTST Based Training
### Train a general forecasting model
```
cd PatchTST_code/scripts
./general_forecast.sh
```

### Finetune the pretrained general forecasting model
```
cd PatchTST_code
python cross_finetune.py --train_for <enrollment_system> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

### Authentication using the finetuned forecasting model
```
cd PatchTST_code
python cross_auth.py --train_for <enrollment_system> --test_for <use-time systemm> --seq_len <seq_len> --label_len <label_len> --pred_len <pred_len> --gpu <gpu_ID>
```

Use the flag ```-l``` if save the log file for all the above runs .