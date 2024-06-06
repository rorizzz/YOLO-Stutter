# YOLO-Stutter: End-to-end Region-Wise Speech Dysfluency Detection

## Workflow
<img src="Resources/workflow.png" alt="image-20240321090057059" style="zoom: 20%; display: block; margin-right: auto; margin-left: 0;" />


## Datasets
We open sourced our two simulated datasets, VCTK-TTS and VCTK-Stutter. The download links are as follows:

| Dataset                        | URL                                                                                            |
|-----------------------------|------------------------------------------------------------------------------------------------|
| VCTK-TTS               | [link](https://drive.google.com/drive/folders/1p5bRQOqMP3AkaoVp3hIGpiJ6ePkYoaeZ?usp=sharing)  |
| VCTK-Stutter                | [link](https://drive.google.com/drive/folders/1EVAetm36uQa0JdTUC2qPUEK3dWRq3qdt?usp=sharing) |


```
${DATASET}
├── disfluent_audio/  # simulated audio (.wav)
├── disfluent_labels/ # simualted labels (.json)	      
└── gt_text/  # ground truth text (.txt)
```


## Environment configuration
Please refer [environment.yml](environment.yml)

If you have Miniconda/Anaconda installed, you can directly use the command: `conda env create -f environment.yml`



## Dysfluency simulation
We use [VITS](https://github.com/jaywalnut310/vits) as our TTS model.

0. Clone this repository

0. Download [VITS pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing), here we need `pretrained_vctk.pth` to achieve multi-speaker.
   1. create a folder `dysfluency_simulation/path/to`, and put the downloaded model into it.

0. Build Monotonoic Alignment Search
```sh
cd dysfluency_simulation/monotonic_align
python setup.py build_ext --inplace
```
3. Generate simulated speech
```sh
# Phoneme level
python generate_phn.py

# Word level
python generate_word.py
```

