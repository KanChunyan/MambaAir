# MabamAir


## Platform

- NVIDIA 4090 24GB GPU, PyTorch

## Usage

1. Install Python 3.8. For convenience, execute the following command.

````
pip install -r requirements.txt
````

2. Prepare Data. You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), then place the downloaded data in the folder ````./dataset````.

3. Train and evaluate model. 
````
bash ./scripts/long_term_forecast/ETT_script/PerimidFormer_ETTh2.sh
