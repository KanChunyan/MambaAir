# MambaAir


## Platform

- NVIDIA 4090 24GB GPU, PyTorch

## Usage

1. Install Python 3.8. For convenience, execute the following command.

````
pip install -r requirements.txt
````

2. Prepare Data. You can obtain the well pre-processed datasets from [Google Drive](), then place the downloaded data in the folder ````./dataset````.

3. Train and evaluate model. 
````
bash ./scripts/long_term_forecast/96_look-back_window/Air_script/PerimidFormer.sh
