# MSP-Podcast-SER-Challenge-2024
MSP-Podcast SER Challenge 2024 - L’antenne du Ventoux

## Install from GitHub

1. Clone the GitHub repository and install the requirements:

    ```bash
    git clone [https://github.com/speechbrain/speechbrain.git](https://github.com/Chaanks/MSP-Podcast-SER-Challenge-2024.git
    cd speechbrain
    pip install -r requirements.txt
    pip install --editable .
    ```

## 🏃‍♂️ Train a model

You can train a model for the task 1 using the following steps:

```python
cd Task_1
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
--data_folder /local_disk/apollon/jduret/corpus/msp \
--model_name wav2vec2
```
