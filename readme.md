## Self-supervised Physician Attribute Representation Fusion for Medical Recommendation
The official implementation of the paper "Self-supervised Physician Attribute Representation Fusion for Medical Recommendation".

## Dataset
The data used in this paper is derived from the publicly available doctor recommendation dataset by [Lu et al](https://github.com/polyusmart/Doctor-Recommendation). The dataset covers 359 doctors in 14 departments and contains over 110,000 conversation records. To ensure the accuracy and validity of the data, we exclude conversation records that do not provide information about the patient's gender and age.

### Model
<img src="https://github.com/wangwangweiwei/picx-images-hosting/raw/master/RSSA/111.58h5ukmo8l.webp"/>

## Dependencies and Installation
We recommend using `Python>=3.8`, `PyTorch>=2.0.0`, and `CUDA>=11.8`.
```bash
conda create --name SPARF-MRec python=3.8
conda activate SPARF-MRec
pip install -U pip

# Install requirements
pip install -r requirements.txt
```

## How to run
### Self-supervised learning
This section explores the relationships among doctors' self-descriptions, communication abilities, and professional competence, while also considering patients' assessments of the various skills possessed by doctors. The module aims to determine whether a conversation is a record of a consultation with the target doctor.
The dataset for self-supervised learning is avaliable at `data`. To run training of self-supervised learning task, turn to the `ssl_step` directory, run:
```bash
python train.py
```

### Embedding
We utilize pre-trained models for patient representation learning, doctor representation learning, and dialogue representation learning to encode patients, doctors, and dialogues. Through this process, we acquire various essential embeddings:
- `train_profile_embedding.json`: Contains doctor profile embeddings.
- `dialog_professional_embeddings.json`: Stores doctor professional competence embeddings.
- `dialog_emotion_embeddings.json`: Holds embeddings related to doctor communication abilities.
- `query_embeddings.json`: Contains embeddings representing patient consultation needs.

### Recommendation
Through the doctor recommendation module, we can combine various embeddings obtained through `SSL` for the doctor and the embedding for the patient to derive the probability of patient satisfaction with the doctor.
#### Training
To run training of recommendation task, turn to the `re_step` directory, run:
```bash
python train_single_up.py
```

#### validation
We select the model with the lowest validation loss and save it for subsequent testing.

#### test
For test purposes, particularly within the `eval.sh` script, we employ [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) to assess predictions using information retrieval metrics, including precision@N (P@N), mean average precision (MAP), and ERR@N. In this context, N is configured as 1 for P@N and 5 for ERR@N.
- Obtain the test output `test.dat` and `test_best_model.pt_score.txt` by running:
   ```bash
      python test_single.py
   ```
- Move `test.dat` and `test_best_model.pt_score.txt` to the `utils` folder to obtain `sorted_test_best_model.pt.dat` by running:
  ```bash
    python sort_by_score.py
  ```
- Obtain the recommendation results by running:
  ```bash
  ./eval.sh
  ```
> Note that if you encounter any issues, please first check the dependencies or file paths in the code. If the issue persists, feel free to contact me via email.
