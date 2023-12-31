# Introduction
This framerwork is meant to make working with the [MIND dataset](https://msnews.github.io/) easier and allow benchmarking and comparision of different models.\
The base of the project is this [repo](https://github.com/aqweteddy/NRMS-Pytorch) which was updated to use the latest version of pytroch-lightning.\
Fastformer was borrowed from this [repo](https://github.com/lucidrains/fast-transformer-pytorch).\
Implemented models:
- [x] [NRMS](https://aclanthology.org/D19-1671.pdf)
- [x] [NRMS+Fastformer](https://arxiv.org/pdf/2108.09084v6.pdf) : the only description provided in the paper -as far as I could figure out- on how this is supposed to be implemented is this line
  > In addition, in the news recommendation task, following (Wu et al., 2019) we use Fastformer in a hierarchical way to first learn news embeddings
  > from news titles and then learn user embeddings from the embeddings of historical clicked news. We use Adam (Bengio and LeCun, 2015) for model optimization.

  ¯\\\_(ツ)_/¯
- [ ] Fastformer+PLM-NR : I still have no clue what they mean by this, I still need to decipher this one out...

# Running the models
- Create a conda environment and activate it
- Install the requirements
  ```
  pip install -r requirements.txt
  ```
- Install and login to wandb https://docs.wandb.ai/quickstart
- Run the following command to download and parse the tsv files
  ```
  python src/dataset_utils.py --action download --size large
  ```
- Run the following command to preprocess the news for inference
  ```
  python src/dataset_utils.py --action preprocess --size large
  ```
- Run the following command to train
  ```
  python src/main.py
  ```
- Run the following command to generate the `prediction.txt` file
  ```
  python src/evaluate.py --model paht_to_ckpt
  ```