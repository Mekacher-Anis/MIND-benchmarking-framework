import random
from typing import List
import orjson as json
import torch
from gensim.models import Word2Vec
from torch.utils import data
from tqdm import tqdm
from recommenders.datasets import mind, download_utils
import zipfile
import os
import json
import pandas as pd

def download_mind(path: str, dataset_size: str):
    train_folder_path = os.path.join(path, dataset_size, 'train')
    valid_folder_path = os.path.join(path, dataset_size, 'valid')
    test_folder_path = os.path.join(path, dataset_size, 'test')

    # download only if not exists
    if os.path.exists(train_folder_path) and os.path.exists(valid_folder_path) and os.path.exists(test_folder_path):
        return {'train': train_folder_path, 'valid': valid_folder_path, 'test': test_folder_path}

    # download zip files (recommenders doesn't download the test zip)
    train_zip_path, valid_zip_path = mind.download_mind(size=dataset_size, dest_path=os.path.join(path, dataset_size))
    url = mind.URL_MIND[dataset_size][0].replace('train', 'test')
    test_zip_path = download_utils.maybe_download(url=url, work_directory=os.path.join(path, dataset_size))

    # extract zip files
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(train_folder_path)
    with zipfile.ZipFile(valid_zip_path, 'r') as zip_ref:
        zip_ref.extractall(valid_folder_path)
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(test_folder_path)

    # turn tsv to json files similar to the structure defined here
    # https://github.com/aqweteddy/NRMS-Pytorch
    mind2json(train_folder_path, 'train')
    mind2json(valid_folder_path, 'valid')
    
    return {'train': train_folder_path, 'valid': valid_folder_path, 'test': test_folder_path}

def mind2json(path: str, dataset_type: str):
    # turn tsv to json files
    with open(os.path.join(path, 'news.tsv')) as f:
        news = pd.read_csv(f, sep='\t', header=None)
        news.columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        news.drop_duplicates(subset=['title'], inplace=True)

    with open(os.path.join(path, 'behaviors.tsv')) as f:
        behaviors = pd.read_csv(f, sep='\t', header=None)
        behaviors.columns = ['id', 'user_id', 'time', 'history', 'impressions']
    
    users_parsed = []
    article_id_to_idx = {}
    user_id_to_idx = {}

    for i, row in behaviors.iterrows():
        user_id = row['user_id']
        if user_id not in user_id_to_idx:
            user_id_to_idx[user_id] = len(user_id_to_idx)
        user_id = user_id_to_idx[user_id]
        impressions = str(row['impressions']).split(' ')
        clicked = []
        for impression in impressions:
            if dataset_type == 'test':
                article_id = impression
            else:
                article_id, label = impression.split('-')
                if label == '0':
                    continue
                # get article idx
            if article_id not in article_id_to_idx:
                article_id_to_idx[article_id] = len(article_id_to_idx)
            clicked.append(article_id_to_idx[article_id])
        history = str(row['history']).split(' ')
        for article_id in history:
            if article_id not in article_id_to_idx:
                article_id_to_idx[article_id] = len(article_id_to_idx)
        users_parsed.append({'user_id': user_id, 'push': clicked, 'history': history})
    
    news_parsed = []

    for i, row in news.iterrows():
        article_id = row['id']
        if article_id not in article_id_to_idx:
            article_id_to_idx[article_id] = len(article_id_to_idx)
        article_id = article_id_to_idx[article_id]
        # title is category + subcategory + title
        title = ['[', row['category'], row['subcategory'], ']', *row['title'].split(' ')]
        news_parsed.append({'id': article_id, 'title': title})
    
    json.dump(users_parsed, open(os.path.join(path, 'users.json'), 'w'))
    json.dump(news_parsed, open(os.path.join(path, 'news.json'), 'w'))
    

class Dataset(data.Dataset):
    def __init__(self, data_path: str, w2v, maxlen: int = 15, pos_num: int = 1, neg_k: int = 4, dataset_size: str = 'small', data_type: str = 'train'):
        self.dataset_size = dataset_size
        self.data_type = data_type
        self.paths = download_mind(data_path, dataset_size)
        self.articles = self.load_json(os.path.join(self.paths[data_type], 'news.json'))
        self.users = self.load_json(os.path.join(self.paths[data_type], 'users.json'))
        self.maxlen = maxlen
        self.neg_k = neg_k
        self.pos_num = pos_num
        self.w2id = w2v.key_to_index

    def load_json(self, file: str):
        with open(file, 'r') as f:
            return json.loads(f.read())

    def sent2idx(self, tokens: List[str]):
        # tokens = tokens[3:]
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        """getitem

        Args:
            idx (int): 
        Data:articles
            return (
                user_id (int): 1
                click (tensor): [batch, num_click_docs, seq_len]
                cand (tensor): [batch, num_candidate_docs, seq_len]
                label: candidate docs label (0 or 1)
            )
        """
        push = self.users[idx]['push']
        random.shuffle(push)
        push = push[:self.pos_num]
        uid = self.users[idx]['user_id']
        click_doc = [self.sent2idx(self.articles[p]['title']) for p in push]
        cand_doc = []
        cand_doc_label = []
        # neg
        for i in range(self.neg_k):
            neg_id = -1
            while neg_id == -1 or neg_id in push:
                neg_id = random.randint(0, len(self.articles) - 1)
            cand_doc.append(self.sent2idx(self.articles[neg_id]['title']))
            cand_doc_label.append(0)
        # pos
        try:
            cand_doc.append(self.sent2idx(
                self.articles[push[random.randint(50, len(self.push) - 1)]['title']]))
            cand_doc_label.append(1)
        except Exception:
            try:
                cand_doc.append(self.sent2idx(self.articles[push[0]]['title']))
            except:
                print(push[0])
                print(self.articles[push[0]])
            cand_doc_label.append(1)

        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        return torch.tensor(click_doc), torch.tensor(cand_doc), torch.tensor(cand_doc_label, dtype=torch.float).argmax(0)

class ValDataset(Dataset):
    def __init__(self, num=5, *args, **kwargs) -> None:
        kwargs['data_type'] = 'valid'
        super(ValDataset, self).__init__(*args, **kwargs)
        self.num = num
    
    def __getitem__(self, idx: int):
        push = self.users[idx]['push']
        random.shuffle(push)
        uid = self.users[idx]['user_id']
        click_doc = [self.sent2idx(self.articles[p]['title']) for p in push[:self.pos_num]]
        
        true_num = 1
        # true_num = random.randint(1, min(self.num, len(push)) )
        f_num = self.num - true_num
        cand_doc = random.sample(push, true_num) # true
        cand_doc_label = [1] * true_num
        cand_doc.extend(random.sample(range(0, len(self.articles)), f_num)) # false
        cand_doc_label.extend([0] * f_num)
        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        cand_doc = [self.sent2idx(self.articles[cand]['title']) for cand in cand_doc]
        return torch.LongTensor(click_doc), torch.LongTensor(cand_doc), torch.LongTensor(cand_doc_label)


class TestDataset(data.Dataset):
    def __init__(self, data_path: str, w2v, maxlen: int = 15, dataset_size: str = 'small'):
        self.dataset_size = dataset_size
        self.paths = download_mind(data_path, dataset_size)
        
        with open(os.path.join(self.paths['test'], 'news.tsv')) as f:
            self.news = pd.read_csv(f, sep='\t', header=None)
            self.news.columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
            self.news.drop_duplicates(subset=['title'], inplace=True)

        with open(os.path.join(self.paths['test'], 'behaviors.tsv')) as f:
            self.behaviors = pd.read_csv(f, sep='\t', header=None)
            self.behaviors.columns = ['id', 'user_id', 'time', 'history', 'impressions']
            self.behaviors['history'] = self.behaviors['history'].fillna('')
            self.behaviors['history'] = self.behaviors['history'].str.split(' ')
            self.behaviors['impressions'] = self.behaviors['impressions'].str.split(' ')
            
        self.maxlen = maxlen
        if w2v is not None:
            self.w2id = w2v.key_to_index


    def sent2idx(self, tokens: List[str]):
        # tokens = tokens[3:]
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens

    def __len__(self):
        return len(self.behaviors.index)
    
    def __getitem__(self, idx: int):
        if not (self.behaviors.id == idx + 1).any():
            return None
        history = self.behaviors[self.behaviors.id == idx + 1]['history'].item()
        impressions = self.behaviors[self.behaviors.id == idx + 1]['impressions'].item()
        history_enc = [self.news[self.news['id'] == p]['title'] for p in history]
        cand_imp = [self.news[self.news['id'] == p]['title'] for p in impressions]
        impid = self.behaviors[self.behaviors.id == idx + 1]['id'].item()
        return impid, history_enc, cand_imp
        # history_enc = [self.sent2idx(self.news[self.news['id'] == p]['title'].item()) for p in history]        
        # cand_imp = [self.sent2idx(self.news[self.news['id'] == p]['title'].item()) for p in impressions]
        # return impid, torch.LongTensor(history_enc), torch.LongTensor(cand_imp)

if __name__ == '__main__':
    ds = TestDataset('./data', None, dataset_size='large')
    for i in ds:
        if not i:
            break
        print(i)
        pass
