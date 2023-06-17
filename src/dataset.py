import random
from typing import List
import orjson as json
import torch
from gensim.models import Word2Vec
from torch.utils import data
from tqdm import tqdm
from recommenders.datasets import mind
import zipfile
import os
import json
import pandas as pd


class Dataset(data.Dataset):
    def __init__(self, data_path: str, w2v, maxlen: int = 15, pos_num: int = 1, neg_k: int = 4):
        train_path, valid_path = self.download(data_path)
        self.articles = self.load_json(os.path.join(train_path, 'news.json'))
        self.users = self.load_json(os.path.join(train_path, 'users.json'))
        self.maxlen = maxlen
        self.neg_k = neg_k
        self.pos_num = pos_num

        self.w2id = w2v.key_to_index


    def download(self, path: str):
        train_folder_path = os.path.join(path, 'train')
        valid_folder_path = os.path.join(path, 'valid')

        # download only if not exists
        if os.path.exists(train_folder_path) and os.path.exists(valid_folder_path):
            return train_folder_path, valid_folder_path

        train_zip_path, valid_zip_path = mind.download_mind(size='small', dest_path=path)

        # extract zip files
        with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
            zip_ref.extractall(train_folder_path)
        with zipfile.ZipFile(valid_zip_path, 'r') as zip_ref:
            zip_ref.extractall(valid_folder_path)

        self.mind2json(train_folder_path)
        self.mind2json(valid_folder_path)
        
        
        return train_folder_path, train_zip_path
    
    def mind2json(self, path: str):
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
                article_id, label = impression.split('-')
                if label == '0':
                    continue
                # get article idx
                if article_id not in article_id_to_idx:
                    article_id_to_idx[article_id] = len(article_id_to_idx)
                clicked.append(article_id_to_idx[article_id])
            users_parsed.append({'user_id': user_id, 'push': clicked})
        
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


if __name__ == '__main__':
    w2v = Word2Vec.load('./word2vec/wiki_300d_5ws.model')
    ds = ValDataset(50, './data/articles.json', './data/users_list.json',
                 w2v, maxlen=30, pos_num=50, neg_k=4)
    print(ds[10])
    for i in tqdm(ds):
        pass
