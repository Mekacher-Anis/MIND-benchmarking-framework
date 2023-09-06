import itertools
from pprint import pprint
import random
from typing import List
import orjson as json
from torch.utils import data
from recommenders.datasets import mind, download_utils
import zipfile
import os
import json
import pandas as pd
from torchtext.data import get_tokenizer
from tqdm import tqdm
from torch import tensor,LongTensor,device,zeros,int32,long,float



def download_mind(path: str, dataset_size: str):
    train_folder_path = os.path.join(path, dataset_size, 'train')
    valid_folder_path = os.path.join(path, dataset_size, 'valid')
    test_folder_path = os.path.join(path, dataset_size, 'test')

    # download only if not existsdownload_mind
    if os.path.exists(train_folder_path) and os.path.exists(valid_folder_path) and (dataset_size == 'small' or os.path.exists(test_folder_path)):
        return {'train': train_folder_path, 'valid': valid_folder_path, 'test': test_folder_path}


    print("Downloading dataset...")
    # download zip files (recommenders doesn't download the test zip)
    train_zip_path, valid_zip_path = mind.download_mind(size=dataset_size, dest_path=os.path.join(path, dataset_size))
    if dataset_size == 'large':
        url = mind.URL_MIND[dataset_size][0].replace('train', 'test')
        test_zip_path = download_utils.maybe_download(url=url, work_directory=os.path.join(path, dataset_size))

    print("Extracting dataset...")
    # extract zip files
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(train_folder_path)
    with zipfile.ZipFile(valid_zip_path, 'r') as zip_ref:
        zip_ref.extractall(valid_folder_path)
    if dataset_size == 'large':
        with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
            zip_ref.extractall(test_folder_path)
            
    # turn tsv to json files similar to the structure defined here
    # https://github.com/aqweteddy/NRMS-Pytorch
    print("Transforming training dataset to json...")
    mind2json(train_folder_path, 'train')
    print("Transforming valid dataset to json...")
    mind2json(valid_folder_path, 'valid')
    
    return {'train': train_folder_path, 'valid': valid_folder_path, 'test': test_folder_path}

def mind2json(path: str, dataset_type: str):
    # turn tsv to json files
    with open(os.path.join(path, 'news.tsv')) as f:
        news = pd.read_csv(f, sep='\t', header=None)
        news.columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

    with open(os.path.join(path, 'behaviors.tsv')) as f:
        behaviors = pd.read_csv(f, sep='\t', header=None)
        behaviors.columns = ['id', 'user_id', 'time', 'history', 'impressions']
    
    users_parsed = []
    article_id_to_idx = {"empty": 0}
    user_id_to_idx = {}

    for i, row in tqdm(behaviors.iterrows()):
        user_id = row["user_id"]
        if user_id not in user_id_to_idx:
            user_id_to_idx[user_id] = len(user_id_to_idx)
        user_id = user_id_to_idx[user_id]
        impressions = str(row["impressions"]).split(" ")
        pos = []
        neg = []
        for impression in impressions:
            if dataset_type == "test":
                article_id = impression
            else:
                article_id, label = impression.split("-")
                if article_id not in article_id_to_idx:
                    article_id_to_idx[article_id] = len(article_id_to_idx)
                if label == "0":
                    neg.append(article_id_to_idx[article_id])
                else:
                    pos.append(article_id_to_idx[article_id])
        history = str(row["history"]).split(" ") if pd.notna(row["history"]) else []
        for article_id in history:
            if article_id not in article_id_to_idx:
                article_id_to_idx[article_id] = len(article_id_to_idx)
        history = [article_id_to_idx[article_id] for article_id in history]
        # users_parsed.append({"user_id": user_id, "pos": pos, "neg": neg, "history": history})
        for click in pos:
            neg_samples = get_sample(neg, 4)
            users_parsed.append({"user_id": user_id, "pos": click, "neg": neg_samples, "history": history})
        # users_parsed.extend(
        #     [{"user_id": user_id, "pos": click, "history": history} for click in clicked]
        # )

    news_parsed = [
        {
            "id": 0,
            "title": ["[", "]", ""],
        },
    ]

    for i, row in tqdm(news.iterrows()):
        article_id = row["id"]
        if article_id not in article_id_to_idx:
            # print(f'article_id {article_id} not in article_id_to_idx')
            article_id_to_idx[article_id] = len(article_id_to_idx)
        article_id = article_id_to_idx[article_id]
        # title is category + subcategory + title
        title = ["[", row["category"], row["subcategory"], "]", *row["title"].split(" ")]
        news_parsed.append({"id": article_id, "title": title})
        
    # the order of the articles in the dataset is not the same as the parsed ones
    news_parsed = sorted(news_parsed, key=lambda k: k['id'])
    json.dump(users_parsed, open(os.path.join(path, 'users.json'), 'w'))
    json.dump(news_parsed, open(os.path.join(path, 'news.json'), 'w'))
    
def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)
    

class Dataset(data.Dataset):
    def __init__(self, data_path: str, w2v, maxlen: int = 15, pos_num: int = 1, neg_k: int = 4, dataset_size: str = 'large', data_type: str = 'train'):
        self.dataset_size = dataset_size
        self.data_type = data_type
        self.paths = {
            'train': os.path.join(data_path, dataset_size, 'train'),
            'valid': os.path.join(data_path, dataset_size, 'valid'),
            'test': os.path.join(data_path, dataset_size, 'test'),
        }
        self.articles: list = self.load_json(os.path.join(self.paths[data_type], 'news.json'))
        self.users: list = self.load_json(os.path.join(self.paths[data_type], 'users.json'))
        self.maxlen = maxlen
        self.neg_k = neg_k
        self.pos_num = pos_num
        self.w2id = w2v.key_to_index

    def load_json(self, file: str):
        with open(file, 'r') as f:
            return json.loads(f.read())

    def sent2idx(self, tokens: List[str], no_cat=False):
        # tokens = tokens[3:]
        if ']' in tokens and no_cat:
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
        pos = self.users[idx]['pos']
        neg = self.users[idx]['neg']
        if self.data_type != 'valid':
            neg = get_sample(neg, self.neg_k)
        history = self.users[idx]['history']
        random.shuffle(history)

        click_doc = [0] * (self.pos_num - len(history)) + history[:self.pos_num]
        click_doc = [self.sent2idx(self.articles[p]['title']) for p in click_doc]

        cand_doc = [pos] + neg
        cand_doc = [self.sent2idx(self.articles[c]['title']) for c in cand_doc]
        cand_doc_label = [1] + [0] * len(neg)

        tmp = list(zip(cand_doc, cand_doc_label))
        random.shuffle(tmp)
        cand_doc, cand_doc_label = zip(*tmp)
        if self.data_type == 'valid':
            cand_doc_label = LongTensor(cand_doc_label)
        else:
            cand_doc_label = tensor(cand_doc_label, dtype=float)
        return tensor(click_doc), tensor(cand_doc), cand_doc_label

class ValDataset(Dataset):
    def __init__(self, num=5, *args, **kwargs) -> None:
        kwargs['data_type'] = 'valid'
        super(ValDataset, self).__init__(*args, **kwargs)
        self.num = num


class TestDataset(data.IterableDataset):
    def __init__(self, data_path: str, maxlen: int = 30, hist_size: int = 100, dataset_size: str = 'small', device=device('cpu')):
        self.dataset_size = dataset_size
        self.paths = {'test': data_path }
        self.device = device
        self.tokenizer = get_tokenizer("basic_english")
        self.maxlen = maxlen
        self.hist_size = hist_size
        
        if os.path.exists(os.path.join(self.paths['test'], 'news_parsed.pkl')):
            print('Test dataset, using pickled news file')
            self.news: dict = pd.read_pickle(os.path.join(self.paths['test'], 'news_parsed.pkl'))
        else:
            raise Exception('Please preprocess the news file using src/dataset_utils.py before running evaluation.')
    
    def _line_mapper(self, line: str):
        if not line:
            return
        line = line[:-1] # remove newline
        impid, user_id, time, history, impressions = line.split('\t')
        
        history = history.split(' ') if len(history) != 0 else []
        history = history[:self.hist_size]
        history_enc = [self.news[p] for p in history]
        history_enc += [[0] * self.maxlen] * (self.hist_size - len(history))
        
        
        impressions = impressions.split(' ')
        # 300 is the longest impression list in the test set
        impr_mask = ([True] * len(impressions)) + ([False] * (300 - len(impressions)))
        cand_imp = [self.news[p] for p in impressions]
        cand_imp += [[0] * self.maxlen] * (300 - len(impressions))
        
        return impid, tensor(history_enc, device=self.device), tensor(cand_imp, device=self.device), tensor(impr_mask, device=self.device)
    
    def __iter__(self):
        # https://stackoverflow.com/a/69797320
        
        worker_total_num = data.get_worker_info().num_workers
        worker_id = data.get_worker_info().id
        
        #Create an iterator
        file_itr = open(os.path.join(self.paths['test'], 'behaviors.tsv'))

        #Map each element using the line_mapper
        mapped_itr = map(self._line_mapper, file_itr)
        
        #Add multiworker functionality
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        return mapped_itr
    
    def _generator(self):
        with open(os.path.join(self.paths['test'], 'behaviors.tsv')) as behaviors_file:
            while True:
                line = behaviors_file.readline()
                if not line:
                    return
                line = line[:-1] # remove newline
                impid, user_id, time, history, impressions = line.split('\t')
                
                history = history.split(' ') if len(history) != 0 else []
                history = history[:self.hist_size]
                history_enc = [self.news[p] for p in history]
                history_enc += [[0] * self.maxlen] * (self.hist_size - len(history))
                
                
                impressions = impressions.split(' ')
                # 300 is the longest impression list in the test set
                impr_mask = ([True] * len(impressions)) + ([False] * (300 - len(impressions)))
                cand_imp = [self.news[p] for p in impressions]
                cand_imp += [[0] * self.maxlen] * (300 - len(impressions))
                
                yield impid, tensor(history_enc, device=self.device), tensor(cand_imp, device=self.device), tensor(impr_mask, device=self.device)