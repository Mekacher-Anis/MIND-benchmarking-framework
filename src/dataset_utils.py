from dataset import TestDataset, download_mind, mind2json
import gensim.downloader as api

def preprocess_news_for_testing(news_file_path: str):
    print("Doing imports...")
    from torchtext.data import get_tokenizer
    import gensim.downloader as api
    import pickle
    from tqdm import tqdm

    print("Loading word embeddings...")
    w2v = api.load('word2vec-google-news-300')
    w2id = w2v.key_to_index
    maxlen = 30
    tokenizer = get_tokenizer("basic_english")

    def sent2idx(sentence: str, category: str, subcategory: str):
        # tokens = tokenizer(sentence)
        tokens = ["[", category, subcategory, "]", *sentence.split(" ")]
        tokens = [w2id[token.strip()]
                    for token in tokens if token.strip() in w2id.keys()]
        tokens += [0] * (maxlen - len(tokens))
        tokens = tokens[:maxlen]
        return tokens

    print("Parsing news...")
    news_parsed = {}
    with open(news_file_path) as news_file:
        for line in tqdm(news_file, total=2370727):
            if not line:
                break
            line = line[:-1] # remove newline
            id, category, subcategory, title, abstract, url, title_entities, abstract_entities = line.split('\t')
            news_parsed[id] = sent2idx(title, category, subcategory)
    
    print("Dumping pickled news...")
    with open(os.path.join(os.path.dirname(news_file_path),'./news_parsed.pkl'), 'wb+') as news_parsed_file:
        pickle.dump(news_parsed, news_parsed_file)

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='large')
    parser.add_argument('--action', type=str, default='preprocess') # preprocess, download
    args = parser.parse_args()
    
    if args.action == 'download':
        print(f'Downloading MIND {args.size} to {os.path.abspath("data")}')
        # download_mind('data', args.size)
        mind2json(os.path.join('data', 'large', 'train'), 'train')
        mind2json(os.path.join('data', 'large', 'valid'), 'train')
    elif args.action == 'preprocess':
        print('Preprossing news file for test set...')
        news_file_path = os.path.abspath(f'./data/{args.size}/test/news.tsv')
        preprocess_news_for_testing(news_file_path)
                