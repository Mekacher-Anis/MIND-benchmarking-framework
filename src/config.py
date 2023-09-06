hparams = {
    'batch_size': 100,
    'lr': 1e-4,
    'name': 'ranger',
    'version': 'v3',
    'description': 'FastformerNRMS',
    'pretrained_model': 'word2vec-google-news-300',
    'model': {
        'dct_size': 'auto',
        'nhead': 20,
        'embed_size': 300,
        'encoder_size': 400,
        'maxlen': 30,
        'v_size': 200
    },
    'data': {
        'max_hist': 100,
        'neg_k': 4,
        'maxlen': 30,
        'dataset_size': 'large' # 'small', 'large', 'demo'     
    }
}
