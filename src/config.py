hparams = {
    'batch_size': 100,
    'lr': 5e-4,
    'name': 'ranger',
    'version': 'v3',
    'description': 'NRMS lr=5e-4, with weight_decay',
    'pretrained_model': 'word2vec-google-news-300',
    'model': {
        'dct_size': 'auto',
        'nhead': 10,
        'embed_size': 300,
        # 'self_attn_size': 400,
        'encoder_size': 250,
        'maxlen': 15,
        'v_size': 200
    },
    'data': {
        'pos_k': 1,
        'neg_k': 4,
        'maxlen': 15,
        'dataset_size': 'large' # 'small', 'large', 'demo'     
    }
}
