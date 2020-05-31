from glob import glob
import torch

corpus_types = ['train', 'test', 'valid']
all_train_text = []
all_test_text = []
all_valid_text = []
train_length = 0
test_length = 0
valid_length = 0

# train length: 287083, test length: 11489, valid length: 13367
# dict_keys(['src', 'tgt', 'src_sent_labels', 'segs', 'clss', 'src_txt', 'tgt_txt'])

for corpus_type in corpus_types:
    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob('../data/cnndm' + '.' + corpus_type + '.[0-9]*.pt'))  # pts -> list
    if pts:
        for pt in pts:
            dataset = torch.load(pt)  # dataset -> list
            for d in dataset:
                print(d.keys())
                break
            break
            # if corpus_type == 'train':
            #     # all_train_text += dataset
            #     train_length += len(dataset)
            # elif corpus_type == 'test':
            #     # all_test_text += dataset
            #     test_length += len(dataset)
            # elif corpus_type == 'valid':
            #     # all_valid_text += dataset
            #     valid_length += len(dataset)

# print(f"train length: {train_length}, test length: {test_length}, valid length: {valid_length}")
#
# torch.save(all_train_text, '../data/train.pt')
# torch.save(all_test_text, '../data/test.pt')
# torch.save(all_valid_text, '../data/valid.pt')
