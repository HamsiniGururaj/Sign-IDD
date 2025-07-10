#replacement for torchtext.data (custom module)
import torch
from constants import TARGET_PAD
#equivalent of torchtext.data.Field

def tokenize_features(features):
    print(">>> Using NEW tokenize_features with 225-dim")

    # Case 1: Already tokenized into list of frames
    if isinstance(features, list) and all(isinstance(x, list) and len(x) == 225 for x in features):
        print(">>> Already tokenized frames, skipping reshaping")
        tensor_features = [torch.tensor(x, dtype=torch.float32) for x in features]
        print(">>> Frame count:", len(tensor_features), "| Frame dim:", tensor_features[0].shape)
        return tensor_features

    # Case 2: Raw string or flat list
    if isinstance(features, str):
        features = [float(x) for x in features.strip().split()]
    
    feature_dim = 225
    if len(features) % feature_dim != 0:
        raise ValueError(f"Invalid feature vector length: {len(features)} not divisible by {feature_dim}")

    features = torch.tensor(features, dtype=torch.float32)
    features = features.view(-1, feature_dim)

    print(">>> Frame count:", features.shape[0], "| Frame dim:", features.shape[1])
    return [frame for frame in features]



def stack_features(features, _):
    # features: List[List[Tensor[feature_dim]]], each inner list = frames for a sample
    max_len = max(len(seq) for seq in features)
    feat_dim = features[0][0].shape[0]  # assume consistent feature_dim
    pad_val = TARGET_PAD

    padded = []
    for seq in features:
        pad_frame = torch.ones(feat_dim) * pad_val
        padded_seq = seq + [pad_frame] * (max_len - len(seq))
        padded.append(torch.stack(padded_seq, dim=0))  # [max_len, feat_dim]

    return torch.stack(padded, dim=0)  # [batch_size, max_len, feat_dim]

    
class TextField:
    def __init__(self,
                 sequential=True,
                 use_vocab=False,
                 tokenize=str.split,
                 lower=False,
                 include_lengths=False,
                 batch_first=False,
                 pad_token="<pad>",
                 unk_token="<unk>",
                 eos_token=None,
                 init_token=None,
                 dtype=torch.long,
                 preprocessing=None,
                 postprocessing=None):
        self.tokenize = tokenize
        self.lower = lower
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.init_token = init_token
        self.include_lengths = include_lengths
        self.vocab = None
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.batch_first = batch_first
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    #if enabled, convert sentence into lowercase, tokenize and append eos token
    def preprocess(self, sentence):
        #print(f"[preprocess] id: {id(self)} | use_vocab={self.use_vocab} | preprocessing: {self.preprocessing}")
        if isinstance(sentence, list):
            tokens = sentence  # already tokenized
        else:
            if self.lower:
                sentence = sentence.lower()
            tokens = self.tokenize(sentence)
        if self.eos_token:
            tokens.append(self.eos_token)
        
        if self.preprocessing and not self.use_vocab:
            #print("Preprocessing")
            print(">> Preprocessing tokens of type:", type(tokens), "length:", len(tokens))
            tokens = self.preprocessing(tokens)

        return tokens

    #returns a vocabulary dictionary
    def build_vocab(self, dataset, field_name="src", min_freq=1, max_size=None, vocab_file=None):
        from collections import Counter
        counter = Counter()
        for example in dataset:  
            tokens = getattr(example, field_name)  #tokenize each sentence in dataset
            counter.update(tokens)  #count the frequency of each word

        specials = [self.pad_token, self.unk_token]  #add special tokens
        if self.eos_token:
            specials.append(self.eos_token)

        vocab = {tok: idx for idx, tok in enumerate(specials)}  #assign indices to each token (lower index for more frequent tokens)
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in vocab:
                vocab[word] = len(vocab)
                if max_size and len(vocab) >= max_size:
                    break                                       #vocab is a dictionary containing {token:index}

        self.vocab = vocab
        self.itos = {i: w for w, i in vocab.items()}  #itos is a reverse mapping (list:itos[index]->token)

    def numericalize(self, tokens):  #converts the input tokens into their corresponding indices
        #print(">>> self.vocab:", self.vocab)
        #print(">>> type(self.vocab):", type(self.vocab))
        #print(">>> hasattr(self.vocab, 'stoi'):", hasattr(self.vocab, 'stoi'))
        #print(">>> self.unk_token:", self.unk_token)
        
        if self.vocab is None:
            print("None encountered")
            return []
        elif isinstance(self.vocab, dict):
            return [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in tokens]
    # If vocab has a .stoi attribute (e.g., from torchtext)
        elif hasattr(self.vocab, 'stoi'):
            return [self.vocab.stoi.get(tok, self.vocab.stoi[self.unk_token]) for tok in tokens]
        else:
            raise TypeError(f"Unsupported vocab type: {type(self.vocab)}")
    
        #return [self.vocab.stoi.get(tok, self.vocab.stoi[self.unk_token]) for tok in tokens]


#equivalent of torchtext.data.RawField
class RawField:  #no processing of input
    def preprocess(self, x):
        return x  # No change


#equivalent of torchtext.data.Dataset and Example
from torch.utils.data import Dataset
#container for one sample of data from the dataset
class Example:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            setattr(self, k, v)

    @classmethod
    def fromlist(cls, data_list, fields):
        data_dict = {}
        for (name, field), val in zip(fields, data_list):
            #print(f"[fromlist] name: {name}, val type: {type(val)}")
            if hasattr(field, "preprocess"):
                #print(f"[fromlist] field: {name}, use_vocab={getattr(field, 'use_vocab', 'NA')}, preprocessing={getattr(field,                       'preprocessing', 'NA')}")
                val = field.preprocess(val)
            data_dict[name] = val
        return cls(data_dict)


#equivalent of torchtext.data.BucketIterator
from torch.utils.data import DataLoader

#batches and pads the examples, returns a dictionary of batched tensors
def collate_fn(fields, pad_token="<pad>", trg_pad_val=TARGET_PAD):
    def _collate(batch):
        batch_dict = {}
        for name, field in fields.items():
            items = [getattr(ex, name) for ex in batch]

            if isinstance(field, TextField) and field.use_vocab:
                max_len = max(len(x) for x in items)
                padded = []
                for x in items:
                    if field.vocab is None:
                        numer = [0] * len(x)  # fallback: all <unk>
                    else:
                        numer = field.numericalize(x)
                    pad_idx = field.vocab.stoi.get(pad_token, 0) if field.vocab is not None else 0  #default to 0 if vocab is missing
                    padded.append(numer + [pad_idx] * (max_len - len(x)))
                batch_dict[name] = torch.tensor(padded)
            
            elif isinstance(field, TextField) and not field.use_vocab:
                if field.postprocessing is not None:
                    batch_tensor = field.postprocessing(items, None)
                    #print(f">>> Padded {name} shape:", batch_tensor.shape)
                    batch_dict[name] = batch_tensor
                else:
                    # Fallback if postprocessing is not defined
                    max_len = max(len(x) for x in items)
                    padded = []
                    for seq in items:
                        pad_frame = torch.ones(len(seq[0])) * trg_pad_val
                        #print(">>> Sample item type:", type(seq))
                        #print(">>> First element in seq:", seq[0], "Type:", type(seq[0]))
                        padded_seq = seq + [pad_frame] * (max_len - len(seq))
                        padded.append(torch.stack(padded_seq))
                    batch_tensor = torch.stack(padded)
                    #print(f">>> Fallback padded {name} shape:", batch_tensor.shape)
                    batch_dict[name] = batch_tensor
                    
            elif isinstance(field, RawField):
                batch_dict[name] = items
                # Pose vectors: list of list of float tensors (frames)
                #max_len = max(len(x) for x in items)
                #padded = []
                #for seq in items:
                 #   pad_frame = torch.ones(len(seq[0])) * trg_pad_val
                  #  padded_seq = seq + [pad_frame] * (max_len - len(seq))
                   # padded.append(torch.stack(padded_seq))
                #batch_dict[name] = torch.stack(padded)
                #print(f">>> Padded {name} shape:", padded[0].shape)

        return batch_dict
    return _collate

#wraps dataloader with collate function for batching, shuffling and iterating the batches
from torch.utils.data import DataLoader

def make_data_iterator(dataset,fields, batch_size, shuffle=False, pad_token="<pad>", trg_pad_val=TARGET_PAD):
    #print(">>> make_data_iterator called, dataset size:", len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn(fields, pad_token, trg_pad_val)
    )
    #for batch in loader:
        #print(">>> Inside loader loop, batch keys:", batch.keys())
        #print(">>> trg shape:", batch["trg"].shape if "trg" in batch else "N/A")
        #break  # Only inspect first batch
    return loader

#equivalent of batch class
class Batch:
    def __init__(self, batch_dict, pad_index, target_pad):
        self.src = batch_dict["src"]
        self.trg = batch_dict["trg"]
        self.file_paths = batch_dict.get("file_paths", None)

        #print(">> self.trg.shape:", self.trg.shape)
        self.trg_input = self.trg[:, :-1, :]  # remove last frame
        self.trg_output = self.trg[:, 1:, :]  # remove first frame

        self.ntokens = (self.src != pad_index).sum().item()
        self.ntokens_trg = (self.trg_input[:, :, 0] != target_pad).sum().item()
        self.nseqs = self.src.size(0)
        self.src_lengths = (self.src != pad_index).sum(dim=1)
        
        # Masks
        self.src_mask = self.make_pad_mask(self.src, pad_index)  # [B, T_src]
        self.trg_mask = self.make_trg_mask(self.trg_input)       # [B, T_trg]

    @staticmethod
    def make_pad_mask(seq, pad_index):
        # seq: [B, T]
        return (seq != pad_index).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

    @staticmethod
    def make_trg_mask(trg_input):
        # trg_input: [B, T, D]
        batch_size, seq_len = trg_input.shape[0], trg_input.shape[1]
        subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len), dtype=torch.bool), diagonal=1)
        return ~subsequent_mask  # [1, T, T] → broadcastable

