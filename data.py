# coding: utf-8
import sys
import os
import io
import os.path
from typing import Optional

# from torchtext.datasets import TranslationDataset
#from torchtext import data
#from torchtext.data import Dataset, Iterator
import torch

#use custom torchtext module
import torchtext_compat
from torchtext_compat import TextField, RawField, Example, make_data_iterator

from torch.utils.data import Dataset

from constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD
from vocabulary import build_vocab, Vocabulary

#prepares and returns train_data, test_data, dev_data datasets, src_vocab and trg_vocab
def load_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    data_cfg = cfg["data"]
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]  #gloss
    trg_lang = data_cfg["trg"]  #skels
    files_lang = data_cfg.get("files", "files")  
    # Train, Dev and Test Path (file prefixes)
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the frame counter required for the model
    trg_size = cfg["model"]["trg_size"] 
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()  #tokenization logic (split() because level is word)

    # Source field is a tokenised version of the source words
    src_field = TextField(init_token=None, eos_token=EOS_TOKEN,      #has preprocessing logic for source sentences
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           use_vocab=True,
                           include_lengths=True,
                           preprocessing=None)

    # Files field is just a raw text field
    files_field = RawField()    #no preprocessing

    #converts a 2d tensor of shape [N,D] (N pose frames each of D dimensions) into a list of N separate 1d tensors 
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


    
    
    
    #stacks batch of sequences into a tensor
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


    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = TextField(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,))*TARGET_PAD,
                               preprocessing=tokenize_features,
                               postprocessing=stack_features)
    
    #print(">>> src_field id:", id(src_field), "| preprocessing:", src_field.preprocessing)
    #print(">>> reg_trg_field id:", id(reg_trg_field), "| preprocessing:", reg_trg_field.preprocessing)

    #print(">>> FIELD TYPES PASSED TO SignProdDataset:")
    #name="src_field"
    #field=src_field
    #print(f"  {name} => use_vocab={getattr(field, 'use_vocab', 'NA')} | preprocessing={getattr(field, 'preprocessing','NA')}")
    #name="reg_trg_field"
    #field=reg_trg_field
    #print(f"  {name} => use_vocab={getattr(field, 'use_vocab', 'NA')} | preprocessing={getattr(field, 'preprocessing','NA')}")
    #name="files_field"
    #field=files_field
    #print(f"  {name} => use_vocab={getattr(field, 'use_vocab', 'NA')} | preprocessing={getattr(field, 'preprocessing', 'NA')}")
    


    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(path=train_path,
                                 exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                                 fields=(src_field, reg_trg_field, files_field),
                                 trg_size=trg_size,
                                 skip_frames=skip_frames,
                                 filter_pred=
                                 lambda x: len(vars(x)['src'])
                                 <= max_sent_length
                                 and len(vars(x)['trg'])
                                 <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
  

    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None]*trg_size

    # Create the Validation Data
    dev_data = SignProdDataset(path=dev_path,
                               exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                               trg_size=trg_size,
                               fields=(src_field, reg_trg_field, files_field),
                               skip_frames=skip_frames)

    # Create the Testing Data
    test_data = SignProdDataset(
        path=test_path,
        exts=("." + src_lang, "." + trg_lang, "." + files_lang),
        trg_size=trg_size,
        fields=(src_field, reg_trg_field, files_field),
        skip_frames=skip_frames)

    #print("Assigning vocab to src_field:", src_vocab)
    src_field.vocab = src_vocab
    #print("src_field.vocab before return:", src_field.vocab)
    # Inject the updated field back into datasets
    for d in [train_data, dev_data, test_data]:
        d.fields['src'].vocab = src_vocab


    return train_data, dev_data, test_data, src_vocab, trg_vocab

# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch

# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)

#replace all calls to make_data_iter with make_data_iterator from torchtext_compat

from torch.utils.data import Dataset
# Main Dataset Class (custom)
class SignProdDataset(Dataset):
    """Defines a dataset for sign language production."""

    def __init__(self, path, exts, fields, trg_size, skip_frames=1, **kwargs):
        """
        Args:
            path: Common path prefix.
            exts: Tuple of (src_ext, trg_ext, files_ext)
            fields: Tuple of (src_field, trg_field, files_field)
            trg_size: Feature dimension per frame (should be 225 for 75 keypoints × 3)
            skip_frames: Number of frames to skip while loading (default 1 = no skipping)
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        self.fields = dict(fields)

        src_path, trg_path, file_path = tuple(os.path.expanduser(path + x) for x in exts)

        self.examples = []

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
             io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
             io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                # Strip away \n
                src_line, trg_line, files_line = src_line.strip(), trg_line.strip(), files_line.strip()

                # Skip if any field is empty
                if not src_line or not trg_line or not files_line:
                    continue

                # Split the target line by space (each part is a 'x,y,z' string)
                triplet_strs = trg_line.split(" ")

                # Parse each triplet into floats and flatten
                flat_trg_line = []
                for triplet in triplet_strs:
                    coords = triplet.split(",")
                    if len(coords) != 3:
                        raise ValueError(f"Invalid triplet format: '{triplet}'")
                    flat_trg_line.extend([float(c) + 1e-8 for c in coords])

                # Now split into frame-wise chunks of size trg_size
                if len(flat_trg_line) % trg_size != 0:
                    raise ValueError(
                        f"Invalid feature vector length {len(flat_trg_line)} not divisible by {trg_size}.\n"
                        f"Maybe a missing or extra triplet in this line: {trg_line[:60]}..."
                    )

                trg_frames = [
                    flat_trg_line[i : i + trg_size]
                    for i in range(0, len(flat_trg_line), trg_size * skip_frames)
                ]

                self.examples.append(
                    Example.fromlist([src_line, trg_frames, files_line], fields)
                )
    

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        #print(f">>> __getitem__ called for index: {index}")
        return self.examples[index]

