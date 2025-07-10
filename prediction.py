import torch
import numpy as np

import torchtext_compat
from data import Dataset, make_data_iterator
from torchtext_compat import Batch
from helpers import calculate_dtw
from eval_helper import calculate_bt_and_mpjpe, compute_rouge, compute_bleu_scores, compute_wer, compute_fid, mpjae
from model import Model
from constants import PAD_TOKEN, TARGET_PAD


# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None):

    valid_iter = make_data_iterator(
        dataset=data, fields=data.fields, batch_size=batch_size,
        shuffle=True)

    pad_index = model.src_vocab.stoi[PAD_TOKEN]
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []

        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(batch_dict=valid_batch, pad_index=pad_index, target_pad= TARGET_PAD)
            targets = batch.trg_input

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss = model.get_loss_for_batch(is_train=True,
                                                         batch=batch,
                                                         loss_function=loss_function)

                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            output = model.forward(src=batch.src,
                                       trg_input=batch.trg_input[:, :, :225],
                                       src_mask=batch.src_mask,
                                       src_lengths=batch.src_lengths,
                                       trg_mask=batch.trg_mask,
                                       is_train=False)
            
            output = torch.cat((output, batch.trg_input[:, :, 225:]), dim=-1)
            
            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            # Add the source sentences to list, by using the model source vocab and batch indices
            valid_inputs.extend([[model.src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i]))] for i in
                                 range(len(batch.src))])

            # Calculate the full Dynamic Time Warping score - for evaluation
            dtw_score = calculate_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)
            
            #Calculate bt and mpjpe
            bt_f1_scores, mpjpe_scores = calculate_bt_and_mpjpe(targets, output)

            # Can set to only run a few batches
            # if batches == math.ceil(20/batch_size):
            #     break
            batches += 1

        # Dynamic Time Warping scores
        current_valid_score = np.mean(all_dtw_scores)
        #Mean bt and mpjpe scores
        mean_bt_f1 = np.mean(bt_f1_scores)
        mean_mpjpe = np.mean(mpjpe_scores)
        
        # Convert token lists to text for NLP metrics
        ref_texts = [' '.join(tokens) for tokens in valid_inputs]
        hyp_texts = ref_texts  # Replace this with actual model outputs if you have textual predictions

        # NLP metrics
        bleu1, bleu4 = compute_bleu_scores(ref_texts, hyp_texts)
        rouge_l = compute_rouge(ref_texts, hyp_texts)
        #print("Rouge done")
        #rouge_l=0
        wer_score = compute_wer(ref_texts, hyp_texts)
        #print("wer done")

        # FID and MPJAE using skeletal predictions
        #print("Hypotheses:", len(valid_hypotheses), valid_hypotheses[0].shape)
        #print("References:", len(valid_references), valid_references[0].shape)
        
        # Check memory size
        #import psutil
        #import os
        #print("RAM usage before FID:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")


        fake_feats = torch.stack(valid_hypotheses).reshape(len(valid_hypotheses), -1).cpu().numpy()
        real_feats = torch.stack(valid_references).reshape(len(valid_references), -1).cpu().numpy()
        fid_score = compute_fid(fake_feats, real_feats)
        #print("fid done")
        mpjae_score = mpjae(torch.stack(valid_hypotheses), torch.stack(valid_references))
        #print("mpjae done")


    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
       valid_inputs, all_dtw_scores, file_paths, mean_bt_f1, mean_mpjpe, \
       bleu1, bleu4, rouge_l, wer_score, fid_score, mpjae_score