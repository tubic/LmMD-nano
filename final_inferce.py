import os
import argparse
from fairseq_models import AntibodyRobertaModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from fairseq import checkpoint_utils
from tqdm import tqdm
from fairseq.data import Dictionary

from fairseq_models.data.abgen_dataset import AntibodyComplexDataset
from fairseq_models.data.ab_dictionary import ALPHABET
from fairseq_models.modules.utils import compute_rmsd

def main(args):
    
    modelhub = AntibodyRobertaModel.from_pretrained(
        model_name_or_path=args.cktpath,
        inference=True,
    )
    modelhub.cuda()
    modelhub.eval()
    
    num_decode=args.num_decode
    succ, tot = 0, 0
    tot_ppl = 0.
    dataset = AntibodyComplexDataset(
        data_path=args.data_path,
        split=args.split,
        seq_vocab=modelhub.task.source_dictionary,
        tag_vocab=modelhub.task.tag_source_dictionary,
        cdr_types=[args.cdr_type],
        L_target=20,
        pad_idx=modelhub.task.source_dictionary.pad(),
        mask_idx=modelhub.task.mask_idx,
        max_len=256
    )
    print('PDB', 'Native', 'Designed', 'Perplexity', 'RMSD')
    sum_rmsd = 0.

    # 打开一个文件用于写入
    output_file = open('generated_sequences.txt', 'w')

    with torch.no_grad():
        for ab in tqdm(dataset):
            new_cdrs, new_ppl, new_rmsd = [], [], []
            sample = dataset.collater([ab] * 1)

            pdb_ids, batched_seq, batched_tag, batched_label, paratope, epitope, antibody, antibody_seq = sample
                                
            batched_seq, batched_tag, batched_label = [item.to(modelhub.device) for item in [batched_seq, batched_tag, batched_label]]
            paratope = [item.to(modelhub.device) for item in paratope]
            epitope = [item.to(modelhub.device) for item in epitope]
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody_X.to(modelhub.device), antibody_S.to(modelhub.device), antibody_cdr, padding_mask.to(modelhub.device)
            antibody = (antibody_X, antibody_S, antibody_cdr, padding_mask)

            masked_tokens = batched_label.ne(modelhub.task.source_dictionary.pad())
            sample_size = masked_tokens.int().sum()
            
            out = modelhub.model(
                src_tokens=batched_seq,
                tag_tokens=batched_tag,
                paratope=paratope,
                epitope=epitope,
                antibody=antibody,
                masked_tokens=masked_tokens,
                num_decode=num_decode
            )[0]

            bind_X, bind_S, _, _ = paratope
            bind_mask = bind_S > 0

            orig_cdr = ''.join([ALPHABET[i] for i in ab[4]])
            best_cdr, best_ppl, best_rmsd = None, float('inf'), float('inf')

            for cdr, ppl, rmsd in zip(out.handle, out.ppl, [compute_rmsd(out.bind_X.unsqueeze(0)[:, :, 1], bind_X[:, :, 1], bind_mask)] * len(out.handle)):
                if ppl < best_ppl:
                    best_cdr, best_ppl, best_rmsd = cdr, ppl, rmsd

            # 将原始序列和生成的最佳序列写入文件
            output_file.write(f'PDB ID: {pdb_ids}, Native Seq: {antibody_seq}, Native: {orig_cdr}, Designed: {best_cdr}, Perplexity: {best_ppl:.3f}, RMSD: {best_rmsd.item():.4f}\n')

            match = [int(a == b) for a, b in zip(orig_cdr, best_cdr)]
            succ += sum(match)
            tot += len(match)
            tot_ppl += best_ppl

            sum_rmsd += best_rmsd.item()

    # 关闭文件
    output_file.close()

    print(f'PPL = {tot_ppl / len(dataset):.4f}')
    print(f'RMSD = {sum_rmsd / len(dataset):.4f}')
    print(f'Amino acid recovery rate = {succ / tot:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cktpath", type=str, default='/home/data/bo_sun/ABGNN/checkpoints/exp2/pflen5_iter5_loss1_1_2_lr0.0001_bsz8_seed128/checkpoint_best.pt')
    parser.add_argument("--num_decode", type=int, default=10000)
    parser.add_argument("--cdr_type", type=str, default='3')
    parser.add_argument("--data_path", type=str, default="abgnn/finetune/exp2-hern")
    parser.add_argument("--split", type=str, default="train")
    
    args = parser.parse_args()
    
    assert os.path.exists(args.cktpath)
    
    main(args)
