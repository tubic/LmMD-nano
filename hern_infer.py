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

def save_pdb(coords, sequence, filename, chain_id='A', b_factors=None):
    """
    保存PDB格式的结构文件
    coords: 原子坐标 shape: (L, A, 3), L是序列长度，A是每个残基的原子数
    sequence: 氨基酸序列
    filename: 输出文件名
    chain_id: 链ID
    b_factors: B因子，可选
    """
    if b_factors is None:
        b_factors = np.zeros(len(sequence))
    
    atom_types = ['N', 'CA', 'C', 'O', 'CB']  # 主要的骨架原子和CB
    
    with open(filename, 'w') as f:
        atom_idx = 1
        for res_idx, (residue_coords, aa, b_factor) in enumerate(zip(coords, sequence, b_factors)):
            for atom_type, coord in zip(atom_types, residue_coords):
                if not np.isnan(coord).any():  # 只输出有效的原子坐标
                    x, y, z = coord
                    f.write(f"ATOM  {atom_idx:5d}  {atom_type:<4s}{aa:3s} {chain_id}{res_idx+1:4d}    "
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           {atom_type[0]}  \n")
                    atom_idx += 1
        f.write("END\n")

def save_structure_info(filename, info_dict):
    """
    保存结构相关信息到文本文件
    """
    with open(filename, 'w') as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")

def main(args):
    # 创建输出目录
    output_dir = os.path.join(args.output_path, f"cdr{args.cdr_type}_designs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    modelhub = AntibodyRobertaModel.from_pretrained(
        model_name_or_path=args.cktpath,
        inference=True,
    )
    modelhub.cuda()
    modelhub.eval()
    
    # 设置参数
    num_decode = args.num_decode
    succ = 0
    tot = 0
    tot_ppl = 0.
    sum_rmsd = 0.
    topk = args.topk
    
    # 创建数据集
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
    
    # 记录所有结果的文件
    results_file = os.path.join(output_dir, "all_results.txt")
    with open(results_file, 'w') as f:
        f.write("Index\tNative\tDesigned\tPerplexity\tRMSD\n")
    
    with torch.no_grad():
        for idx, ab in enumerate(tqdm(dataset)):
            new_cdrs, new_ppl, new_rmsd = [], [], []
            sample = dataset.collater([ab] * 1)

            batched_seq, batched_tag, batched_label, paratope, epitope, antibody = sample
            
            # 将数据移到GPU
            batched_seq, batched_tag, batched_label = [item.to(modelhub.device) for item in [batched_seq, batched_tag, batched_label]]
            paratope = [item.to(modelhub.device) for item in paratope]
            epitope = [item.to(modelhub.device) for item in epitope]
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody
            antibody_X, antibody_S, antibody_cdr, padding_mask = antibody_X.to(modelhub.device), antibody_S.to(modelhub.device), antibody_cdr, padding_mask.to(modelhub.device)
            antibody = (antibody_X, antibody_S, antibody_cdr, padding_mask)

            masked_tokens = batched_label.ne(modelhub.task.source_dictionary.pad())
            sample_size = masked_tokens.int().sum()
            
            # 模型推理
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

            out_X = out.bind_X.unsqueeze(0)
            rmsd = compute_rmsd(
                out_X[:, :, 1], bind_X[:, :, 1], bind_mask
            )
            # 将rmsd转换为float
            rmsd_float = rmsd.item()
            new_rmsd.extend([rmsd_float] * num_decode)
            new_cdrs.extend(out.handle)
            new_ppl.extend(out.ppl.tolist())
            
            orig_cdr = ''.join([ALPHABET[i] for i in ab[3]])
            new_res = sorted(zip(new_cdrs, new_ppl, new_rmsd), key=lambda x:x[1])
            
            # 保存结果
            for k, (cdr, ppl, rmsd) in enumerate(new_res[:topk]):
                # 保存结构文件
                coords = out_X[0].cpu().numpy()
                pdb_filename = os.path.join(output_dir, f"design_{idx}_{k}_rmsd{rmsd:.3f}.pdb")
                save_pdb(coords, cdr, pdb_filename)
                
                # 保存结构信息
                info_dict = {
                    "Original_CDR": orig_cdr,
                    "Designed_CDR": cdr,
                    "Perplexity": f"{ppl:.3f}",
                    "RMSD": f"{rmsd:.4f}",
                    "Design_Index": k,
                    "Sample_Index": idx
                }
                info_filename = os.path.join(output_dir, f"design_{idx}_{k}_info.txt")
                save_structure_info(info_filename, info_dict)
                
                # 更新统计信息
                match = [int(a == b) for a,b in zip(orig_cdr, cdr)]
                succ += sum(match)
                tot += len(match)
                tot_ppl += ppl
                sum_rmsd += rmsd
                
                # 将结果写入总结果文件
                with open(results_file, 'a') as f:
                    f.write(f"{idx}\t{orig_cdr}\t{cdr}\t{ppl:.3f}\t{rmsd:.4f}\n")
                    
                print(f"{idx}\t{orig_cdr}\t{cdr}\t{ppl:.3f}\t{rmsd:.4f}")
    
    # 计算最终统计结果
    avg_ppl = tot_ppl / len(dataset) / topk
    avg_rmsd = sum_rmsd / len(dataset) / topk
    recovery_rate = succ / tot if tot > 0 else 0
    
    # 输出最终统计结果
    print(f'PPL = {avg_ppl:.4f}')
    print(f'RMSD = {avg_rmsd:.4f}')
    print(f'Amino acid recovery rate = {recovery_rate:.4f}')
    
    # 保存最终统计结果
    stats_file = os.path.join(output_dir, "final_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Average Perplexity: {avg_ppl:.4f}\n")
        f.write(f"Average RMSD: {avg_rmsd:.4f}\n")
        f.write(f"Amino acid recovery rate: {recovery_rate:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cktpath", type=str, 
                       default='/home/data_cdef/bo_sun/trans_ABGNN_4090/ABGNN/checkpoints/exp2_nanobody/pflen5_iter5_loss1_1_2_lr0.0001_bsz8_seed128/checkpoint_best.pt')
    parser.add_argument("--num_decode", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--cdr_type", type=str, default='3')
    parser.add_argument("--data_path", type=str, default="hern_test_data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="./hern_test_data",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    assert os.path.exists(args.cktpath)
    os.makedirs(args.output_path, exist_ok=True)
    
    main(args)
