# merge all csv files in RNC-seq
import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from RNovA_SeqFiller_Inference.utils.BasicClass import ResidueOnlyPeptide

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FDR stage 2 for sequence results.")
    parser.add_argument("target_directory", help="Folder with target *_rnova_denovo_seq.csv files")
    parser.add_argument("decoy_directory", help="Folder with decoy *.decoy_random0.40_rnova_denovo_seq.csv files")
    parser.add_argument(
        "--pattern",
        default="*rnova_denovo_seq.csv",
        help="Glob pattern for decoy CSV files",
    )
    return parser.parse_args()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = parse_args()
    target_directory = Path(args.target_directory)
    decoy_directory = Path(args.decoy_directory)
    all_csv_files = glob(os.path.join(decoy_directory, '*rnova_denovo_seq.csv'))
    for ff in all_csv_files:
        logger.info(ff)
        sample = ff.replace('.mgf.decoy_random0.40_rnova_denovo_seq.csv', '')
        sample = sample.replace(str(decoy_directory) + os.sep, '')
        decoy_df = pd.read_csv(ff)
        target_df = pd.read_csv(str(target_directory / f"{sample}_rnova_denovo_seq.csv"))
        assert len(decoy_df) ==  len(target_df)
        tagged = []
        t_scores = []
        d_scores = []
        t_score_sum = []
        d_score_sum = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            if len(decoy) > 2 and len(target) > 2:
                t_score = [float(i) for i in target['score'].split(';') if i !='-inf']
                d_score = [float(i) for i in decoy['score'].split(';') if i !='-inf']
                if sum(t_score) > sum(d_score):
                    tagged += [(i, 0) for i in t_score]
                    t_scores += t_score
                else:
                    tagged += [(i, 1) for i in d_score]
                    d_scores += d_score
        merged  = sorted(tagged, key=lambda t: t[0],reverse=True)
        tags = [x[1] for x in merged]
        mid = len(tags) // 2
        fdr = round(sum(tags[:mid])/len(tags[:mid]),2)
        while fdr != 0.01:
            # print(mid)
            if fdr > 0.01:
                mid = mid // 2
            elif fdr < 0.01:
                mid = mid + mid // 2
            fdr = round(sum(tags[:mid])/len(tags[:mid]),2)
        logger.info("Amino Acid level fdr %s model score threshold %s", fdr, merged[mid])
        threshold_score = merged[mid][0]
        # whole path fdr
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            if len(decoy) > 2 and len(target) > 2:
                t_score =[float(i) for i in target['score'].split(';') if i !='-inf']
                t_sum = sum([i for i in t_score if i > threshold_score])
                d_score = [float(i) for i in decoy['score'].split(';') if i !='-inf']
                d_sum = sum([i for i in d_score if i > threshold_score])
                if t_sum > d_sum:
                    tagged.append((t_sum, 0))
                else:
                    tagged.append((d_sum, 1))
        merged  = sorted(tagged, key=lambda t: t[0],reverse=True)
        # print(merged)
        tags = [x[1] for x in merged]
        mid = len(tags) // 2
        fdr = round(sum(tags[:mid])/len(tags[:mid]),2)
        # print(fdr)
        while fdr != 0.01:
            if fdr > 0.01:
                mid = mid // 2
            elif fdr < 0.01:
                mid = mid + mid // 2
            fdr = round(sum(tags[:mid])/len(tags[:mid]),2)
        threshold_path_score = merged[mid][0]
        path_fdr_result = target_directory / f"{sample}_rnova_denovo_path_node_path001fdr.csv"
        path_fdr_result = pd.read_csv(path_fdr_result)
        path_fdr_result = {s:np.array(list(map(float,node.split(';'))))[1:] for s, node in zip(path_fdr_result['scan'],path_fdr_result['node_mass'])}
        # print(len(path_fdr_result))
        new_seq_list = {}
        new_score_list = {}
        
        for _, (i,seq, score) in target_df.iterrows():
            # seq = seq.replace('C[UniMod:4]','C|UniMod:4')
            # seq = seq.replace('M[UniMod:35]','M|UniMod:35')
            # seq = seq.replace('D[UniMod:734]','D|UniMod:734')
            # seq = seq.replace('E[UniMod:734]','E|UniMod:734')
            # print(i, path_fdr_result.keys())
            # print(stop)
            if i in path_fdr_result.keys():
                # print(seq)
                merge_flag = (np.abs(ResidueOnlyPeptide(seq).prefix_mass[None,:]-path_fdr_result[i][:,None])<0.02).any(0)
                # print(merge_flag)
                seq = ResidueOnlyPeptide(seq).sequence_residue_seq
                score = np.array(list(map(float,score.split(';'))))
                
                new_seq = []
                new_score = []
                seq_temp = []
                score_temp = 0
                for a,b,c in zip(merge_flag, seq, score):
                    if a:
                        seq_temp.append(b if len(b)==1 else f'{b[0]}[{b[2:]}]')
                        score_temp += c
                        score_temp = score_temp/len(seq_temp)
                        new_seq.append(seq_temp)
                        new_score.append(score_temp)
                        seq_temp = []
                        score_temp = 0
                    else:
                        seq_temp.append(b if len(b)==1 else f'{b[0]}[{b[2:]}]')
                        score_temp += c
                if not merge_flag.any():
                    new_seq.append(seq_temp)
                    new_score.append(score_temp)
                new_seq_list[i] = new_seq
                new_score_list[i] = new_score
            else:
                seq = ResidueOnlyPeptide(seq).sequence_residue_seq
                score = np.array(list(map(float,score.split(';'))))
                new_seq_list[i] = [[s if len(s)==1 else f'{s[0]}[{s[2:]}]' for s in seq]]
                new_score_list[i] = [score.sum()/len(seq)]
        
        output_path = target_directory / f"{sample}_rnova_denovo_seq001fdr.csv"
        with open(output_path, 'w') as fw:
            fw.write('title,sequence,score\n')
            for i in new_seq_list.keys():
                seq_per_psm, score_per_psm = new_seq_list[i], new_score_list[i]
                seq_temp, score_temp = [], []
                scores = 0
                aa_count = 0

                for seq_block, score in zip(seq_per_psm, score_per_psm):
                    pep = ''.join(seq_block)
                    # print(seq_block, score)
                    if score<threshold_score:
                        seq_mass = ResidueOnlyPeptide(''.join(seq_block)).total_residue_mass
                        seq_temp.append(f'{seq_mass}')
                        score_temp.append(f'{score}')
                    else:
                        seq_temp.append("".join(seq_block))
                        score_temp.append(f"{score}")
                        aa_count += len(''.join(seq_block))
                        scores += score
                if scores < threshold_path_score:
                    continue
                fw.write(f"{i},{' '.join(seq_temp)},{';'.join(score_temp)}\n")

if __name__ == "__main__":
    main()
