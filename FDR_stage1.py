# merge all csv files in RNC-seq
import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FDR stage 1 from target/decoy CSVs.")
    parser.add_argument("target_directory", help="Folder with target *_rnova_denovo_path.csv files")
    parser.add_argument("decoy_directory", help="Folder with decoy *_rnova_denovo_path.csv files")
    
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
    output_directory = Path(args.target_directory)

    all_csv_files = glob(os.path.join(target_directory, '*rnova_denovo_path.csv'))
    df = pd.concat((pd.read_csv(f) for f in all_csv_files), ignore_index=True, sort=False)
    for ff in all_csv_files:
        logger.info(ff)
        sample = ff.split('/')[-1].replace('_rnova_denovo_path.csv', '')
        logger.info(sample)
        target_df = pd.read_csv(ff)
        decoy_df = pd.read_csv(decoy_directory / f"{sample}.mgf.decoy_random0.40_rnova_denovo_path.csv")
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
                t_score = [float(i) for i in target['score'].split(';')][1:-1]
                d_score = [float(i) for i in decoy['score'].split(';')][1:-1]
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
        logger.info("node-level fdr %s q_value threshold %s", fdr, merged[mid])
        threshold_score = merged[mid][0]
        # whole path fdr
        tagged = []
        for idx in range(len(decoy_df)):
            decoy = decoy_df.iloc[idx]
            target = target_df.iloc[idx]
            if len(decoy) > 2 and len(target) > 2:
                t_score = [float(i) for i in target['score'].split(';')][1:-1]
                t_sum = sum([i for i in t_score if i > threshold_score])
                d_score = [float(i) for i in decoy['score'].split(';')][1:-1]
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
        output_path = output_directory / f"{sample}_rnova_denovo_path_001fdr.csv"
        with open(output_path, 'w') as f:
            f.write('scan,node_mass,node_score\n')
            for _, (scan, node_mass, node_score, _) in target_df.iterrows():
                node_mass, node_score = np.fromstring(node_mass, sep=';'), np.fromstring(node_score, sep=';')
                mask = node_score>threshold_score
                node_mass, node_score = node_mass[mask], node_score[mask]
                if sum(node_score[1:-1]) < threshold_path_score:
                    continue
                node_mass = np.array2string(node_mass,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                node_score = np.array2string(node_score,separator=';',formatter={'float_kind': lambda x: f"{x:.4f}"},max_line_width=999999).strip('[]')
                # if (~mask).all() or mask.sum()<3: continue
                f.write(f'{scan},{node_mass},{node_score}\n')

if __name__ == "__main__":
    main()
