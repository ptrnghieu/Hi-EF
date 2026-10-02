"""Stratified sample for the appraisal-observability pilot (train/val only; test untouched).

Inputs
  --per-mcis   g16_per_mcis.csv from the G16 notebook (design metadata; contains clip-IV-derived voice flags)
  --split      source_folder_split_seed42.csv
  --annot      Hi-EF annotation.csv (only column 1 = subtitle text is copied into the rater package)
Outputs (in --out)
  design_strata.csv   sample_id, stratum, N_h, weight. DESIGN ONLY: never shown to raters.
  rater_package.csv   item_id (shuffled), subtitles I-III, renamed video files <item>_1..3.mp4. No ids, no clip IV, no strata.
  calibration.csv     same format, 15 extra MCIS (not in the main sample) for the instruction-calibration round.
  item_map.csv        item_id -> sample_id, clip ids, source video paths. DESIGN ONLY: used to copy the three files
                      per item into the rater folder; raters never receive it or the dataset folder.
"""
import argparse
import numpy as np
import pandas as pd

SEED, PER_STRATUM, N_CALIB = 20261002, 50, 15

ap = argparse.ArgumentParser()
ap.add_argument('--per-mcis', required=True)
ap.add_argument('--split', required=True)
ap.add_argument('--annot', required=True)
ap.add_argument('--out', default='.')
a = ap.parse_args()

per = pd.read_csv(a.per_mcis, dtype={'src': str})
sp = pd.read_csv(a.split, dtype=str).set_index('sample_id')
ann = pd.read_csv(a.annot, header=None, dtype=str).set_index(0)
sp = sp.loc[per.sample_id]
assert set(sp.split) <= {'train', 'val'}, 'test rows must not enter the pilot'

lab_I, lab_II = per.y_I.values >= 0, per.y_II.values >= 0          # annotation.csv column 7 present for that clip
spk_I, spk_II = per.B_spoke_I.values.astype(bool), per.B_spoke_II.values.astype(bool)
seen = (per.nL1 + per.nL2 + per.nL3).values > 0                  # clip-III listener has >= 1 face in I-III
s1 = (spk_I & lab_I) | (spk_II & lab_II)
s2 = (spk_I | spk_II) & ~s1
s3 = ~(spk_I | spk_II) & seen
s4 = ~(spk_I | spk_II) & ~seen
S = np.select([s1, s2, s3, s4], ['S1', 'S2', 'S3', 'S4'], '')
assert (S != '').all() and (s1.astype(int) + s2 + s3 + s4 == 1).all(), 'strata must partition the eligible set'

rng = np.random.default_rng(SEED)
N = len(per)
rows = []
for h in ['S1', 'S2', 'S3', 'S4']:
    idx = np.where(S == h)[0]
    pick = rng.choice(idx, PER_STRATUM, replace=False)
    rows += [(per.sample_id.iloc[i], h, len(idx), len(idx) / N) for i in pick]
design = pd.DataFrame(rows, columns=['sample_id', 'stratum', 'N_h', 'weight'])
rest = per.sample_id[~per.sample_id.isin(design.sample_id)].values
calib = rng.choice(rest, N_CALIB, replace=False)

text = lambda c: ann.at[c, 1] if c in ann.index and isinstance(ann.at[c, 1], str) else ''
vid = lambda c: 'video/{}/{}.mp4'.format(*c.split('/'))   # resolve the extension against the local dataset copy


def package(ids, prefix):
    ids = list(ids)
    rng.shuffle(ids)
    pkg, mp = [], []
    for k, s in enumerate(ids):
        r, item = sp.loc[s], f'{prefix}{k + 1:03d}'
        pkg.append({'item_id': item, **{f'text{j}': text(r[f'clip{j}']) for j in (1, 2, 3)},
                    **{f'video{j}': f'{item}_{j}.mp4' for j in (1, 2, 3)}})
        mp.append({'item_id': item, 'sample_id': s, **{f'clip{j}': r[f'clip{j}'] for j in (1, 2, 3)},
                   **{f'source{j}': vid(r[f'clip{j}']) for j in (1, 2, 3)}})
    return pd.DataFrame(pkg), pd.DataFrame(mp)


design.to_csv(f'{a.out}/design_strata.csv', index=False)
main_pkg, main_map = package(design.sample_id, 'P')
cal_pkg, cal_map = package(calib, 'C')
main_pkg.to_csv(f'{a.out}/rater_package.csv', index=False)
cal_pkg.to_csv(f'{a.out}/calibration.csv', index=False)
pd.concat([main_map, cal_map]).to_csv(f'{a.out}/item_map.csv', index=False)
print(pd.Series(S).value_counts().sort_index().to_dict(), '| N =', N)
