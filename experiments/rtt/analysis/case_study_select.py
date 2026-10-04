# Candidate MCIS for the case study / motivation figure, from the SAVED G10 test predictions (descriptive only;
# no model is trained or selected here). Usage:
#   G10_DIR=<dir with RoleNet.npy, PaperBest.npy, RoleNet_noRole.npy, sample_id.npy> \
#   ANNOT=<annotation.csv> SPLIT=<source_folder_split_seed42.csv> OUT=<csv> python case_study_select.py
import os
import numpy as np, pandas as pd

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}
D = os.environ.get('G10_DIR', '.')
ld = lambda n: np.load(os.path.join(D, f'{n}.npy'), allow_pickle=True)
sid = ld('sample_id')
P = {k: ld(f).mean(0) for k, f in (('RoleNet', 'RoleNet'), ('PaperBest', 'PaperBest'), ('noRole', 'RoleNet_noRole'))}
ann = pd.read_csv(os.environ['ANNOT'], header=None, dtype=str).set_index(0)
sp = pd.read_csv(os.environ['SPLIT'], dtype=str).set_index('sample_id').loc[sid]
yB, yA = sp.clip4_emotion.map(E2I).values, sp.clip3_emotion.map(E2I).values
txt = lambda c: ann.at[c, 1] if c in ann.index and isinstance(ann.at[c, 1], str) else ''
cert = sp.clip4.map(lambda c: ann.at[c, 8] if c in ann.index else None).values
pred = {k: v.argmax(1) for k, v in P.items()}
r = np.arange(len(sid))
df = pd.DataFrame({
    'sample_id': sid, 'episode': sp.source_folder.values, 'A_III': [EMO[i] for i in yA], 'B_IV': [EMO[i] for i in yB],
    'mirror': yA == yB, 'certainty_IV': cert,
    'RoleNet': [EMO[i] for i in pred['RoleNet']], 'PaperBest': [EMO[i] for i in pred['PaperBest']],
    'noRole': [EMO[i] for i in pred['noRole']],
    'p_true_RoleNet': P['RoleNet'][r, yB].round(3), 'p_true_PaperBest': P['PaperBest'][r, yB].round(3),
    'text_I': sp.clip1.map(txt).values, 'text_II': sp.clip2.map(txt).values, 'text_III': sp.clip3.map(txt).values,
    'text_IV': sp.clip4.map(txt).values, 'clip1': sp.clip1.values, 'clip2': sp.clip2.values, 'clip3': sp.clip3.values,
    'clip4': sp.clip4.values})
df['margin'] = (df.p_true_RoleNet - df.p_true_PaperBest).round(3)
rn_ok, pb_ok = pred['RoleNet'] == yB, pred['PaperBest'] == yB
df['category'] = np.select(
    [rn_ok & ~pb_ok & ~df.mirror, rn_ok & ~pb_ok & df.mirror, ~rn_ok & ~df.mirror & np.isin(yB, [1, 2, 6]), ~rn_ok & pb_ok],
    ['win_shift', 'win_mirror', 'fail_shift_rare', 'loss'], 'other')
print("counts:", df.category.value_counts().to_dict(), "| test MCIS", len(df))
print("RoleNet right & PaperBest wrong:", int((rn_ok & ~pb_ok).sum()), "| PaperBest right & RoleNet wrong:", int((~rn_ok & pb_ok).sum()))
cols = ['sample_id', 'A_III', 'B_IV', 'RoleNet', 'PaperBest', 'p_true_RoleNet', 'p_true_PaperBest', 'certainty_IV',
        'text_II', 'text_III', 'text_IV']
for cat in ('win_shift', 'win_mirror', 'fail_shift_rare'):
    sub = df[df.category == cat].sort_values(['certainty_IV', 'margin'], ascending=[True, cat.startswith('fail')])
    print(f"\n== {cat}: top candidates ==")
    with pd.option_context('display.max_colwidth', 60, 'display.width', 250):
        print(sub[cols].head(8).to_string(index=False))
df.to_csv(os.environ.get('OUT', 'case_study_candidates.csv'), index=False)
