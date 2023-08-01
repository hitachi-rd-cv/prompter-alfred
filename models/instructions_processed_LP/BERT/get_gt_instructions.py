#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd

data_dir = 'data/alfred_data/'
split_types = ['unseen', 'seen']
pddl_tmpl = 'val_%s_text_with_ppdl_low_appended.p'
lang_inputs = pickle.load(open('data/input_lang.p', 'rb'))

with open(os.path.join(data_dir, 'alfred_dicts/obj2idx.p'), 'rb') as f:
    obj2idx = pickle.load(f)
    idx2obj = {v:k for k, v in obj2idx.items()}
with open(os.path.join(data_dir, 'alfred_dicts/recep2idx.p'), 'rb') as f:
    recep2idx = pickle.load(f)
    idx2recep = {v:k for k, v in recep2idx.items()}
with open(os.path.join(data_dir, 'alfred_dicts/toggle2idx.p'), 'rb') as f:
    toggle2idx = pickle.load(f)
    idx2toggle = {v:k for k, v in toggle2idx.items()}

inst = pickle.load(open('instruction2_params_val_unseen_appended.p', 'rb'))

for split in split_types:
    inst_dict = dict()
    lang_input = lang_inputs[f"val_{split}"]

    pddl_fname = os.path.join(data_dir, pddl_tmpl % split)
    with open(pddl_fname, 'rb') as f:
        pddl_dict = pickle.load(f)
        pddl_df = pd.DataFrame(pddl_dict)

    for high_lang, low_lang in zip(lang_input["x"], lang_input["x_low"]):
        row = pddl_df.query("x_low == @low_lang")
        if len(row) == 0:
            row = pddl_df.query("x == @high_lang")

        info_dict = dict()
        info_dict["task_type"] = row.y.tolist()[0]
        info_dict["mrecep_target"] = idx2obj[row.mrecep_targets.tolist()[0]]
        info_dict["object_target"] = idx2obj[row.object_targets.tolist()[0]]
        info_dict["parent_target"] = idx2recep[row.parent_targets.tolist()[0]]
        info_dict["sliced"] = row.s.tolist()[0]

        inst_dict[low_lang] = info_dict

    with open(f'instruction2_params_val_{split}_gt.p', 'wb') as f:
        pickle.dump(inst_dict, f)
