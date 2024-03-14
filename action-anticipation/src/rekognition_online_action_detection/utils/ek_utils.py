import os.path as osp
import pandas as pd


def action_to_verb_map(path_to_data,
                       action_offset=False,
                       verb_offset=False):
    actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'))
    a_to_v = {a[1]['id'] + action_offset: a[1]['verb'] + verb_offset
              for a in actions.iterrows()}
    return a_to_v


def action_to_noun_map(path_to_data,
                       action_offset=False,
                       noun_offset=False):
    actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'))
    a_to_n = {a[1]['id'] + action_offset: a[1]['noun'] + noun_offset
              for a in actions.iterrows()}
    return a_to_n
