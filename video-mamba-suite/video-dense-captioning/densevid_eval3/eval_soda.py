import sys
import os
from os.path import dirname, abspath

pdvc_dir = dirname(dirname(abspath(__file__)))
sys.path.append(pdvc_dir)
sys.path.append(os.path.join(pdvc_dir, 'densevid_eval3/SODA'))

import numpy as np
from densevid_eval3.SODA.soda import SODA
from densevid_eval3.SODA.dataset import ANETCaptions
from densevid_eval3.eval_para import eval_para

def eval_tool(prediction, referneces=None, metric='Meteor', soda_type='c', verbose=False):

    args = type('args', (object,), {})()
    args.prediction = prediction
    args.references = referneces
    args.metric = metric
    args.soda_type = soda_type
    args.tious = [0.3, 0.5, 0.7, 0.9]
    args.verbose = verbose
    args.multi_reference = False

    data = ANETCaptions.from_load_files(args.references,
                                        args.prediction,
                                        multi_reference=args.multi_reference,
                                        verbose=args.verbose,
                                        )
    data.preprocess()
    if args.soda_type == 'a':
        tious = args.tious
    else:
        tious = None
    evaluator = SODA(data,
                     soda_type=args.soda_type,
                     tious=tious,
                     scorer=args.metric,
                     verbose=args.verbose
                     )
    result = evaluator.evaluate()

    return result

def eval_soda(p, ref_list,verbose=False):
    score_sum = []
    for ref in ref_list:
        r = eval_tool(prediction=p, referneces=[ref], verbose=verbose, soda_type='c')
        score_sum.append(r['Meteor'])
    soda_avg = np.mean(score_sum, axis=0) #[avg_pre, avg_rec, avg_f1]
    soda_c_avg = soda_avg[-1]
    results = {'soda_c': soda_c_avg}
    return results


if __name__ == '__main__':

    p_new = '../save/old/cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358/2021-08-21-21-47-13_debug_2021-08-21_20-46-20_epoch8_num4917_score0_top1000.json'
    p_vitr= '../save/old/cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358/2021-08-21-21-47-20_cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358_epoch8_num4917_score0_top1000.json.tmp'

    for p in [p_new, p_vitr]:
        print('\n')
        print(p)
        ref_list = ['data/anet/captiondata/val_1.json', 'data/anet/captiondata/val_2.json']
        score=eval_soda(p, ref_list, verbose=False)
        print(score)
        para_score = get_para_score(p, referneces=['../data/anet/captiondata/para/anet_entities_val_1_para.json', '../data/anet/captiondata/para/anet_entities_val_2_para.json'])
        print(para_score)


        # metric = ['Meteor', 'Cider']
        # score_type = ['standard_score', 'precision_recall', 'paragraph_score']
        # dvc_score = soda3.eval_tool(predictions=[p], referneces=ref_list, metric=metric,score_type=score_type)[0]
