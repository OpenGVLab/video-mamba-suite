from densevid_eval3.para_evaluate import ANETcaptions

def eval_para(prediction, referneces, verbose=False):
    args = type('args', (object,), {})()
    args.submission = prediction
    args.references = referneces
    args.all_scorer = True
    args.verbose = verbose

    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             verbose=args.verbose,
                             all_scorer=args.all_scorer)
    evaluator.evaluate()
    output = {}

    for metric, score in evaluator.scores.items():
        # print ('| %s: %2.4f'%(metric, 100*score))
        output['para_'+metric] = score
    return output