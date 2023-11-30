
from trectools import TrecRun, TrecQrel, TrecEval
from tira.rest_api_client import Client
from glob import glob
import pandas as pd
tira = Client()

def load_qrels(dataset):
    return TrecQrel(tira.download_dataset('ir-lab-jena-leipzig-wise-2023', dataset, truth_dataset=True) + '/qrels.txt')

training_qrels = load_qrels('training-20231104-training')
validation_qrels = load_qrels('validation-20231104-training')




def evaluate_run(run, qrels):
    run = TrecRun(run)
    trec_eval = TrecEval(run, qrels)

    return {
        'run': run.get_runid(),
        'nDCG@10': trec_eval.get_ndcg(depth=10),
        'nDCG@10 (unjudgedRemoved)': trec_eval.get_ndcg(depth=10, removeUnjudged=True),
        'MAP': trec_eval.get_map(depth=10),
        'MRR': trec_eval.get_reciprocal_rank()
    }




df = []
for r in glob('runs_milestone3/*.txt'):
    df += [evaluate_run(r, training_qrels)]
df = pd.DataFrame(df)
df.sort_values('nDCG@10', ascending=False)