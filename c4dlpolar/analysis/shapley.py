import os
import re

import numpy as np


def load_scores(score_dir, prefix='lightning', file_type="eval", score_index=0):
    files = os.listdir(score_dir)
    pattern = re.compile(f"{file_type}-{prefix}-(?P<sources>.+).csv")
    files = [fn for fn in files if pattern.match(fn)]
    
    scores = {}
    for fn in files:
        sources = pattern.match(fn)['sources']
        if sources == 'null':
            sources = ''
        path = os.path.join(score_dir, fn)
        scores_src = np.loadtxt(path)
        if np.ndim(scores_src) == 0:
            scores_src = [scores_src]
        scores[sources] = scores_src[score_index]

    return scores

def load_ens_scores(prefix='lightning', file_type="eval", score_index=0, runs=["run1","run2","run3"]):
    scores={}
    score_dir = "../runs/"
    for run in runs:
        files=os.listdir(os.path.join(score_dir,run,"results"))
        pattern = re.compile(f"{file_type}-{prefix}-(?P<sources>.+).csv")
        files = [fn for fn in files if pattern.match(fn)]
    
        for fn in files:
            sources = pattern.match(fn)['sources']
            if sources == 'null':
                sources = ''
            path = os.path.join(score_dir,run,"results", fn)

            scores_src = np.loadtxt(path)
            if np.ndim(scores_src) == 0:
                scores_src = [scores_src]
            if sources in scores.keys():
                scores[sources] = np.append(scores[sources],scores_src[score_index])
            else:
                scores[sources] = scores_src[0]
    return scores

def shapley_value(scores, source):
    scores = {frozenset(k): v for (k,v) in scores.items()}
    source_keys = [k for k in scores if source in k]
    N = max(len(k) for k in source_keys)
    source = set(source)

    s = []
    for num_sources in range(1, N+1):
        sources_num = [k for k in source_keys if len(k)==num_sources]
        s_num = []
        for sources in sources_num:
            sources_excl = sources - source
            contrib = scores[sources] - scores[sources_excl]
            s_num.append(contrib)
        s.append(np.mean(s_num))

    return np.mean(s)

def get_all_files(run="run1"):
    files={}
    prefixes=("lightning","hail","rain")

    sources_str = ("p","q","null","pq","r","rp","rq","rpq")

    for p in prefixes:
        files[p] = {}
        if p == "rain":
            fn = f"eval-rain-"
        else:
            fn = f"eval_leadtime-{p}-"
        for src in sources_str:
            dir = f"../runs/{run}/results/"
            files[p][src] = os.path.join(dir,fn+src+".csv")

    return files

def load_scores(files, score_index=0):
    scores = {}
    for src,fn in files.items():
        if src == 'null':
            src = ''
        scores_src = np.loadtxt(fn)
        if np.ndim(scores_src) == 0:
            scores_src = [scores_src]
        scores[src] = scores_src[score_index]

    return scores
