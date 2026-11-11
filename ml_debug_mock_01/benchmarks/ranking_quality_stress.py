"""Reproducible macro-ranking stress workloads. No timing assertions.

Run with: python -m benchmarks.ranking_quality_stress
"""
from __future__ import annotations
import argparse
import random
import statistics
import time
from ml_eval.analytics.quality_advanced import group_map_at_k,group_mrr_at_k
from ml_eval.analytics.ranking_advanced import group_ndcg_at_k,topk_per_query


def build(n=100000,queries=500,seed=31,skew=False):
    rng=random.Random(seed)
    rows=[]
    for i in range(n):
        q=(0 if skew and i%5 else rng.randrange(queries))
        rows.append({"query_id":q,"doc_id":i,
                     "score":rng.choice([0.1,0.2,0.4,0.4,0.9,rng.random()]),
                     "clicked":int(rng.random()<0.13),
                     "relevance":rng.randrange(4)})
    rng.shuffle(rows)
    return rows


def bench(fn,repeats=3):
    durations=[]
    value=None
    for _ in range(repeats):
        start=time.perf_counter()
        value=fn()
        durations.append(time.perf_counter()-start)
    return value,statistics.median(durations)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rows",type=int,default=100000)
    parser.add_argument("--queries",type=int,default=500)
    parser.add_argument("--skew",action="store_true")
    opts=parser.parse_args()
    rows=build(opts.rows,opts.queries,skew=opts.skew)
    for label,fn in [
        ("NDCG@10",lambda:group_ndcg_at_k(rows,10)),
        ("MAP@10",lambda:group_map_at_k(rows,10)),
        ("MRR@10",lambda:group_mrr_at_k(rows,10)),
        ("TopK",lambda:len(topk_per_query(rows,10))),
    ]:
        val,seconds=bench(fn)
        print(f"{label:9} result={val:.6f} elapsed={seconds:.5f}s")
if __name__=="__main__":
    main()
