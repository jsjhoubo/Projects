"""Local performance and correctness workloads; opt-in, CPU-only."""
import time,random,statistics
from ml_eval.analytics import classification as c, ranking as r, calibration as a

def timed(fn,repeat=3):
    measurements=[]
    result=None
    for _ in range(repeat):
        started=time.perf_counter()
        result=fn()
        measurements.append(time.perf_counter()-started)
    return result,statistics.median(measurements)

def benchmark_large_binary(n=100000):
    rng=random.Random(17)
    scores=[rng.random() for _ in range(n)]
    labels=[rng.randrange(2) for _ in range(n)]
    results={}
    for name,fn in (("brier",lambda:a.brier(scores,labels)),("ece",lambda:a.ece(scores,labels)),("f1",lambda:c.fbeta(c.threshold_scores(scores),labels))):
        value,seconds=timed(fn)
        results[name]={"value":value,"seconds":seconds}
    return results

def benchmark_ranking(n=2000):
    rng=random.Random(19)
    scores=[rng.random() for _ in range(n)]
    labels=[rng.randrange(4) for _ in range(n)]
    return timed(lambda:r.ndcg(scores,labels,k=100))

if __name__=="__main__":
    print(benchmark_large_binary())
    print(benchmark_ranking())
