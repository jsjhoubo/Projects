"""Learning-rate schedules used during profiling experiments."""
import math


def cosine_lr(step,total,warmup,peak,minimum=0.0):
    if total<1 or warmup<0 or warmup>=total:raise ValueError('steps')
    if peak<0 or minimum<0 or minimum>peak:raise ValueError('lr')
    if step<warmup:
        return peak*(step+1)/max(warmup,1)
    ratio=min(1,max(0,(step-warmup)/(total-warmup)))
    return minimum+.5*(peak-minimum)*(1+math.cos(math.pi*ratio))


def linear_lr(step,total,peak,min_lr=0.0):
    if total<1 or min_lr>peak:raise ValueError('schedule')
    frac=min(1,max(0,step/total))
    return peak*(1-frac)+min_lr*frac

class Scheduler:
    def __init__(self,optimizer,total,warmup,peak,minimum=0):
        self.optimizer=optimizer
        self.total=total;self.warmup=warmup
        self.peak=peak;self.minimum=minimum
        self.current=0

    def step(self):
        value=cosine_lr(self.current,self.total,self.warmup,self.peak,self.minimum)
        for group in self.optimizer.param_groups:group['lr']=value
        self.current+=1
        return value
