from ._core import *

def sort_events(events,time_key):
    return sorted(events,key=lambda e:(e[time_key],str(e)))

def last_write_wins(events,key_field,time_field):
    out={}
    for event in events:
        key=event[key_field]
        if key not in out or event[time_field]>=out[key][time_field]:
            out[key]=event
    return out

def split_before(events,time_field,cutoff):
    train=[];test=[]
    for event in events:
        (train if event[time_field]<cutoff else test).append(event)
    return train,test

def detect_future_leakage(events,feature_time,label_time):
    return [i for i,event in enumerate(events) if event[feature_time]>event[label_time]]

def sliding_window(values,size,step=1):
    values=list(values)
    if size<1 or step<1:raise ValueError("invalid window parameters")
    return [values[i:i+size] for i in range(0,max(0,len(values)-size+1),step)]

def exponential_moving_average(values,alpha=0.1):
    if not 0<alpha<=1:raise ValueError("invalid alpha")
    iterator=iter(values)
    try:prev=float(next(iterator))
    except StopIteration:return []
    result=[prev]
    for current in iterator:
        prev=alpha*float(current)+(1-alpha)*prev
        result.append(prev)
    return result
