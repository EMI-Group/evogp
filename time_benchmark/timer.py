import time


def timer(
        origin_func,
        *args,
        **kwargs
):
    tic = time.time()
    res = origin_func(*args, **kwargs)
    cost_time = time.time() - tic
    return res, cost_time
