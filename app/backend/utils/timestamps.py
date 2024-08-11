from datetime import datetime
import time
from functools import wraps

def get_stamp():
    now = str(datetime.now())
    stamp = now.replace(':','-').replace('.', '-').replace(' ', '-')
    return stamp

def log_time(func):
    """
    A decorator that logs the time taken to execute a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_taken = (end_time - start_time)/60
        msg = f"{func.__name__} executed in {time_taken:.3f} minutes."
        print(msg)
        with open(f'logs/logs.txt', 'a') as f:
            f.write('\n' + get_stamp() + ', ' + msg)

        return result
    return wrapper
