import threading
import time
from multiprocessing.dummy import Pool as ThreadPool

def worker(num):
    """thread worker function"""
    print( 'Worker: {:d}'.format(num) )
    return


start = time.time()
 
pool = ThreadPool(4)
my_array = list(range(50000))
pool.map(worker, my_array)

for i in range(50000):
    worker(i)

end = time.time()
print()
print(end - start)