import math
from statistics import mean

def m_r(x, r):
    n = len(x)
    return 1/n*sum([math.pow(i - mean(x), r) for i in x])

def s(x):
    return math.sqrt(m_r(x, 2))

def sqrt_b1(x):
    return m_r(x, 3) / math.pow(s(x), 3)

def b2(x):
    return m_r(x, 4) / math.pow(s(x), 4)

x = [1, 2, 2, 3, 4, 5, 3, 2, 4, 3, 2, 2, 2]

print(m_r(x, 3))
print(sqrt_b1(x))
print(b2(x))
