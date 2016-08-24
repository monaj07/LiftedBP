import itertools
import numpy as np
import scipy

a = {1,2,3}
combs = list(itertools.product(a , a))
print(combs)

a = {'a':3, 'v':4, 'e':6}
c = 1
for k,v in a.items():
    c *= v

print(c)


class d(object):
    a = None

d.a = [3,4]
def func(g):
    g.a = [6]

func(d)
print(d.a)

