import itertools
import numpy as np
import scipy

a = {1,2,3}
combs = list(itertools.product(a , a))
print(combs)