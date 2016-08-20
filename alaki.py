import itertools

a = {1,2,3}
combs = list(itertools.product(a , a))
print(combs)