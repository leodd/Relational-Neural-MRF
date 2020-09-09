# importing the required module
import timeit


mysetup = '''
import numpy as np
a = np.arange(6000000).reshape(-1, 3)
b = np.arange(90000).reshape(-1, 2)
b = set(map(tuple, b))
print(len(a), len(b))
'''

mycode = '''
res = False
for row in b:
	temp = True
	for idx, v in zip((0, 1), row):
		temp &= a[:, idx] == v
	res |= temp
# print(res)
'''

mycode = '''
res = set(map(tuple, a[:, [0, 1]])) & b
# res = np.apply_along_axis(lambda r: tuple(r) in b, 1, a[:, [0, 1]])
# print(res)
'''

# timeit statement
print(timeit.timeit(setup = mysetup,
					stmt = mycode,
					number = 1))
