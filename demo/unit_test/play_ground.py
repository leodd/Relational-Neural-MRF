# importing the required module
import timeit

# code snippet to be executed only once
mysetup = '''
import numpy as np
a = np.arange(60000).reshape(-1, 3)
'''

# code snippet whose execution time is to be measured
mycode = '''
res = True
for idx, v in zip((0, 1, 2), (6, 7, 8)):
	res &= a[:, idx] == v
res = a[res]
'''

# timeit statement
print(timeit.timeit(setup = mysetup,
					stmt = mycode,
					number = 1))
