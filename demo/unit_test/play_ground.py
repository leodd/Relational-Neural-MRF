# importing the required module
import timeit

# code snippet to be executed only once
mysetup = '''
import numpy as np
a = np.arange(600000).reshape(-1, 3)
'''

# code snippet whose execution time is to be measured
mycode = '''
set(map(tuple, a))
'''

# timeit statement
print(timeit.timeit(setup = mysetup,
					stmt = mycode,
					number = 1))
