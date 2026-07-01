# this test example uses a pre-trained XOR network to calculate the basic test case of a x-or function
# an xor function should return 0 when 2 values are exual and 1 when they are not

from neural import *

n = load_network("xor.nn")

data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[0])
]

for x, y in data:
    print(x, f_propagation(n, x), "expected:", y)
    print(x, round(f_propagation(n, x)[0]), "-->", y)