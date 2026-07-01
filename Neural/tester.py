from neural import *
import tqdm 

N = int(input("Amount of bits of input (SCALES QUADRATICALLY): "))
D = int(input("Dimensions of output: "))
def gen_data(input_bits: int, outputs: int):
    data = []
    for i in range(2**input_bits):
        new = "{0:b}".format(i)
        new = "0"*(input_bits - len(new)) + str(new)
        out = []
        for i in range(outputs): out.append(random.choice([0, 1]))
        data.append(([int(x) for x in list(new)], out))
    return data

def flagg(outp, expected, tolerance=0.3):
    if len(outp) != len(expected):
        raise ValueError("Shape mismatch between output and expected output!")

    for i in range(len(outp)):
        if abs(outp[i] - expected[i]) <= tolerance:
            continue # No flag thrown -> False
        else:
            return True
    return False

if __name__ == "__main__":
    print("WARNING: program ran as __main__, training on randomly generated binary dataset...")
    ep = int(input('epochs 10^x: x = '))
    n = Network(shape=[N, 4, 4, D], is_random=True)
    
    data = gen_data(N, D)
    n.radiate()
    
    for tqdm in tqdm.tqdm(range(10**ep)):
        x, y = random.choice(data)
        b_propagation(n, x, y, learning_rate=0.4)
    
    for x, y in data:
        outp = f_propagation(n, x)
        print(x, outp, "expected:", y, "####FLAGGED####"*int(flagg(outp, y)))

