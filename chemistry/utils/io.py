import numpy as np


def to_chemcraft_struct(charges, x, comment=''):
    lines = [str(len(charges)), comment]
    for i in range(len(charges)):
        lines.append('{}\t{:.11f}\t{:.11f}\t{:.11f}'.format(charges[i], *x[i * 3: i * 3 + 3]))
    lines.append('')
    return '\n'.join(lines)


def from_chemcraft_struct(s):
    lines = s.split('\n')

    cnt = int(lines[0])

    charges, struct = [], []
    for i in range(cnt):
        splitted = lines[2 + i].split()
        charges.append(splitted[0])
        struct.append(np.array(list(map(float, splitted[1:]))))

    return charges, np.concatenate(struct)


if __name__ == '__main__':
    import numpy as np

    X = np.array([0.000000000, -0.859799324, 0.835503236,
                  0.000000000, -0.100462324, 1.431546236,
                  0.000000000, -1.619136324, 1.431546236])

    print(to_chemcraft_struct([1, 1, 8], X, 'chemcraft format test'))
