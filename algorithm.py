"""
Algorithm for solving stochastic multi-objective optimization problem

author Perestoronin Daniil
"""

z1 = [
    [0.1, 0.4, 0.5, 0.8, 1.0],
    [0.3, 0.5, 0.6, 0.8, 0.9],
    [0.1, 0.3, 0.5, 1.0, 1.1],
    [0.2, 0.3, 0.4, 1.0, 1.2]
]

z2 = [
    [3, 15, 25, 40, 50],
    [5, 10, 25, 35, 40],
    [5, 15, 25, 30, 60],
    [10, 15, 20, 35, 50]
]


def print_matrix(m):
    for row in m:
        for el in row:
            print(el, end=' ')
        print()


def max_element(m):
    max_el = None
    for row in m:
        for el in row:
            if max_el is None or el > max_el:
                max_el = el
    return max_el


def normalize(m, k=1):
    rows = len(m)
    cols = len(m[0])
    max_el = max_element(m)
    norm_m = [[0 for x in range(cols)] for y in range(rows)]
    for i in range(0, rows):
        for j in range(0, cols):
            norm_m[i][j] = k * m[i][j] / max_el
    return norm_m


print_matrix(z1)
print()
print_matrix(z2)
print()

print_matrix(normalize(z1, 10))
print()
print_matrix(normalize(z2, 10))
print()
