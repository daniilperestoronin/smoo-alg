"""
Algorithm for solving stochastic multi-objective optimization problem

author Perestoronin Daniil
"""

import copy
import math

import texttable

criterion = [
    [
        [0.1, 0.4, 0.5, 0.8, 1.0],
        [0.3, 0.5, 0.6, 0.8, 0.9],
        [0.1, 0.3, 0.5, 1.0, 1.1],
        [0.2, 0.3, 0.4, 1.0, 1.2]
    ],
    [
        [3, 15, 25, 40, 50],
        [5, 10, 25, 35, 40],
        [5, 15, 25, 30, 60],
        [10, 15, 20, 35, 50]
    ]
]

lambd = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def print_matrix(m, col_name='', row_name=''):
    matrix = copy.deepcopy(m)
    if isinstance(matrix[0], list):
        rows = len(matrix)
        cols = len(matrix[0])
    else:
        rows = 1
        cols = len(matrix)
    head = [col_name + str(x + 1) for x in range(cols)]
    head.insert(0, ' ')
    cols_type = ['a' for x in range(cols)]
    cols_type.insert(0, 't')
    table = texttable.Texttable()
    table.set_cols_dtype(cols_type)
    table.set_cols_align(['l' for x in range(cols + 1)])
    table.set_precision(1)
    table.header(head)
    for i in range(0, rows):
        if isinstance(matrix[0], list):
            row = matrix[i]
            row.insert(0, row_name + str(i + 1))
        else:
            row = matrix
            row.insert(0, row_name)
        table.add_row(row)
    print(table.draw())


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


def removal_uncertainties(z, lam):
    rows = len(z)
    cols = len(lam)
    unc_m = [[0 for x in range(cols)] for y in range(rows)]

    def z1(xs):
        return sum(xs)

    def z2(xs):
        return math.sqrt(sum(xs))

    for i in range(0, rows):
        for j in range(0, cols):
            unc_m[i][j] = (1 - lam[j]) * z1(z[i]) + lam[j] * z2(z[i])
    return unc_m


def optimality_principle(unc_crt):
    cr_n = len(unc_crt)
    rows = len(unc_crt[0])
    cols = len(unc_crt[0][0])
    opt_dec = []
    opt_dic_val = []
    for i in range(0, cols):
        max_cr = None
        max_c = 0
        for j in range(0, rows):
            cr_sum = 0
            for cr in range(0, cr_n):
                cr_sum = cr_sum + unc_crt[cr][j][i]
            if max_cr is None or max_cr <= cr_sum:
                max_cr = cr_sum
                max_c = j
        opt_dic_val.append(max_cr)
        opt_dec.append(max_c)
    return opt_dec, opt_dic_val


def calculate_model(crt, lam):
    print(' VARIABLES ')
    print()
    for z in crt:
        print_matrix(z, 's', 'x')
        print()
    print()
    print(' NORMALIZED VARIABLES ')
    print()

    crt_norm = []
    for z in crt:
        z_n = normalize(z, 10)
        crt_norm.append(z_n)
        print_matrix(z_n, 'l', 'x')
        print()
    print()
    print(' REMOVAL UNCERTAINTIES ')
    print()

    crt_unc = []
    for z in crt_norm:
        z_u = removal_uncertainties(z, lam)
        crt_unc.append(z_u)
        print_matrix(z_u, 'l', 'x')
        print()

    print()
    print(' DECISION ')
    print()
    opt_dec, opt_dic_val = optimality_principle(crt_unc)
    print_matrix(opt_dec, 'l', 'x')
    print_matrix(opt_dic_val, 'l', 'max')


calculate_model(criterion, lambd)
