"""
Algorithm for solving stochastic multi-objective optimization problem

author Perestoronin Daniil
"""

import copy
import math
import random

import texttable

criterion = [
    [
        # f(x,w|1)
        [10, 8, 8, 6, 5],
        [7, 7, 6, 5, 2],
        [9, 9, 9, 9, 8],
        [6, 5, 5, 3, 2]
    ],
    [
        # f(x,w|2)
        [99.6, 99.2, 99.0, 98.9, 96.9],
        [99.9, 99.0, 97.9, 96.9, 93.5],
        [99.9, 99.9, 99.0, 98.9, 97.9],
        [99.9, 99.5, 98.9, 95.4, 93.3]
    ],
    [
        # f(x,w|3)
        [98.4, 98.0, 97.3, 94.1, 91.0],
        [97.5, 95.1, 92.5, 90.4, 88.2],
        [98.3, 97.3, 96.0, 95.5, 94.1],
        [95.3, 94.2, 93.5, 92.1, 91.2]
    ],
    [
        # f(x,w|4)
        [8, 8, 7, 7, 6],
        [9, 9, 9, 8, 6],
        [10, 10, 10, 10, 10],
        [10, 10, 9, 8, 7]
    ],
    [
        # f(x,w|5)
        [7, 5, 5, 4, 3],
        [7, 7, 7, 7, 6],
        [10, 9, 8, 8, 7],
        [9, 7, 6, 6, 6]
    ],
    [
        # f(x,w|6)
        [8, 8, 8, 8, 8],
        [10, 10, 10, 6, 5],
        [10, 9, 8, 6, 5],
        [10, 9, 9, 7, 6]
    ],
    [
        # f(x|7)
        [-3500000, -3800000, -4500000, -5000000, -5700000],
        [-4800000, -4800000, -4900000, -4950000, -5000000],
        [-4500000, -4600000, -4700000, -5450000, -6000000],
        [-4890000, -5010000, -5150000, -5205140, -5505140]
    ]
]
prob = [0.2, 0.2, 0.4, 0.15, 0.05]
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


def print_for_latex(m):
    rows = len(m)
    if isinstance(m[0], list):
        cols = len(m[0])
    else:
        cols = 1
    if cols > 1:
        for i in range(0, rows):
            print('$x_' + str(i + 1), end='$ & ')
            for j in range(0, cols):
                print(round(m[i][j], 2), end=' & ' if j != cols - 1 else '\hline')
            print()
    else:
        print('$x_1$', end=' & ')
        for i in range(0, rows):
            print(round(m[i], 2), end=' & ' if i != rows - 1 else '\hline')
        print()


def max_element(m):
    max_el = None
    for row in m:
        for el in row:
            if max_el is None or math.fabs(el) > math.fabs(max_el):
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


def removal_uncertainties(z, p, lam):
    rows = len(z)
    cols = len(lam)
    unc_m = [[0 for x in range(cols)] for y in range(rows)]

    def z1(x):
        xp = [a * b for a, b in zip(x, p)]
        return sum(xp)

    def z2(x):
        v = []
        for j in range(0, len(x)):
            v.append(((x[j] - z1(x)) ** 2) * p[j])
        return math.sqrt(sum(v))

    for i in range(0, rows):
        for j in range(0, cols):
            unc_m[i][j] = (1 - lam[j]) * z1(z[i]) + lam[j] * z2(z[i])
    return unc_m


def lin_conv_opt_pr(unc_crt):
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
                max_c = j + 1
        opt_dic_val.append(max_cr)
        opt_dec.append(max_c)
    return opt_dec, opt_dic_val


def multipl_conv_opt_pr(unc_crt):
    cr_n = len(unc_crt)
    rows = len(unc_crt[0])
    cols = len(unc_crt[0][0])
    opt_dec = []
    opt_dic_val = []
    for i in range(0, cols):
        max_cr = None
        max_c = 0
        for j in range(0, rows):
            cr_mult = 1
            for cr in range(0, cr_n):
                cr_mult = cr_mult * unc_crt[cr][j][i]
            if max_cr is None or max_cr <= cr_mult:
                max_cr = cr_mult
                max_c = j + 1
        opt_dic_val.append(max_cr)
        opt_dec.append(max_c)
    return opt_dec, opt_dic_val


def ideal_point_opt_pr(unc_crt):
    cr_n = len(unc_crt)
    rows = len(unc_crt[0])
    cols = len(unc_crt[0][0])

    def ideal_point():
        ideal_p = [[0 for x in range(cols)] for y in range(cr_n)]
        for i in range(0, cr_n):
            for j in range(0, cols):
                p_max = unc_crt[i][0][j]
                for k in range(1, rows):
                    if p_max < unc_crt[i][k][j]:
                        p_max = unc_crt[i][k][j]
                ideal_p[i][j] = p_max
        return ideal_p

    i_point = ideal_point()

    opt_dec = []
    opt_dic_val = []
    for i in range(0, cols):
        min_cr = None
        min_c = 0
        for j in range(0, rows):
            cr_i_p = 0
            for cr in range(0, cr_n):
                cr_i_p = cr_i_p + (i_point[cr][i] - unc_crt[cr][j][i]) ** 2
            if min_cr is None or min_cr > cr_i_p:
                min_cr = cr_i_p
                min_c = j + 1
        opt_dic_val.append(min_cr)
        opt_dec.append(min_c)
    return opt_dec, opt_dic_val


def calculate_model(crt, prb, lam):
    print(' VARIABLES ')
    print()
    for z in crt:
        print_matrix(z, 'w', 'x')
        print()
        print_for_latex(z)
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
        z_u = removal_uncertainties(z, prb, lam)
        crt_unc.append(z_u)
        print_matrix(z_u, 'l', 'x')
        print()

    print()
    print(' DECISION ')
    print()
    print(' linear convolution :')
    opt_dec, opt_dic_val = lin_conv_opt_pr(crt_unc)
    print_matrix(opt_dec, 'l', 'x')
    print_matrix(opt_dic_val, 'l', 'max')
    print()

    print(' multiplicative convolution :')
    opt_dec, opt_dic_val = multipl_conv_opt_pr(crt_unc)
    print_matrix(opt_dec, 'l', 'x')
    print_matrix(opt_dic_val, 'l', 'max')
    print()

    print(' ideal point convolution :')
    opt_dec, opt_dic_val = ideal_point_opt_pr(crt_unc)
    print_matrix(opt_dec, 'l', 'x')
    print_matrix(opt_dic_val, 'l', 'min')
    print()


def print_calc_for_latex(crt, prb, lam):
    print(' VARIABLES ')
    print()
    for z in crt:
        print_for_latex(z)
        print()
    print()
    print(' NORMALIZED VARIABLES ')
    print()

    crt_norm = []
    for z in crt:
        z_n = normalize(z, 10)
        crt_norm.append(z_n)
        print_for_latex(z_n)
        print()
    print()
    print(' REMOVAL UNCERTAINTIES ')
    print()

    crt_unc = []
    for z in crt_norm:
        z_u = removal_uncertainties(z, prb, lam)
        crt_unc.append(z_u)
        print_for_latex(z_u)
        print()

    print()
    print(' DECISION ')
    print()
    print(' linear convolution :')
    opt_dec, opt_dic_val = lin_conv_opt_pr(crt_unc)
    print_for_latex(opt_dec)
    print_for_latex(opt_dic_val)
    print()

    print(' multiplicative convolution :')
    opt_dec, opt_dic_val = multipl_conv_opt_pr(crt_unc)
    print_for_latex(opt_dec)
    print_for_latex(opt_dic_val)
    print()

    print(' ideal point convolution :')
    opt_dec, opt_dic_val = ideal_point_opt_pr(crt_unc)
    print_for_latex(opt_dec)
    print_for_latex(opt_dic_val)
    print()


def study_stability_solution(crt, prb, lam, eps, opt_pr):
    def get_perturbed_data(data):
        crt_copy = copy.deepcopy(data)
        cr_n = len(crt_copy)
        rows = len(crt_copy[0])
        cols = len(crt_copy[0][0])
        for i in range(0, cr_n):
            for j in range(0, cols):
                for k in range(1, rows):
                    crt_copy[i][k][j] = crt_copy[i][k][j] + random.uniform(-eps, eps)
        return crt_copy

    p_crt = get_perturbed_data(crt)

    crt_norm = []
    for z in p_crt:
        z_n = normalize(z, 10)
        crt_norm.append(z_n)

    crt_unc = []
    for z in crt_norm:
        z_u = removal_uncertainties(z, prb, lam)
        crt_unc.append(z_u)

    if opt_pr == 'linear convolution':
        return lin_conv_opt_pr(crt_unc)

    if opt_pr == 'multiplicative convolution':
        return multipl_conv_opt_pr(crt_unc)

    if opt_pr == 'ideal point convolution':
        return ideal_point_opt_pr(crt_unc)


# calculate_model(criterion, prob, lambd)
print_calc_for_latex(criterion, prob, lambd)

for i in range(0, 10):
    opt_dec, opt_dic_val = study_stability_solution(criterion, prob, lambd, 0.3 * i, 'linear convolution')
    print(0.01 * i, end=' = ')
    print_for_latex(opt_dec)
print()
for i in range(0, 10):
    opt_dec, opt_dic_val = study_stability_solution(criterion, prob, lambd, 0.2 * i, 'multiplicative convolution')
    print(0.01 * i, end=' = ')
    print_for_latex(opt_dec)
print()
for i in range(0, 10):
    opt_dec, opt_dic_val = study_stability_solution(criterion, prob, lambd, 0.01 * i, 'ideal point convolution')
    print(0.01 * i, end=' = ')
    print_for_latex(opt_dec)
