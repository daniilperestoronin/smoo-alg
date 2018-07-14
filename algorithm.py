"""
Algorithm for solving stochastic multi-objective optimization problem

author Perestoronin Daniil
"""

import copy
import math
import random

import texttable

f = [
    # f_1
    [0.25, 11, 1, 121, 0.4],
    # f_2
    [12, 16, 2.5, 95, 6.4],
    # f_3
    [0.4, 5, 13, 10, 0.9],
    # f_4
    [2.25, 16, 3, 17, 0.8],
    # f_5
    [0.32, 32, 1, 24, 0.12],
    # f_6
    [0.25, 4, 1, 19, 1],
    # f_7
    [112, 431.3, 1122.1, 121, 132]
]

x = [
    # x^1
    [10, 23, 432, 213, 12],
    # x^2
    [12, 13, 332, 299, 19],
    # x^3
    [8, 15, 465, 313, 9],
    # x^4
    [16, 34, 324, 315, 34],
]

w = [0.2, 0.4, 0.6, 0.8, 1]

prob = [0.2, 0.2, 0.4, 0.15, 0.05]

lambd = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def get_criterion(f, x, w):
    rows = len(f)
    cols = len(w)
    dep = len(x)
    crt = [[[0 for x in range(cols)] for y in range(dep)] for z in range(rows)]
    for i in range(0, rows):
        for j in range(0, dep):
            for k in range(0, cols):
                crt[i][j][k] = sum([fi*xj for fi, xj in zip(f[i], x[j])])*w[k]
    return crt


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


def study_stability_solution(f, x, w, prb, lam, eps, opt_pr):

    def get_perturbed_data(fp, xp, wp):
        f_copy = copy.deepcopy(fp)
        cr_n = len(f_copy)
        rows = len(f_copy[0])
        for i in range(0, cr_n):
            for j in range(0, rows):
                    f_copy[i][j] = f_copy[i][j] + random.uniform(-eps, eps)
        return get_criterion(f_copy, xp, wp)

    p_crt = get_perturbed_data(f, x, w)

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


cr = get_criterion(f, x, w)
calculate_model(cr, prob, lambd)

# print_calc_for_latex(cr, prob, lambd)

# for i in range(0, 10):
#     opt_dec, opt_dic_val = study_stability_solution(f, x, w, prob, lambd, 0.3 * i, 'linear convolution')
#     print(0.01 * i, end=' = ')
#     print(opt_dec)
# print()
#
# for i in range(0, 10):
#     opt_dec, opt_dic_val = study_stability_solution(f, x, w, prob, lambd, 0.2 * i, 'multiplicative convolution')
#     print(0.01 * i, end=' = ')
#     print(opt_dec)
# print()

for i in range(0, 10):
    opt_dec, opt_dic_val = study_stability_solution(f, x, w, prob, lambd, 1 * i, 'ideal point convolution')
    print(0.01 * i, end=' = ')
    print(opt_dic_val)

