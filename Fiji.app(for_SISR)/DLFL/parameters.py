# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: c:\Users\UCLA\Desktop\1.52_copy\python files\parameters.py
# Compiled at: 2018-04-25 14:38:46
# Size of source mod 2**32: 149 bytes


def import_parameters(file_path):
    dict = {}
    for line in open(file_path):
        tmp = line.split(' ')
        dict[tmp[0]] = tmp[2][:-1]

    return dict
# okay decompiling parameters.pyc
