"""
More types for argparse.
"""

import argparse


def pos_int(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError(
                'value must be a positive integer')
    return x

def nonneg_int(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError(
                'value must be a non-negative integer')
    return x

