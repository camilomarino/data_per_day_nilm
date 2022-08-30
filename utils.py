import os
import sys


def block_print():
    """
    Disable prints.
    Extracted from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    sys.stdout = open(os.devnull, "w")


def enable_print():
    """
    Enable prints.
    """
    sys.stdout = sys.__stdout__
