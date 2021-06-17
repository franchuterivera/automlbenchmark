from functools import partial

import pandas as pd
import numpy as np


def bold_formatter(x, value, num_decimals=2):
    """Format a number in bold when (almost) identical to a given value.
    https://github.com/pandas-dev/pandas/issues/38328

    Args:
        x: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.

    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    if round(x, num_decimals) == round(value, num_decimals):
        return f"{{\\bfseries\\num{{{x:.{num_decimals}f}}}}}"
    else:
        return f"\\num{{{x:.{num_decimals}f}}}"


df = pd.DataFrame(np.array([[1.123456, 2.123456, 3.123456, 4.123456],
                            [11.123456, 22.123456, 33.123456, 44.123456],
                            [111.123456, 222.123456, 333.123456, 444.123456],]),
                   columns=['a', 'b', 'c', 'd'])

col_names = ['a in \\si{\\meter}',
             'b in \\si{\\volt}',
             'c in \\si{\\seconds}',
             'd']

# Colums to format with maximum condition and 2 floating decimals
max_columns_2f = ['a']

# Colums to format with minimum condition and 2 floating decimals
min_columns_2f = ['b', 'c']

# Colums to format with minimum condition and 4 floating decimals
min_columns_4f= ['d']

fmts_max_2f = {column: partial(bold_formatter, value=df[column].max(), num_decimals=2) for column in max_columns_2f}
fmts_min_2f = {column: partial(bold_formatter, value=df[column].min(), num_decimals=2) for column in min_columns_2f}
fmts_min_4f = {column: partial(bold_formatter, value=df[column].min(), num_decimals=4) for column in min_columns_4f}

fmts = dict(**fmts_max_2f, **fmts_min_2f, **fmts_min_4f)

print(df.to_latex(
            index=False,
            header=col_names,
            formatters=fmts,
            escape=False))
