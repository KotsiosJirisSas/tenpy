"""Tools for handling strings."""
# Copyright (C) TeNPy Developers, GNU GPLv3

from typing import Sequence
import numpy as np

__all__ = ['is_non_string_iterable', 'vert_join', 'to_mathematica_lists', 'format_like_list',
           'join_as_many_as_possible']


def is_non_string_iterable(x):
    """Check if x is a non-string iterable, (e.g., list, tuple, dictionary, np.ndarray)"""
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


def vert_join(strlist, valign='t', halign='l', delim=' '):
    r"""Join multiline strings vertically such that they appear next to each other.

    Parameters
    ----------
    strlist : list of str
        the strings to be joined vertically
    valign : ``'t', 'c', 'b'``
        vertical alignment of the strings: top, center, or bottom
    halign : ``'l', 'c', 'r'``
        horizontal alignment of the strings: left, center, or right
    delim : str
        field separator between the strings

    Returns
    -------
    joined : str
        a string where the strings of strlist are aligned vertically

    Examples
    --------
    >>> from tenpy.tools.string import vert_join
    >>> print(vert_join(['a\nsample\nmultiline\nstring', str(np.arange(9).reshape(3, 3))],
    ...                 delim=' | '))  # doctest: +NORMALIZE_WHITESPACE
    a         | [[0 1 2]
    sample    |  [3 4 5]
    multiline |  [6 7 8]]
    string    |
    """
    # expand tabs, split to newlines
    strlist = [str(s).expandtabs().split('\n') for s in strlist]
    numstrings = len(strlist)
    # number of lines in each string
    numlines = [len(lines) for lines in strlist]
    # maximum number of lines = total number of lines in the resulting string
    totallines = max([0] + numlines)
    # width for each of the strings
    widths = [max([len(l) for l in lines]) for lines in strlist]
    # translate halign to string format mini language
    halign = {'l': '<', 'c': '^', 'r': '>'}[halign]
    fstr = ['{0: ' + halign + str(w) + 's}' for w in widths]

    # create a 2d table
    res = [[' ' * widths[j] for j in range(numstrings)] for i in range(totallines)]

    for j, lines in enumerate(strlist):
        if valign == 't':
            voffset = 0
        elif valign == 'b':
            voffset = totallines - len(lines)
        elif valign == 'c':
            voffset = (totallines - len(lines)) // 2  # rounds to int
        else:
            raise ValueError('invalid valign ' + str(valign))

        for i, l in enumerate(lines):
            res[i + voffset][j] = fstr[j].format(l)  # format to fixed widths[j]

    # convert the created table to a single string
    res = '\n'.join([delim.join(lines) for lines in res])
    return res


def join_as_many_as_possible(msgs: Sequence[str], separator: str, priorities: Sequence[int] = None,
                             max_len: int = None, fill: str = '...') -> str:
    """Like ``separator.join(msgs)`` but truncated if the result is too long.

    We append the ``fill`` value to indicate that entries were omitted.
    By default, the first entries in ``msgs`` are kept.
    If ``priorities`` are specified, the messages are sorted according to their priority first
    (from high to low).
    """
    if len(msgs) == 0:
        return ''
    if max_len is None or sum(len(m) for m in msgs) + len(separator) * (len(msgs) - 1) <= max_len:
        if priorities:
            return separator.join(msgs[n] for n in np.argsort(-np.array(priorities)))
        return separator.join(msgs)

    if priorities is None:
        order = range(len(msgs))
    else:
        order = np.argsort(-np.array(priorities))

    # build arrays whose n-th element represent the resulting length if n+1 msgs are kept
    cum_lengths = np.cumsum([len(msgs[n]) for n in order])
    candidate_lengths = cum_lengths + np.arange(1, len(msgs) + 1) * len(separator) + len(fill)
    num_msgs = np.where(candidate_lengths > max_len)[0][0]

    return separator.join([msgs[n] for n in order[:num_msgs]] + [fill])


def to_mathematica_lists(a):
    """convert nested `a` to string readable by mathematica using curly brackets '{...}'."""
    if isinstance(a, str):
        return '"' + str(a) + '"'
    try:
        iter(a)
        s = "{" + ", ".join([to_mathematica_lists(suba) for suba in a]) + "}"
        return s
    except TypeError:
        if isinstance(a, float) or isinstance(a, complex):
            return str(a).replace('e', '*^').replace('j', ' I')
        return str(a)


def format_like_list(it) -> str:
    """Format elements of an iterable as if it were a plain list.

    This means surrounding them with brackets and separating them by `', '`."""
    return f'[{", ".join(map(str, it))}]'
	
def joinstr(strlist, valign='c', delim=''):
	# TODO, does not handle tabs properly
	# TODO, vary the vertical justification
	""" Join strings with multilines
		no newline at the end of everything

		with tabs, it tries its best to guess where it is (if any of the strings has more than one line)
		"""
	# below are all lists, an item for each one in strlist
	numstr = len(strlist)
	slist = []		# list of string in strlist[i]
	numlines = []	# number of lines in strlist[i]
	#strwidth = []		# max width of strlist[i]
	empty_str = []		# a string with only spaces and tabs, with strwidth number of characters (use for padding)
	for s in strlist:
		if isinstance(s, str):
			list_of_lines = s.split('\n')
		else:
			list_of_lines = str(s).split('\n')		# convert to string
		slist.append(list_of_lines)
		numlines.append(len(list_of_lines))
		list_of_str_lengths = list(len(l) for l in list_of_lines)
		the_longest_line = list_of_lines[np.argmax(list_of_str_lengths)]
		empty_maxlen_liststr = [' '] * len(the_longest_line)
		for i in xrange(len(the_longest_line)):
			if the_longest_line[i] == '\t': empty_maxlen_liststr[i] = '\t'
		empty_str.append( "".join(empty_maxlen_liststr) )
	maxlines = max(numlines)
	s = ""
	for i in xrange(maxlines):
		for t in xrange(numstr):
			if i < int((maxlines - numlines[t]) / 2)  or  i >= int((maxlines - numlines[t]) / 2) + numlines[t]:
				s += empty_str[t]
			else:
				print_str = slist[t][i - int((maxlines - numlines[t]) / 2)]
				s += print_str + empty_str[t][len(print_str):]
			if t < numstr - 1: s += delim
		if i < maxlines - 1: s += '\n'
	return s



