#!/usr/bin/python
import operator

from typing import List, Tuple


def _wedit_dist_init(len1: int, len2: int) -> List[List[float]]:
    lev: List[List[float]] = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _wedit_dist_step(
    lev: List[List[float]],
    i: int,
    j: int,
    s1: str,
    s2: str,
    transpositions: bool = False,
):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + _wedit_dist_deletion_cost(c1, c2)
    # skipping a character in s2
    b = lev[i][j - 1] + _wedit_dist_insertion_cost(c1, c2)
    # substitution
    c = lev[i - 1][j - 1] + (_wedit_dist_substitution_cost(c1, c2) if c1 != c2 else 0)

    # transposition
    #    d = c + 1  # never picked by default
    #    if transpositions and last_left > 0 and last_right > 0:
    #        d = lev[last_left - 1][last_right - 1] + i - last_left + j - last_right - 1

    # pick the cheapest
    lev[i][j] = min(a, b, c)  # , d)


def _wedit_dist_backtrace(lev) -> List[Tuple[int, int, float]]:
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment: List[Tuple[int, int, float]] = [(i, j, lev[i][j])]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j, lev[i][j]))
    return list(reversed(alignment))


def _wedit_dist_substitution_cost(c1: str, c2: str) -> float:
    if c1 == " " and c2 != " ":
        return 1000000
    if c2 == " " and c1 != " ":
        return 30
    for c in ",.;-!?'":
        if c1 == c and c2 != c:
            return 20
        if c2 == c and c1 != c:
            return 20
    return 1


def _wedit_dist_deletion_cost(c1: str, c2: str) -> float:
    if c1 == " ":
        return 2
    if c2 == " ":
        return 1000000
    return 0.8


def _wedit_dist_insertion_cost(c1: str, c2: str) -> float:
    if c1 == " ":
        return 1000000
    if c2 == " ":
        return 2
    return 0.8


def wedit_distance_align(s1: str, s2: str) -> List[Tuple[int, int, float]]:
    """
    Calculate the minimum Levenshtein edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost.
    For example, mapping "rain" to "shine" would involve 2
    substitutions, 2 matches and an insertion resulting in
    the following mapping:
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)]
    NB: (0, 0) is the start state without any letters associated
    See more: https://web.stanford.edu/class/cs124/lec/med.pdf

    In case of multiple valid minimum-distance alignments, the
    backtrace has the following operation precedence:

    1. Skip s1 character
    2. Skip s2 character
    3. Substitute s1 and s2 characters

    The backtrace is carried out in reverse string order.

    This function does not support transposition.

    :param s1, s2: The strings to be aligned
    :type s1: str
    :type s2: str
    :rtype: List[Tuple(int, int, float)]
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _wedit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _wedit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                transpositions=False,
            )

    # backtrace to find alignment
    alignment = _wedit_dist_backtrace(lev)
    #    for (i,j,w) in alignment:
    #        print (s1[i - 1], s2[j - 1], w)
    return alignment
