#!/usr/bin/python
import re


def basic_tokenise(string: str) -> str:
    """Modify string to separate punctuation"""
    # Insert a space *before* punctuation characters
    for char in r',.;?!:)("…/”“„″′‘‚':
        string = re.sub("(?<! )" + re.escape(char) + "+", " " + char, string)
    # Insert a space *after* quotation mark / apostroph characters
    for char in "'\"’”“„″′‘‚":
        string = re.sub(char + "(?! )", char + " ", string)
    return string.strip()
