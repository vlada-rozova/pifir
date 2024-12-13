import numpy as np
import pandas as pd
import re


def clean_text(text, return_mapping=False):
    """
    Apply simple text preprocessing to reports to 
    remove duplicated whitespaces, 
    add a whitespace after a question mark, 
    remove minus at the start of a sentence,
    remove whitespaces before and after a dash,
    and convert to lower case.
    v2.pet from 10.01.24
    """   
    if return_mapping:
        # Create a list of character position indices
        mapping = list(range(0, len(text)))
    
    # Separate a question mark from the following word with a space
    pattern = re.compile(r"\?(?=\w)")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.end()] + [np.nan]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r"? ", text)
    
    # Remove newline before the hypen inside a hyphenated word
    pattern = re.compile(r"(?<=[a-z])\n-(?=[a-z])")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start()]
            i = m.end() - 1

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r"-", text)
    
    # Remove newline after the hypen inside a hyphenated word
    pattern = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start() + 1]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r"-", text)
    
    # Remove minus at the start of a sentence
    pattern = re.compile(r"\s-(?=\w)")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start() + 1]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" ", text)
    
    # Separate a / from the following word with a space
    pattern = re.compile(r"\s/(?=\w{3,})")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.end()] + [np.nan]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" / ", text)
    
    # Separate a / from the preceding word with a space
    pattern = re.compile(r"(?<=\w{3})/\s")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start()] + [np.nan] + mapping[m.start():m.end()]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" / ", text)
    
    # Add whitespaces around /
    pattern = re.compile(r"(?<=\w{3})/(?=\w{3})")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start()] + [np.nan] + mapping[m.start():m.end()] + [np.nan]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" / ", text)
    
    # Replace multiple whitespaces with a single space
    pattern = re.compile(r"\s+")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start() + 1]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" ", text)
    
    # Rstrip
    text = text.rstrip()
    
    # Convert to lowercase
    text = text.lower()
    
    if return_mapping:
        return mapping
    else:
        return text
    
    