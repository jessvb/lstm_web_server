"""
This file runs an unsmoothed maximum likelihood character based text generation model.
This model builds upon the code provided in the following jupyter notebook:
https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139?utm_content=bufferefcf2&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer

Most of the models require an 8 character long history to predict the next character,
though the program is made to be flexible and determine the required charecter length
without the user knowing ahead of time.

If given a string that the model has not seen before, the program will gradually
remove the first character, then the next, and so on of the string until it finds
a string of characters with the same last n-1, then n-2, etc. characters. This
really only applies to the seed text.

Commented out below is the function used to train these model.
"""

"""
from collections import *

def train_char_lm(fname, order=4):
    with open (fname, 'r', encoding='utf-8') as file:
        data = file.read()

    lm = defaultdict(Counter)
    pad = "~" * order
    data = pad + data
    for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1
    def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm
"""

import sys
import json

models = {
    'aliceInWonderland':'max-likelihood-models/AiW_model.json',
    'drSeuss':          'max-likelihood-models/drSeuss_.json',
    'hamlet':           'max-likelihood-models/hamlet_model.json',
    'harryPotter':      'max-likelihood-models/harryPotter_model.json',
    'hungerGames':      'max-likelihood-models/hungerGames_model.json',
    'nancy':            'max-likelihood-models/nancy_model.json',
    'narnia':           'max-likelihood-models/narnia_model.json',
    'shakespeare':      'max-likelihood-models/shakespeare_model.json',
    'tomSawyer':        'max-likelihood-models/tomSawyer_model.json',
    'wizardOfOz':       'max-likelihood-models/WoOz_model.json',
}


from random import random

def generate_letter(lm, history, order):
    """Samples the model's probability distribution and returns the letter"""
    history = history[-order:]
    dist = lm[history]
    x = random()
    for c,v in dist:
        x = x - v
        if x <= 0: return c

def generate_text(lm, order, nletters=1000):
    """The original generate text function. The seed text is just the prefix used in training"""
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
    return "".join(out)

def gen_text (lm, seed, nletters=1000):
    """Same as generate_text, except now handles keys its not seen before"""
    for k in lm.keys():
        order = len(k)

    if len(seed) < order:
        seed = ' ' * order + seed

    history = seed[-order:]
    out = []
    for i in range(nletters):
        if history not in lm:
            if history.lower() in lm:
                history = history.lower()
                break
            def find_suitable_replacement():
                for removed_letters in range (1, order):
                    for k, v in lm.items():
                        if k[-order+removed_letters:] == history[-order+removed_letters:]:
                            return k
            history = find_suitable_replacement()
        c = generate_letter(lm, history, order)
        history = history[-order+1:] + c
        out.append(c);
    return "".join(out)

# loads the saved model
with open(models[sys.argv[1]]) as f:
    model = json.loads(f.read())

# uses the loaded model to generate the results
results = gen_text(model, sys.argv[2], nletters=int(sys.argv[3]))
print(results)
