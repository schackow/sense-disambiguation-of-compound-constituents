import os
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.corpus import webtext
import spacy
from collections import defaultdict
import json
import glob
from datetime import datetime
import bigjson

def add_head_context(stim_morph, token):
    if token.head.is_stop or token.text.lower() in stim_morph:
        if token.head.dep_ != 'ROOT':
            add_head_context(stim_morph, token.head)
    else:
        head_morph = get_morphy(token.head.text.lower())
        context = str(token.dep_ + ',' + head_morph + ',' + token.head.pos_)
        all_context_occurences[context][stim_morph] = all_context_occurences[context][stim_morph] + 1
        all_stim_occurences[stim_morph] = all_stim_occurences[stim_morph] + 1
        
def add_children_contexts(stim_morph, token, token_dep):
    for child in token.children:
        if child.is_stop or child.text.lower() in stim_morph:
            add_children_contexts(stim_morph, child, token_dep)
        else:
            child_morph = get_morphy(child.text.lower())
            context = str(child_morph + ',' + token_dep + ',' + child.pos_)
            all_context_occurences[context][stim_morph] = all_context_occurences[context][stim_morph] + 1
            all_stim_occurences[stim_morph] = all_stim_occurences[stim_morph] + 1

def get_morphy(stim):
    morphy = wn.morphy(stim)
    if morphy is not None:
        return morphy
    return stim

def get_sentences():
    sents= [sent for sent in gutenberg.sents()]
    sents.extend([sent for sent in brown.sents()])
    sents.extend([sent for sent in reuters.sents()])
    sents.extend([sent for sent in webtext.sents()])
    print('length of sentences: ' + str(len(sents)))
    return sents

def get_sentences(start, stop):
    fail = False
    with open('data/wiki_sentences_copy.json', 'rb') as f:
        j = bigjson.load(f)
        sents = []
        for index in range(start, stop):
            try:
                sents.append(j[index])
            except IndexError as error:
                print(error)
                fail = True
                break
            except Exception as exception:
                print(exception)

    sents = [nltk.word_tokenize(s) for s in sents]
    print('length of sentences: ' + str(len(sents)))
    return sents, fail

nlp = spacy.load('en_core_web_sm')

# create contexts
all_stims = []
with open('data/stims_c1_c1_synsets', 'r') as wcw:
    all_stims = wcw.read().splitlines()
all_stims = [get_morphy(stim) for stim in all_stims]
print('Length of stims to be found: {stims}'.format(stims = len(all_stims)))
i = 0
all_context_occurences = defaultdict(lambda: defaultdict(int))
all_stim_occurences = defaultdict(int)
all_word_occurences = 0
fail = False
ind = 0
while not fail:
    ind = ind + 1
    sents, fail = get_sentences((ind - 1) * 1000, ind * 1000)
    for index in range(len(sents)):
        sent = sents[index]
        strsent = ' '.join(sent)
        sentset = set(get_morphy(word.lower()) for word in sent)
        all_word_occurences = all_word_occurences + len(sent)
        parse = None
        for stim_morph in all_stims:
            if stim_morph in sentset:
                i = i + 1
                if parse is None:
                    parse = nlp(strsent)
                for token in parse:
                    token_morph = get_morphy(token.text.lower())
                    if token_morph == stim_morph:
                        if token.is_stop:
                            continue
                        add_head_context(stim_morph, token)
                        add_children_contexts(stim_morph, token, token.dep_)
            elif stim_morph in strsent.lower():
                i = i + 1
                if parse is None:
                    parse = nlp(strsent)
                #for chunk in parse.noun_chunks:
                #    if chunk.text.lower() == token_morph or chunk.text.lower().replace(" ", "-") == token_morph or chunk.text.lower().replace(" ", "") == token_morph:
                for token in parse:
                    token_morph = get_morphy(token.text.lower())
                    if token.dep_ == "compound" and token_morph in stim_morph:
                        if token.is_stop:
                            continue
                        add_head_context(stim_morph, token)
                        add_children_contexts(stim_morph, token, token.dep_)


        #if index % 1000 == 0:
    with open('data/contexts' + str(ind) + '.json', 'w') as context_file:
        json.dump(all_context_occurences, context_file)

    with open('data/stims' + str(ind) + '.json', 'w') as stim_file:
        json.dump(all_stim_occurences, stim_file)

    with  open('data/words' + str(ind) + '.json', 'w') as word_file:
        json.dump(all_word_occurences, word_file)
    print(str(ind) + '  ' + str(i) + ' ' + str(len(all_context_occurences)) + ' ' + str(len(all_stim_occurences)))
    i = 0
    all_context_occurences = defaultdict(lambda: defaultdict(int))
    all_stim_occurences = defaultdict(int)
    all_word_occurences = 0

print('building contexts complete, now cleanup...')

#cleanup contexts
all_context_occurences = defaultdict(lambda: defaultdict(int))
all_stim_occurences = defaultdict(int)
all_word_occurences = 0
filenames = glob.glob('data/contexts*.json')
print('Combining contexts: {files}'.format(files = filenames))
for filename in filenames:
    with open(filename, 'r') as f:
        for context, value in json.loads(f.read()).items():
            for stim, count in value.items():
                all_context_occurences[context][stim] = all_context_occurences[context][stim] + count
    os.remove(filename)

with open('data/contexts.json', 'w') as context_file:
    json.dump(all_context_occurences, context_file)

for filename in glob.glob('data/stims*.json'):
    with open(filename, 'r') as f:
        for stim, value in json.loads(f.read()).items():
            all_stim_occurences[stim] = all_stim_occurences[stim] + value
    os.remove(filename)

with open('data/stims.json', 'w') as stim_file:
    json.dump(all_stim_occurences, stim_file)
    
for filename in glob.glob('data/words*.json'):
    with open(filename, 'r') as f:
        all_word_occurences = all_word_occurences + json.loads(f.read())
    os.remove(filename)

with  open('data/words.json', 'w') as word_file:
    json.dump(all_word_occurences, word_file)
