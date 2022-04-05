from collections import defaultdict
import json
import math
from datetime import datetime
from scipy.sparse import dok_matrix
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix

context_file = open("data/contexts.json", "r")
all_context_occurences = defaultdict(lambda: defaultdict(int), {key:defaultdict(int, value) for key, value in json.loads(context_file.read()).items()})
context_file.close()
all_contexts = list(all_context_occurences.keys())

stim_file = open("data/stims.json", "r")
all_stim_occurences = defaultdict(int, json.loads(stim_file.read()))
stim_file.close()

word_file = open("data/words.json", "r")
all_word_occurences = int(word_file.read())
word_file.close()

all_stims = list(all_stim_occurences.keys())
pcs = defaultdict(int, {context:(sum(total.values()) / all_word_occurences) for context, total in all_context_occurences.items()})
pws = defaultdict(float, {stim:(total / all_word_occurences) for stim, total in all_stim_occurences.items()})
sums = defaultdict(int, {context: sum(value for value in all_context_occurences[context].values()) for context in all_contexts})

k = 20
def APPMI(pwc, pw, pc):
    if pwc == 0.0:
        return 0.0
    return max(0.0, math.log((pwc * pwc) /((pw * pw) * pc) + k))

def get_morphy(stim):
    morphy = wn.morphy(stim)
    if morphy is not None:
        return morphy
    return stim

def get_full_inputs():
    inputs = []
    with open("data/fullexpansionsamples.txt", 'r') as inputfile:
        lines = json.loads(inputfile.read())
        for line in lines:
            input = {}
            input['stim'] = line['stim']
            input['const'] = line['const']
            input['syns'] = [word for word in line['syns'] if line['syns'][word] == 1]
            inputs.append(input)
    return inputs

def expand_set(S, n, p, MVC, MCV, set_context_keys, SET):
    len_S = len(S)
    scores = []
    for i, c in enumerate(set_context_keys):
        a = sum(MVC[SET.index(word), i] for word in S)
        try:
            f = sum(1 if MCV[i, SET.index(word)] > 0 else 0 for word in S) / len_S
        except :
            print(MCV.shape, S, SET, i)
            exit()
        fp = f**p
        scores.append((c, (fp * a), fp))

    scores.sort(key=lambda tup: tup[1], reverse = True)

    W = {}
    for c in scores[:n]:
        index_c = set_context_keys.index(c[0])
        W[index_c] = c[2]

    AT = MVC.transpose()
    AT_NEU = csr_matrix((len(set_context_keys), len(SET)))

    for index in range(AT.shape[0]):
        if index in W:
            AT_NEU[index] = AT[index] * W[index]
    AWB = AT_NEU.transpose().dot(MCV)
    K = dok_matrix((len(SET), 1))
    for index, word in enumerate(SET):
        if word in S:
            K[index] = 1
    E = AWB.dot(K)

    expansion = [e for e in zip(E, SET)]
    expansion.sort(key=lambda tup: tup[0], reverse = True)
    return expansion[:n]

def execute_single_matches():
    with open("data/expansions_small_matches.txt", "a") as file:
                file.write('[\n')
    for count, comp in enumerate(get_full_inputs()):
        stim = get_morphy(comp['stim'])
        const = get_morphy(comp['const'])
        correctsyns = comp['syns']
        syns = [get_morphy(synset.name().split('.')[0]) for synset in wn.synsets(comp['const'])]
        SET = [stim, const]
        SET.extend(syns)
        if (const not in correctsyns):
            SET.remove(const)
        SET = list(set(SET))
        set_contexts = {}
        for ct in all_contexts:
            context = all_context_occurences[ct]
            if any(syn in context for syn in SET):
                set_contexts[ct] = context
        A = dok_matrix((len(SET), len(set_contexts)))
        set_context_keys = list(set_contexts.keys())

        notin = [s for s in SET if s not in all_stims]
        if any(notin):
            with open("data/expansions_small_matches.txt", "a") as file:
                file.write('\n{{"count":{count},\n"set":{S},\n'.format(count = count, S = [stim, const]))
                file.write('"time":"{time}",\n'.format(time=datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
                file.write('"invalid": "synonyms {notin} not in stims"\n'.format(notin = notin))
                file.write('},\n')
            continue

        for i, st in enumerate(SET):
            for j, context in enumerate(set_context_keys):
                try:
                    pwc = set_contexts[context][st] / sums[context]
                    A[i, j] = APPMI(pwc, pws[st], pcs[context])
                except:
                    print(str(i) + " " + st + " " + str(j) + " " + str(context))

        B = dok_matrix((len(set_contexts), len(SET)))
        for i, context in enumerate(set_context_keys):
            for j, st in enumerate(SET):
                try:
                    pcw = set_contexts[context][st] / all_stim_occurences[st]
                    B[i, j] = APPMI(pcw, pcs[context], pws[st])
                except:
                    print(str(i) + " " + str(context) + " " + str(j) + " " + st)

        sorted = expand_set([stim, const], len(SET), 1.5, A, B, set_context_keys, SET)
        words = [word[1] for word in sorted]

        if stim in words: words.remove(stim)
        if const in words: words.remove(const)
        with open("data/expansions_small_matches.txt", "a") as file:
            file.write('\n{{"count":{count},\n"set":{S},\n'.format(count = count, S = [stim, const]))
            file.write('"time":"{time}",\n'.format(time=datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            
            if any(word in correctsyns and words.index(word) < 3 for word in words):
                file.write('"correct1":{correct}\n'.format(correct=words))
                for syn in correctsyns:
                    file.write(',"{corsyn}":{place}\n'.format(corsyn=syn, place=words.index(syn)))
            else:
                file.write('"incorrect1":{incorrect}\n'.format(incorrect=words))
                for syn in correctsyns:
                    file.write(',"{corsyn}":{place}\n'.format(corsyn=syn, place=words.index(syn)))
            file.write('},\n')

    with open("data/expansions_small_matches.txt", "a") as file:
        file.write('\n]')

execute_single_matches()
