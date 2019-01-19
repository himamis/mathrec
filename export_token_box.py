from utilities import parse_arg, progress_bar
import pathlib, os, inkml, sys, traceback
from trainer.sequence import create_parser
from database_create.methods import *
import pickle
from inkml import graphics as g
import cv2
from inkml.graphics import normalize_points
import numpy as np

import pathlib, os, inkml, sys, traceback

export_training = True
export_validating = False

first_part = True
second_part = True
third_part = True
fourth_part = False
fifth_part = True
sixth_part = True


fname = "training"
if not export_training:
    if export_validating:
        fname = "validating"
    else:
        fname = "testing"


if first_part:
    database_query = "public.database.name <> 'CROHME2016_data/Test2016_INKML_GT' AND " \
                     "public.database.name <> 'CROHME2014_data/TestEM2014GT'"
    if not export_training:
        if export_validating:
            database_query = "public.database.name = 'CROHME2014_data/TestEM2014GT'"
        else:
            database_query = "public.database.name = 'CROHME2016_data/Test2016_INKML_GT'"

    formulas = query("SELECT formula.id, formula.formula "
                     "FROM public.formula, public.writer, public.database "
                     "WHERE public.database.id = public.writer.databaseid "
                     "AND public.writer.id = public.formula.writerid "
                     "AND " + database_query)

    # Big for
    result = []
    for index, formula in enumerate(formulas):
        progress_bar("Processing formulas. ", index + 1, len(formulas))
        formula_id, formula_truth = formula

        tgroups = query("SELECT public.tracegroup.id, public.tracegroup.truth "
                            "FROM public.tracegroup "
                            "WHERE public.tracegroup.formulaid = " + str(formula_id))
        skip = False
        tracegroups = []
        for tracegroup_id, tracegroup_truth in tgroups:
            traces = query("SELECT public.trace.trace "
                           "FROM public.trace "
                           "WHERE public.trace.tracegroupid = " + str(tracegroup_id))
            if traces == []:
                skip = True
                break
            tracegroups.append((tracegroup_truth, traces[0]))

        if not skip:
            result.append((formula_truth, tracegroups))

    pickle.dump(result, open('/Users/balazs/token_trace/{}_partial.pkl'.format(fname), 'wb'))

# What I have:
# result = [(formula text, tracegroups)]
# tracegroups = [(token, traces)]
# traces = [[x, y, (z)?]]

# What I want
# result = [([tokens], tracegroups)]
# ...
# traces = [[x, y]]

# base_dir = '/Users/balazs/new_data'
# encoding_vb, decoding_vb = os.path.join(base_dir, 'vocabulary.pkl')
def remove(command, f):
    while command in f:
        start = f.index(command)
        i = start + len(command)
        while f[i] != "{" and f[i] == " ":
            i += 1

        start_end = i


        # Sanity check
        if f[i] != "{":
            f = f[0:start] + " " + f[i:]
            continue

        index = start_end + 1
        close = 1
        while close != 0:
            if f[index] == "{":
                close += 1
            elif f[index] == "}":
                close -= 1
            index += 1
        if start_end - index - 1 > 1:
            raise ValueError(f, "Problem")

        f = f[0:start] + " " + f[start_end+1:index - 1] + " " + f[index:]

    return f

dictionary = open('/Users/balazs/university/WAP/data/dictionary.txt')
dictionary = dictionary.readlines()
words = {'\\to'}
for line in dictionary:
    stripped = line.strip().split()
    words |= {stripped[0]}

if second_part:
    result = pickle.load(open('/Users/balazs/token_trace/{}_partial.pkl'.format(fname), 'rb'))
    from parsy import string, alt
    vocab = reversed(sorted(words | {" "}))
    parser = alt(*map(string, vocab))
    parser = parser.many()

    start_index = 0
    new_result = []
    if export_training:
        skip = {12569}
    elif export_validating:
        skip = {99, 907}
    else:
        skip = {}
    for index, (formula, tracegroups) in enumerate(result[start_index:]):
        if start_index + index in skip:
            continue
        if formula == "$ x ^ {\\frac {p} {q}} = \\sqrt {x ^ {p}} ABOVE {q} = \\sqrt {x ^ {p}} ABOVE {q} $":
            formula = "x ^ {\\frac{p}{q}} = \\sqrt[q]{x^{p}} = \\sqrt[q]{x^{p}}"
        elif formula == "$ \\sqrt {648 + 648} ABOVE {4} + 8 $":
            formula = "\\sqrt[4]{648 + 648} + 8"
        elif formula == "$ \\sqrt {\\sqrt {x} ABOVE {n}} ABOVE {m} $":
            formula = "\\sqrt[m]{\\sqrt[n]{x}}"
        progress_bar("Processing", index + 1, len(result))

        f = formula.strip()
        f = f.replace("\\left(", "(").replace("\\right)", ")")
        f = f.replace("\\left (", "(").replace("\\right )", ")")
        f = f.replace("\\Bigg(", "(").replace("\\Bigg)", ")")
        f = f.replace("\\Big(", "(").replace("\\Big)", ")")
        f = f.replace("\\left |", "|").replace("\\right |", "|")
        f = f.replace("\\left|", "|").replace("\\right|", "|")
        f = f.replace("\\left\\{", "\\{").replace("\\right\\}", "\\}")
        f = f.replace("\\left[", "[").replace("\\right]", "]")
        f = f.replace("\\left [", "[").replace("\\right ]", "]")
        f = f.replace(" \\lt ", "<").replace(" \\gt ", ">")
        f = f.replace("\\lt", "<").replace("\\gt", ">")
        f = f.replace("\\lbrack", "(").replace("\\rbrack", ")")
        f = f.replace("$\\frac", "\\frac")
        f = f.replace("\\dots", "\\ldots")
        f = f.replace("\\!", " ")
        f = f.replace("\\ ", " ")
        f = f.replace("\t", " ")
        f = f.replace("}${", "}{")
        f = f.replace("\\;", " ")
        f = f.replace("\\Pi", "\\pi")
        f = f.replace("\\parallel", " | | ")
        if f[0] == "$":
            f = f[1:]
        elif f.startswith(" $"):
            f = f[2:]
        elif f.startswith("  $"):
            f = f[3:]
        if f[-1] == "$":
            f = f[:-1]

        f = remove("\\mbox", f)
        f = remove("\\mathrm", f)

        try:
            parsed = parser.parse(f)
            parsed = [p for p in parsed if p != ' ']
            print_this = "".join(parsed)
            pass
            # new_result.append((formula[0], f))
        except Exception as e:
            print(index)
            print(formula)
            print(f)
            print(str(e)[-10:])
            exit(1)
            #break

        new_tracegroups = []
        for truth, traces in tracegroups:
            new_traces = []
            for trace in traces:
                new_traces.append([(t[0], t[1]) for t in trace])
            new_tracegroups.append((truth, new_traces))

        new_result.append((parsed, new_tracegroups))

    pickle.dump(new_result, open('/Users/balazs/token_trace/{}.pkl'.format(fname), 'wb'))


if third_part:
    result = pickle.load(open('/Users/balazs/token_trace/{}.pkl'.format(fname), 'rb'))
    for formula, tracegroups in result:

        for index, (truth, traces) in enumerate(tracegroups):
            if truth == "\\lt":
                truth = "<"
            elif truth == "\\gt":
                truth = ">"
            #print(truth)
            assert truth in words
            tracegroups[index] = (truth, traces)

    pickle.dump(result, open('/Users/balazs/token_trace/' + fname, 'wb'))


# Sanity check with !
if False:
    training_result = pickle.load(open('/Users/balazs/token_trace/training.pkl', 'rb'))
    for formula, tracegroups in training_result:
        if "!" in formula:  # Look at tracegroups
            tt = [t for t, _ in tracegroups]
            print(tt)


if fourth_part:
    training_result = pickle.load(open('/Users/balazs/token_trace/{}.pkl'.format(fname), 'rb'))
    input_vocab = set()
    output_vocab = set()
    for formula, tracegroups in training_result:
        output_vocab |= set(formula)
        input_vocab |= set([truth for truth, _ in tracegroups])

    print(input_vocab)
    print(output_vocab)

    all_tokens = input_vocab | output_vocab
    all_tokens = sorted(all_tokens)
    vocabulary = {token: i + 1 for i, token in enumerate(all_tokens)}
    pickle.dump(vocabulary, open('/Users/balazs/token_trace/vocabulary.pkl', 'wb'))

if fifth_part:
    training_result = pickle.load(open('/Users/balazs/token_trace/{}.pkl'.format(fname), 'rb'))
    new_result = []
    for index, (formula, tracegroups) in enumerate(training_result):
        progress_bar("Processing", index + 1, len(training_result))
        tracegroups_only = [t for _, t in tracegroups]
        truth_only = [tr for tr, _ in tracegroups]
        normalize_points(tracegroups_only, unit=True)
        zipped = zip(truth_only, tracegroups_only)
        new_result.append((formula, zipped))

    pickle.dump(new_result, open('/Users/balazs/token_trace/{}_scaled.pkl'.format(fname), 'wb'))


info = np.finfo(np.float32)


if sixth_part:
    training_result = pickle.load(open('/Users/balazs/token_trace/{}_scaled.pkl'.format(fname), 'rb'))
    new_result = []
    for index, (formula, tracegroups) in enumerate(training_result):
        new_tracegroups = []
        progress_bar("Processing", index + 1, len(training_result))
        for token, tracegroup in tracegroups:
            min_x = info.max
            min_y = info.max
            max_x = info.min
            max_y = info.min
            for trace in tracegroup:
                for x, y in trace:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
            new_tracegroups.append((token, (min_x, min_y, max_x, max_y)))

        new_result.append((formula, new_tracegroups))

    pickle.dump(new_result, open('/Users/balazs/token_trace/{}_data.pkl'.format(fname), 'wb'))
