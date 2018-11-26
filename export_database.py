from utilities import parse_arg, progress_bar
import pathlib, os, inkml, sys, traceback
from trainer.sequence import create_parser
from database_create.methods import *
import pickle
from inkml import graphics as g
import cv2

formulas = query("select formula.id, formula.writerid, formula.formula FROM public.database, public.writer, public.formula WHERE public.formula.writerid = public.writer.id AND public.writer.databaseid = public.database.id AND public.database.name LIKE 'CROHME2016_data/Test2016_INKML_GT';")
graphics = g.Graphics()
vocabulary = set()

result = []
replacemenets = []

vocabulary = pickle.load(open('/Users/balazs/real_data/vocabulary.pkl', 'rb'))
parser = create_parser(vocabulary)

for index, formula in enumerate(formulas):
    progress_bar("Processing images", index, len(formulas))


    formula_id = formula[0]
    tracegroups = query("SELECT * FROM public.tracegroup WHERE formulaid=" + str(formula_id))
    inkml_traces = []
    for tracegroup in tracegroups:
        tracegroup_id = tracegroup[0]
        vocabulary.add(tracegroup[2])
        traces = query("SELECT * FROM public.trace WHERE tracegroupid="+str(tracegroup_id))
        for trace in traces:
            inkml_traces.append(trace[2])

    image = graphics.create_image(inkml_traces)
    truth = formula[2]
    fixed = truth.strip()
    if fixed.endswith("\\!"):
        fixed = fixed[:-2]
    fixed = fixed.strip()
    if fixed.endswith("$"):
        fixed = fixed[:-1]
    fixed = fixed.strip()
    if fixed.startswith("\\ "):
        fixed = fixed[2:]
    fixed = fixed.strip()
    fixed = fixed.replace("$", "")
    fixed = fixed.replace("\\!", "")
    fixed = fixed.strip()

    for repl in replacemenets:
        if repl[0] == fixed:
            fixed = repl[1]
            break

    try:
        parser.parse(fixed)
    except Exception as e:
        print("\nCould not parse i(" + str(index) + ") =" + fixed)
        print("Problem: \t\t" + str(e)[-10:])
        ok = 'f'
        while ok != 'y':
            new_input = input("Expression: ")
            #if new_input == '':
            #    print("using old value")
            #    new_input = fixed
            ok = input("Sure? \'%s\' " % new_input)
            if ok == 'i':
                cv2.imshow("Window", image)
                cv2.waitKey(0)

        replacemenets.append((fixed, new_input))
        fixed = new_input

    result.append((image, fixed))


pickle.dump(result, open('/Users/balazs/test_2016.pkl', 'wb'))
#pickle.dump(vocabulary, open('/Users/balazs/vocabulary.pkl', 'wb'))