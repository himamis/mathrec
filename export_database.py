from utilities import parse_arg, progress_bar
import pathlib, os, inkml, sys, traceback
from database_create.methods import *
import pickle
from inkml import graphics as g
import cv2

formulas = query("SELECT * FROM public.formula")
graphics = g.Graphics()
vocabulary = set()

result = []

for formula in formulas:
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
    if truth[0] == "$":
        truth = truth[1:]
    if truth[-1] == "$":
        truth = truth[:-1]
    result.append((image, truth))


pickle.dump(result, open('/Users/balazs/images.pkl', 'wb'))
pickle.dump(vocabulary, open('/Users/balazs/vocabulary.pkl', 'wb'))