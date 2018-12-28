from utilities import parse_arg, progress_bar
import pathlib, os, inkml, sys, traceback
from trainer.sequence import create_parser
from database_create.methods import *
import pickle
from inkml import graphics as g
import cv2

tracegroups = query("SELECT tracegroup.id, tracegroup.truth "
                 "FROM public.tracegroup, public.writer, public.formula, public.database "
                 "WHERE public.tracegroup.formulaid = public.formula.id "
                 "AND public.formula.writerid = public.writer.id "
                 "AND public.writer.databaseid = public.database.id "
                 "AND public.database.name <> 'CROHME2016_data/Test2016_INKML_GT' "
                 "AND public.database.name <> 'CROHME2014_data/TestEM2014GT' ")

result = {}

for index, tracegroup in enumerate(tracegroups):
    progress_bar("Processing tgroupid", index, len(tracegroups))
    tracegroup_id = tracegroup[0]
    tracegroup_truth = tracegroup[1]

    traces = query("SELECT trace.trace FROM public.trace WHERE trace.tracegroupid=" + str(tracegroup_id))
    if result.get(tracegroup_truth) is None:
        result[tracegroup_truth] = []

    result[tracegroup_truth].append(traces)

pickle.dump(result, open('/Users/balazs/export/tokengroup.pkl', 'wb'))