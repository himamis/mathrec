try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np


class InkML:

    def __init__(self, string):
        self.root = ET.fromstring(string)
        self.annotations = {}
        self.symbols = {}
        self.format = []
        self.trace_groups = []
        self.ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        self.extract_data()

    @property
    def truth(self):
        return self.annotations.get('truth')

    @property
    def writer(self):
        return self.annotations.get('writer')

    @property
    def age(self):
        return self.annotations.get('age')

    @property
    def gender(self):
        return self.annotations.get('gender')

    @property
    def hand(self):
        return self.annotations.get('hand')

    def _findall_node(self, xpath, node):
        return node.findall(xpath, self.ns)

    def _findall(self, xpath):
        return self._findall_node(xpath, self.root)

    def _extract_root_annotation(self, annotation):
        type = annotation.attrib.get('type')
        if type is None:
            type = annotation.attrib.get('typx')
        if type is None:
            raise ValueError("Annotation has no attrib type")
        self.annotations[type] = annotation.text

    def _extract_channel(self, channel):
        self.format.append((channel.attrib['name'], channel.attrib['type']))

    def _extract_trace(self, trace):
        identifier = trace.attrib["id"]
        if self.symbols.get(identifier) is not None:
            print("Trace ID already exist: " + str(identifier))
            exit()
        point_array = []
        for points in trace.text.split(","):
            point = points.strip().split(" ")
            point = [float(pt) for pt in point]
            point_array.append(point)

        self.symbols[identifier] = point_array

    def _extract_trace_groups(self, trace_group):
        references = []
        truths = []

        for trace_view in self._findall_node("./inkml:traceView", trace_group):
            references.append(trace_view.attrib["traceDataRef"])
        for annotation in self._findall_node("./inkml:annotation[@type='truth']", trace_group):
            truths.append(annotation.text)

        if len(references) > 0:
            if len(truths) == 0:
                print("No truth associated")
                exit()
            elif len(truths) > 1:
                print("Multiple truths associated")
                exit()
            else:
                self.trace_groups.append((truths[0],references))

    def extract_data(self):
        for annotation in self._findall("./inkml:annotation"):
            self._extract_root_annotation(annotation)
        for channel in self._findall("./inkml:traceFormat/inkml:channel"):
            self._extract_channel(channel)
        for trace in self._findall("./inkml:trace"):
            self._extract_trace(trace)
        for trace_group in self._findall(".//inkml:traceGroup"): # Trace groups that have at least one traceView
            self._extract_trace_groups(trace_group)
