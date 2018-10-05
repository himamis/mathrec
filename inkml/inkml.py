try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np


class InkML:

    def __init__(self, string):
        self.root = ET.fromstring(string)
        self.truth = ""
        self.symbols = []
        self.format = []
        self.extract_data()

    def extract_data(self):
        for child in self.root:
            if child.tag == '{http://www.w3.org/2003/InkML}annotation' and child.attrib['type'] == 'truth':
                self.truth = child.text
            elif child.tag == '{http://www.w3.org/2003/InkML}traceFormat':
                for traceFormatChild in child:
                    if traceFormatChild.tag == '{http://www.w3.org/2003/InkML}channel':
                        self.format.append((traceFormatChild.attrib['name'], traceFormatChild.attrib['type']))
            elif child.tag == '{http://www.w3.org/2003/InkML}trace':
                point_array = []
                for points in child.text.split(","):
                    point = points.strip().split(" ")
                    point_array.append(np.array((float(point[0]), float(point[1])), dtype=np.float32))
                self.symbols.append(np.array(point_array, dtype=np.float32))


