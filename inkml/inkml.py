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
        self.extract_data()

    def extract_data(self):
        for child in self.root:
            if child.tag == '{http://www.w3.org/2003/InkML}annotation' and child.attrib['type'] == 'truth':
                self.truth = child.text
            if child.tag == '{http://www.w3.org/2003/InkML}trace':
                point_array = []
                for points in child.text.split(","):
                    point = points.strip().split(" ")
                    point_array.append(np.array((int(point[0]), int(point[1])), dtype=np.int32))
                self.symbols.append(np.array(point_array, dtype=np.int32))


