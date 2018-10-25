from inkml import Graphics
from inkml import InkML
from utilities import parse_arg
from file_utils import list_files
import png
import os

package_name = "TC11_package"
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/')
output_dir = parse_arg('--output-dir', '/Users/balazs/university/handwritten_images')

dirs = ["CROHME2016_data/TEST2016_INKML_GT", "CROHME2014_data/TestEM2014GT"] + \
       ["CROHME2013_data/TrainINKML/" + folder for folder in ["MfrDB", "MathBrush", "Kaist", "HAMEX", "extension", "expressmatch"]] + \
       ["CROHME2012_data/trainData"]


filtered_strings = ["\\int", "\\sum", "\\prod", "\\geq", "\\leq", "\\neq", "\\ldots", "\\pm",
                    "\\existsp", "\\lim", "\\infty", "\\rightarrow", "\\leftarrow", "\\mbox"]

def latex_filter(latex):
    '''Returns true if latex is passable'''
    for filter in filtered_strings:
        if filter in latex:
            return None
    return latex.replace(" ", "").replace("$", "").replace("\\cos", "cos").replace("\\sin", "sin").replace("\\tan", "tan") \
            .replace("\\cot", "cot").replace("\\log", "log").replace("\\ln", "ln")


if __name__ == "__main__":
    graphics = Graphics()
    if not os.path.exists(os.path.join(output_dir, package_name)):
        os.makedirs(os.path.join(output_dir, package_name))
        os.makedirs(os.path.join(output_dir, package_name, "images"))
    data_file = open(os.path.join(output_dir, package_name, "data.txt"), "w")

    for dir in dirs:
        for fname in list_files(os.path.join(data_base_dir, package_name, dir)):
            if fname.endswith(".inkml"):
                try:
                    short_fname = fname[len(os.path.join(data_base_dir, package_name, dir)) + 1:]
                    file = open(fname, "r")
                    string = file.read()
                    inkml = InkML(string)
                    filtered = latex_filter(inkml.truth)
                    if filtered is None:
                        continue
                    image = graphics.create_image(inkml)

                    fname = os.path.join(dir, short_fname[:len(short_fname) - len(".inkml")] + ".png").replace("/", "_")
                    text = fname + "\t" + filtered
                    data_file.write(text + "\n")
                    data_file.flush()
                    png.from_array(image, 'RGB').save(os.path.join(output_dir, package_name, "images", fname))
                except Exception as e:
                    print("Got exception ", e)
    data_file.close()