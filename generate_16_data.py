from utilities import progress_bar
import pathlib, os, inkml, sys, traceback

first_part = True

formulas = []

if first_part:
    package_directory = '/Users/balazs/database/TC11_package'
    inkml_directory = 'CROHME2016_data/TEST2016_INKML_GT'
    files_directory = os.path.join(package_directory, inkml_directory)
    if not os.path.exists(files_directory):
        print("Path does not exist: " + files_directory)
        exit()
    print("Processing directory: " + files_directory)
    inkml_files = sorted(pathlib.Path(files_directory).glob("*.inkml"))
    no_files = len(inkml_files)
    #dbid = maybe_add_database(inkml_directory)
    print("Number of files: " + str(no_files))
    for index in range(no_files):
        progress_bar("Processing files", index, no_files - 1)
        inkml_file_path = inkml_files[index]
        with open(str(inkml_file_path), mode="rb") as inkml_file:
            try:
                contents = inkml_file.read().decode(errors='replace')
                if contents != "":
                    ink = inkml.InkML(contents)

                    fname = os.path.splitext(inkml_file_path.name)[0]

                    formulas.append((fname, ink.truth))
                    # DO STUFF
                else:
                    print("Empty contents in file: " + str(inkml_file_path))
            except:
                e = sys.exc_info()[0]
                print(e)
                print("Couldnt parse file: " + str(inkml_file_path))
                print(traceback.format_exc())
                exit()

data_dir = '/Users/balazs/university/WAP/data'
dict = open(os.path.join(data_dir, "dictionary.txt"))
dict = dict.readlines()
words = set()
for line in dict:
    l = line.strip().split()
    words |= {l[0]}

from parsy import string, alt
vocab = reversed(sorted(words | {" "}))
parser = alt(*map(string, vocab))
parser = parser.many()

new_forms = []
for formula in formulas:
    f = ""
    f = formula[1]
    f = f.replace("\\left(", "(").replace("\\right)", ")")
    f = f.replace("\\left (", "(").replace("\\right )", ")")
    f = f.replace("\\left\\{", "\\{").replace("\\right\\}", "\\}")
    if f[0] == "$":
        f = f[1:]
    if f[-1] == "$":
        f = f[:-1]
    if "\\mbox{" in f:
        start = f.index("\\mbox{") + len("\\mbox{")
        index = start
        close = 1
        while close != 0:
            if f[index] == "{":
                close += 1
            elif f[index] == "}":
                close -= 1
            index += 1
        if start - index - 1 > 1:
            print("Problem")
        else:
            f = f[0:start - len("\\mbox{")] + f[start:index-1] + f[index:]
    while "\\mathrm{" in f:
        start = f.index("\\mathrm{") + len("\\mathrm{")
        index = start
        close = 1
        while close != 0:
            if f[index] == "{":
                close += 1
            elif f[index] == "}":
                close -= 1
            index += 1
        f = f[0:start - len("\\mathrm{")] + f[start:index-1] + f[index:]
    try:
        parsed = parser.parse(f)
        new_forms.append((formula[0], f))
    except Exception as e:
        print(formula[0])
        print(f)
        print(str(e)[-10:])

with open("/Users/balazs/university/WAP/data/16_caption.txt", "w") as f:
    for formula in new_forms:
        parsed = parser.parse(formula[1])
        parsed = filter(lambda a: a != " ", parsed)
        str = " ".join(parsed)
        f.write(formula[0] + "\t" + str + "\n")