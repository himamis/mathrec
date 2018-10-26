from utilities import parse_arg, progress_bar
import pathlib, os, inkml, sys, traceback
from database_create.methods import *

package_name = "TC11_package"
database_directory = parse_arg("--data-base-dir", "/Users/balazs/database")
package_directory = os.path.join(database_directory, package_name)
if not os.path.exists(package_directory):
    print("Path does not exist: " + package_directory)
    exit()

inkml_directories = ["CROHME2011_data/CROHME_testGT", "CROHME2011_data/CROHME_training",
                     "CROHME2012_data/testDataGT", "CROHME2012_data/trainData"] \
                  + ["CROHME2013_data/TrainINKML/" + name for name in ["HAMEX", "KAIST", "MathBrush", "MfrDB", "expressmatch", "extension"]] \
                  + ["CROHME2013_data/TestINKMLGT", "CROHME2014_data/TestEM2014GT", "CROHME2016_data/Test2016_INKML_GT"]

clear_all_tables()
for inkml_directory in inkml_directories:
    files_directory = os.path.join(package_directory, inkml_directory)
    if not os.path.exists(files_directory):
        print("Path does not exist: " + files_directory)
        exit()
    print("Processing directory: " + files_directory)
    inkml_files = sorted(pathlib.Path(files_directory).glob("*.inkml"))
    no_files = len(inkml_files)
    dbid = maybe_add_database(inkml_directory)
    print("Number of files: " + str(no_files))
    for index in range(no_files):
        progress_bar("Processing files", index, no_files - 1)
        inkml_file_path = inkml_files[index]
        with open(str(inkml_file_path), mode="rb") as inkml_file:
            try:
                contents = inkml_file.read().decode(errors='replace')
                if contents != "":
                    ink = inkml.InkML(contents)
                    add_inkml(ink, dbid)
                else:
                    print("Empty contents in file: " + str(inkml_file_path))
            except:
                rollback()
                close()
                e = sys.exc_info()[0]
                print(e)
                print("Couldnt parse file: " + str(inkml_file_path))
                print(traceback.format_exc())
                exit()

commit()