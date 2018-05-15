import pickle

def _write_pkl(object, file):
    with open(file, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)

def _read_pkl(file):
    with open(file, 'rb') as input:
        content = pickle.load(input)
    return content

def _read_lines(file):
    with open(file, 'r') as f:
        content = f.read().splitlines()
    return content;

def _write_list(file, list):
    with open(file, 'w') as output:
        for token in list:
            output.write("%s\n" % token)

def _read_content(file):
    with open(file, 'r') as f:
        content = f.read()
    return content;

def _write_string(file, string):
    with open(file, 'w') as output:
        output.write(string)