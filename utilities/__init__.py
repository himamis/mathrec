from trainer import params
import sys


def parse_arg(name, default=None,required=False):
    """
    Parses the command line arguments.

    :param name: the name of the argument
    :param default: the default value
    :param required: true if param is required.
    :return: the value of the argument or the default value
    """
    if name in sys.argv:
        i_arg = sys.argv.index(name) + 1
        if i_arg < len(sys.argv):
            print(name + '\t resolved to \t' + sys.argv[i_arg])
            return sys.argv[i_arg]
        else:
            if required:
                raise Exception("Arg " + name + " is missing")
            else:
                print(name + '\t using default \t' + str(default))
                return default
    else:
        if required:
            raise Exception("Arg " + name + " is missing")
        else:
            print(name + '\t using default \t' + str(default))
            return default


def progress_bar(name, value, endvalue, bar_length=50, width=20):
    percent = float(value) / endvalue

    arrow = '-' * int(round(percent * bar_length) - 1) + '>'

    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if value == endvalue:
        sys.stdout.write('\n\n')


def log(message):
    if params.verbose:
        print(message)
