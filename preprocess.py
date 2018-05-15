import utils
import pickle
import os.path


def create_vocabulary(formulas_file, output_file):
    '''
    A method to create the vocabulary of a list of formulas. Have in mind that start token, 
    end token and unknown token are not included!

    Formuals file format: Each line represents one formula. Each token is seperated by a whitespace.
    Output file format: Each line represents one token.

    Arguments:
        formulas_file - a string representing the location of the formulas file
        output_file - a string representing the location of the output file
    '''
    formulas = utils._read_lines(formulas_file)
    vocabulary = set()
    for formula in formulas:
        for token in formula.split():
            vocabulary.add(token)
    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, "unknown")
    vocabulary.append("st")
    vocabulary.append("et")
    utils._write_list(output_file, vocabulary)
    return len(vocabulary)

def create_pkl_files(vocabulary_file, formulas_file, train_file, validate_file, test_file, output_dir, images_dir):
    '''
    A method to create special .pkl files. One for training, one for validating and one for testing. The format 
    of one such a .pkl file is: [('name _of_img.png', [451, 23, 123, 48, 9, 345, ...]), ...]. So a list of tuples. 
    One tuple includes a string, which represents the name of the img, and a list of integers, where one int represents exactly one token.

    Vocabulary file format: Each line represents one token.
    Formulas file format: Each line represents one formula. The tokens don't need to be seperated by a whitespace - but can.
    Train/Validate/Test file format: Line of [nr] [name] [rest]. The nr^th line in the formulas file belongs to the image name+'png'.

    Arguments:
        vocabulary_file - a string representing the location of the vocabulary file
        formulas_file - a string representing the location of the formulas file
        train_file - a string representing the location of the train file
        validate_file - a string representing the location of the validate file
        train_file - a string representing the location of the train file
        ouput_dir - a string representing the location where the .pkl files are saved
        images_dir - a string representing the location where all images are saved to find out, which images are not there
    '''
    tokens = utils._read_lines(vocabulary_file)
    vocabulary = {}
    for i, val in enumerate(tokens):
        vocabulary[val] = i

    formulas = utils._read_lines(formulas_file)
    train_size = _create_pkl_file(vocabulary, formulas, train_file, output_dir + '/train.pkl', images_dir)
    validation_size = _create_pkl_file(vocabulary, formulas, validate_file, output_dir + '/validate.pkl', images_dir)
    test_size = _create_pkl_file(vocabulary, formulas, test_file, output_dir + '/test.pkl', images_dir)
    return train_size, validation_size, test_size


def _create_pkl_file(vocabulary, formulas, input_file, output_file, images_dir):
    input = utils._read_lines(input_file)
    arr = []
    for str in input:
        strarr = str.split()
        tokens = formulas[int(strarr[0])].split()
        if len(tokens) == 0:
            continue
        img_name = strarr[1] + ".png"
        if not os.path.isfile(images_dir+img_name):
            continue
        sequence = [vocabulary['st']]
        for token in tokens:
            sequence.append(vocabulary[token])
        sequence.append(vocabulary['et'])
        arr.append((img_name, sequence))
    arr = sorted(arr, key=lambda elem: len(elem[1]))
    utils._write_pkl(arr, output_file)
    return len(arr)

def create_norm_formulas_file(formulas_file, output_file):
    formulas = utils._read_lines(formulas_file)
    new_formulas = []
    i = len(formulas)
    j = 0
    for formula in formulas:
        tokens = tokenize(formula)
        new_formula = normalize(tokens)
        new_formulas.append(new_formula)
        j += 1
        print(j, '/', i)
    utils._write_list(output_file, new_formulas)

def tokenize(str):
    tokens = [] # token list
    token = "" # current token
    bs = False # is backslash active
    prev = "a" # previous token
    for c in str:
        if c.isspace(): # check if c is space, tab, newline, ...
            if bs and prev != "\\"  or prev == "\\" and c == " ":
                tokens.append(token)
                token = ""
            bs = False
            continue

        if bs and prev != "\\" and not c.isalpha():
            tokens.append(token)
            token = ""
            bs = False

        token += c

        if bs:
            if not c.isalpha():
                tokens.append(token)
                token = ""
                bs = False
        else:
            if c == "\\":
                bs = True
            else:
                tokens.append(token)
                token = ""
        prev = c
    if token != "":
        tokens.append(token)
    return tokens

def normalize(tokens):
    new_tokens = _remove_formulas(tokens)
    new_tokens = _remove_tokens(new_tokens)
    new_tokens = _reomve_tokens_with_followings(new_tokens)
    new_tokens = _replace_tokens(new_tokens)
    new_tokens = _special_treatments(new_tokens)
    new_tokens = _add_necessary_braces(new_tokens)
    new_tokens = _remove_unnecessary_braces(new_tokens)
    new_tokens = _remove_empty_lines(new_tokens)
    str = ""
    for token in new_tokens:
        str += token
        str += " "
    return str

_illegal_tokens = ["\\vss", "\\hss", "\\vline", "\\line", "\\cline", "\\linethickness", "\\thicklines", "\\vector", "\\circle", "\\oval", "\\special", "\\qbezier", "\\put", "\\multiput", 
                   "\\footnote", "\\footnotemark", "\\framebox", "\\fbox", "\\fboxsep", "\\newcommand", "\\m", "\\llap", "\\rlap", "\\crcr", "\\buildrel"]
def _remove_formulas(tokens):
    for token in tokens:
        if token in _illegal_tokens:
            return []
    return tokens

_tokens_to_remove = ["", "\\big", "\\bigl", "\\bigm", "\\bigr", "\\bigg", "\\biggl", "\\biggm", "\\biggr", "\\Big", "\\Bigl", "\\Bigm", "\\Bigr", "\\Bigg", "\\Biggl", "\\Biggm", "\\Biggr", 
                     "\\left", "\\middle", "\\right", "\\mathopen", "\\mathclose", "\\mathstrut", "\\boldmath", "\\unboldmath", "\\do", "\\quad", "\\qquad", "\\hfill", "\\hfil", "\\hline", 
                     "\\def", "\\nolinebreak", "\\nonumber", "\\null", "\\relax", "\\smash", "\\uppercase", "\\expandafter", "\\strut", "\\,", "\\;", "\\:", "\\!", "\\/", "\\-", "\\*", "\\", 
                     "\\(", "\\)", "\\[", "\\]", "~", "\\huge", "\\Huge", "\\tiny", "\\scriptsize", "\\footnotesize", "\\small", "\\normalsize", "\\large", "\\Large", "\\LARGE", "\\rmfamily", 
                     "\\textrm", "\\rm", "\\sffamily", "\\textsf", "\\sf", "\\ttfamily", "\\texttt", "\\tt", "\\upshape", "\\textup", "\\up", "\\itshape", "\\textit", "\\it", "\\slshape", 
                     "\\textsl", "\\sl", "\\scshape", "\\textsc", "\\sc", "\\em", "\\emph", "\\em", "\\mdseries", "\\textmd", "\\md", "\\bfseries", "\\textbf", "\\bf", "\\mathrm", "\\mathbf", 
                     "\\mathbin", "\\mathit", "\\mathnormal", "\\mathop", "\\mathord", "\\mathrel", "\\mathsf", "\\mathtt", "\\ensuremath", "\\mit", "\\lefteqn", "\\vcenter", "\\ooalign", 
                     "\\scriptstyle", "\\scriptscriptstyle", "\\displaystyle", "\\textstyle", "\\textnormal", "\\cal", "\\mathcal"]
def _remove_tokens(tokens):
    return [token for token in tokens if token not in _tokens_to_remove and not (("space" in token or "skip" in token) and not ("\\v" in token or "\\h" in token))]

_tokens_to_remove_with_followings_1 = ["\\mathversion", "\\phantom", "\\hphantom", "\\vphantom", "\\noalign", "\\arraystretch", "\\raisebox", "\\label", "\\skew"]
_tokens_to_remove_with_followings_2 = ["\\renewcommand", "\\setlength", "\\setcounter"]
_tokens_to_remove_with_following_text = ["\\hrule", "\\vrule", "\\lower", "\\raise", "\\arraycolsep", "\\tabcolsep", "\\unitlength", "\\vskip", "\\hskip", "\\kern", "\\mkern", "\\verb", "\\everymath"]
def _reomve_tokens_with_followings(tokens):
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        counter = 0
        if token in _tokens_to_remove_with_followings_1 or token in _tokens_to_remove_with_followings_2:
            if token in _tokens_to_remove_with_followings_2:
                counter = 2
            else: 
                counter = 1
            for _ in range(counter):
                j = i + 1
                if tokens[j] == "{":
                    i = _find_closing_brace(tokens, j)
                else:
                    i = j
        elif token in _tokens_to_remove_with_following_text:
            i += 1
            while i < len(tokens):
                token = tokens[i]
                if not (token.isalnum() or token in ["+", "-", "=", "."]):
                    i -= 1
                    break;
                i += 1
        else:
            new_tokens.append(token)
        i += 1
    return new_tokens

_tokens_to_change = {"long":"", "Long":"", "big": "", "wide":"", "small":"", "protect":""}
_tokens_to_replace = {"\\arrowvert":"|", "\\vert":"|", "\\mid":"|", "\\parallel":"||", "\\Vert":"||", "\\wedge":"\\land", "\\vee":"\\lor", "\\sum":"\\Sigma", "\\prod":"\\Pi", "\\amalg":"\\coprod", 
                      "\\sp":"^", "\\sb":"_", "\\setminus":"\\backslash",  "\\slash":"/",  "\\lbrack":"[",  "\\rbrack":"]",  "\\lbrace":"\\{",  "\\rbrace":"\\}",  "\\ointop":"\\oint", 
                      "\\rightarrowfill":"\\rightarrow",  "'":"\\prime",  "\\o":"\\phi",  "`":"\\lq", "\\le":"\\leq",   "\\ge":"\\geq",  "\\l":"l",  "\\to":"\\rightarrow",  "\\perp":"\\bot", 
                      "\\bmod":"\\mod",  "\\diamond":"\\diamondsuit",  "\\dagger":"\\dag",  "\\ddagger":"\\ddag",  "\\colon":":",  "\\cdotp":"\\cdot",  "\\cdots":"\\hdots",  "\\dots":"\\hdots", 
                      "\\ldots":"\\hdots",  "\\dotfill":"\\hdots",  "\\=":"\\overline",  "\\bar":"\\overline",  "\\b":"\\underline",  "\\.":"\\dot",  "\\~":"\\tilde",  "\\^":"\\hat", 
                      "\\`":"\\grave",  "\\'":"\\acute",  "\\\"":"\\ddot",  "\\vec":"\\overrightarrow"}
_tokens_to_replace_and_add = {"\\sqrt":["\\root", "{", "2", "}", "\\of"], "ne":["\\not", "="], "neq":["\\not", "="], "doteq":["\\dot", "="], "\\brack":["\\atopwithdelims", "[", "]"], "\\brace":["\\atopwithdelims", "\\{", "\\}"]}
def _replace_tokens(tokens):
    change = _tokens_to_change.keys()
    replace = _tokens_to_replace.keys()
    replace_add = _tokens_to_replace_and_add.keys()
    new_tokens = []
    for token in tokens:
        for key in change:
            if key in token:
                token = token.replace(key, _tokens_to_change[key])
        if token in replace:
            token = _tokens_to_replace[token]
        elif token in replace_add:
            tokens_to_add = _tokens_to_replace_and_add[token]
            for i in range(len(tokens_to_add)-1):
                new_tokens.append(tokens_to_add[i])
            token = tokens_to_add[-1]
        new_tokens.append(token)
    return new_tokens

def _special_treatments(tokens):
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "\\makebox":
            while i < len(tokens) - 1:
                if tokens[i+1] == "{":
                    break;
                i += 1
        elif token in ["\\parbox", "\\vspace", "\\hspace"]:
            while i < len(tokens):
                i += 1
                if tokens[i] == "}":
                    break;
        elif token == "\\operatorname":
            i += 1
            while tokens[i] != "{":
                i += 1
            i += 1
            token = "\\"
            while tokens[i] != "}":
                token += tokens[i]
                i += 1
            new_tokens.append(token)
        elif token in ["\\begin", "\\end"]:
            while tokens[i] != "}":
                i += 1
            token += "{array}"
            new_tokens.append(token)
        else:
            new_tokens.append(token)
        i += 1
    new_tokens = _special_treatments_delims(new_tokens)
    return new_tokens

def _special_treatments_delims(tokens):
    new_tokens = []
    if "\\atopwithdelims" in tokens:
        i = tokens.index("\\atopwithdelims")
        j = _find_opening_brace(tokens, i)
        k = _find_closing_brace(tokens, i)
        new_tokens += tokens[:j]
        new_tokens.append(tokens[i+1])
        new_tokens.append(tokens[j])
        new_tokens += _special_treatments_delims(tokens[j+1:i])
        new_tokens.append("\\atop")
        new_tokens += _special_treatments_delims(tokens[i+3:k])
        new_tokens.append(tokens[k])
        new_tokens.append(tokens[i+2])
        new_tokens += _special_treatments_delims(tokens[k+1:])
    elif "\\overwithdelims" in tokens:
        i = tokens.index("\\overwithdelims")
        j = _find_opening_brace(tokens, i)
        k = _find_closing_brace(tokens, i)
        new_tokens += tokens[:j]
        new_tokens.append(tokens[i+1])
        new_tokens += ["\\frac", "{"]
        new_tokens += _special_treatments_delims(tokens[j+1:i])
        new_tokens += ["}", "{"]
        new_tokens += _special_treatments_delims(tokens[i+3:k])
        new_tokens.append("}")
        new_tokens.append(tokens[i+2])
        new_tokens += _special_treatments_delims(tokens[k+1:])
    elif "\\binom" in tokens:
        i = tokens.index("\\binom")
        j = _find_closing_brace(tokens, i+1)
        k = _find_closing_brace(tokens, j+1)
        new_tokens += tokens[:i]
        new_tokens += ["(", "{"]
        new_tokens += tokens[i+1:j+1]
        new_tokens.append("\\atop")
        new_tokens += tokens[j+1:k+1]
        new_tokens += ["}", ")"]
        new_tokens += _special_treatments_delims(tokens[k+1:])
    elif "\\pmod" in tokens:
        i = tokens.index("\\pmod")
        if tokens[i+1] == "{":
            j = _find_closing_brace(tokens, i+1)
        else:
            j = i+1
        new_tokens += tokens[:i]
        new_tokens += ["(", "\\mod"]
        new_tokens += tokens[i+1:j+1]
        new_tokens.append(")")
        new_tokens += _special_treatments_delims(tokens[j+1:])
    else:
        new_tokens = tokens
    return new_tokens

_tokens_as_functions = {"\\underline":1, "\\underbrace":1, "\\tilde":1, "\\textcircled":1, "\\stackrel":2, "\\root":1, "\\overrightarrow":1, "\\overline":1, "\\overleftarrow":1, 
                        "\\overbrace":1, "\\of":1, "\\multicolumn": 3, "\\mathring":1, "\\hat":1, "\\grave":1, "\\frac":2, "\\dot":1, "\\ddot":1, "\\d":1, "\\check":1, "\\c":1, 
                        "\\breve":1, "\\acute":1, "_":1, "^":1}
def _add_necessary_braces(tokens):
    func_tokens = _tokens_as_functions.keys()
    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        new_tokens.append(token)
        if token in func_tokens:
            num_args = _tokens_as_functions[token]
            for _ in range(num_args):
                if(len(tokens) <= i+1):
                    print(tokens)
                if tokens[i+1] == "{":
                    j = _find_closing_brace(tokens, i+1)
                    new_tokens.append("{")
                    new_tokens += _add_necessary_braces(tokens[i+2:j])
                    new_tokens.append("}")
                else:
                    j = i+1
                    new_tokens.append("{")
                    new_tokens.append(tokens[j])
                    new_tokens.append("}")
                i = j
        i += 1
    return new_tokens

def _remove_unnecessary_braces(tokens):
    if "{" not in tokens:
        return tokens
    func_tokens = _tokens_as_functions.keys()
    new_tokens = []
    m = 0
    n = len(tokens)
    while m < n:
        sub_tokens = tokens[m:n]
        if "{" in sub_tokens:
            i = sub_tokens.index("{") + m
            j = _find_closing_brace(tokens, i)
            k = i-1
            should_delete = True
            for arg in range(3):
                token = tokens[k]
                if k < 0:
                    break
                elif token not in func_tokens:
                    if token == "}":
                        k = _find_opening_brace(tokens, k)-1
                        continue
                    else:
                        break
                elif _tokens_as_functions[token] <= arg:
                    break
                else:
                    should_delete = False
                    break
            new_tokens += tokens[m:i]
            atop = tokens[i:j]
            if should_delete and "\\atop" in atop:
                a = atop.index("\\atop")
                a = _find_opening_brace(atop, a)
                should_delete = a != 0
            if should_delete and i != 0:
                should_delete = tokens[i-1]!="\\begin{array}"
            if not should_delete:
                new_tokens.append("{")
            new_tokens += _remove_unnecessary_braces(tokens[i+1:j])
            if not should_delete:
                new_tokens.append("}")
            m = j+1
        else:
            new_tokens += sub_tokens
            break;
    return new_tokens

def _remove_empty_lines(tokens):
    if "\\begin{array}" not in tokens:
        return tokens
    i = 0
    j = len(tokens)
    new_tokens = []
    while i < j:
        sub_tokens = tokens[i:j]
        if "\\\\" in sub_tokens:
            k = sub_tokens.index("\\\\") + i
            should_delete = True
            for m in range(i, k):
                token = tokens[m]
                if token == "\\\\" or token == "\\end{array}":
                    break
                elif token != "&":
                    should_delete = False
                    break
            if not should_delete and i != k:
                new_tokens += tokens[i:k+1]
            i = k
        else:
            if len(sub_tokens) != 0 and len(new_tokens) != 0 and sub_tokens[0] == "\\end{array}" and new_tokens[-1] == "\\\\":
                new_tokens.pop()
            new_tokens += sub_tokens
            i = j
        i += 1
    return new_tokens

def _find_closing_brace(tokens, i_start):
    counter = 1
    for i in range(i_start+1, len(tokens)):
        token = tokens[i]
        if token == "{":
            counter += 1
        elif token == "}":
            counter -= 1
            if counter == 0:
                return i

def _find_opening_brace(tokens, i_start):
    counter = 1
    for i in range(i_start-1, -1, -1):
        token = tokens[i]
        if token == "}":
            counter += 1
        elif token == "{":
            counter -= 1
            if counter == 0:
                return i