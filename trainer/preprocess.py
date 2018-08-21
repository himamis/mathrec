import sys
from trainer import utils


def preprocess(formulas, formulas_path, vocabulary_path, dataset_paths, imgs_path):
    if type(formulas) is type(int()):
        formulas = form.create(formulas)
        utils.write_list(formulas_path, formulas)
    elif type(formulas) is type(None):
        formulas = utils.read_lines(formuals_path)
    # else: type should be a list

    vocabulary = create_vocabulary(formulas)
    train, validate, test = create_datasets(vocabulary, formulas, imgs_path)
    utils.write_list(vocabulary, vocabulary_path)
    utils.write_pkl(dataset_paths[0], train)
    utils.write_pkl(dataset_paths[1], validate)
    utils.write_pkl(dataset_paths[2], test)
    return formulas

def create_vocabulary(formulas):
    '''
    A method to create the vocabulary of a list of formulas. Start token has index 0 and end token index 1.

    Arguments:
        formulas - list of strings of formulas
    '''
    vocabulary = set()
    for formula in formulas:
        for token in formula.split():
            vocabulary.add(token)
    vocabulary = sorted(vocabulary)
    vocabulary.insert(0, "<st>")
    vocabulary.insert(1, "<et>")
    return vocabulary


def create_datasets(vocabulary, formulas, images_dir):
    '''
    A method to create the data sets for training, validating and testing, with the relation 80 : 10 : 10. The format of one such a data set is: 
            [('index_of_formula.png', [0, 23, 123, 48, 9, 345, ..., 1], (width, height)), ...)]. 
    So a list of tuples. One tuple includes a string, which represents the name of the img, a list of integers, 
    where one int represents exactly one token and the size of the image. The size one needs later, because all images 
    need to have the same size. Also each data set is ordered by the length of the sequence (output[1]).

    Arguments:
        vocabulary - a list of strings of tokens
        formulas - a list of strings of formulas
        images_dir - a path to the location where all images are saved to
    '''
    vocab_dict = {token:i for i,token in enumerate(vocabulary)}

    formulas_tuples = []
    for i, formula in enumerate(formulas):
        img_name = str(i) + '.png'
        seq = [vocab_dict[token] for token in formula]
        seq.insert(0, vocab_dict['<st>'])
        seq.append(vocab_dict['<et>'])
        with Image.open(images_dir + img_name) as img:
            img_size = img.size
        formulas_tuples.append((img_name, seq, img_size))

    total_size = len(formulas)
    train, validate, test = formulas_tuple[:int(total_size / 10 * 8)], formulas_tuple[int(total_size / 10 * 8):int(total_size / 10 * 9)], formulas_tuple[int(total_size / 10 * 9):]
    train = sorted(train, key=lambda x: len(x[1]))
    validate = sorted(validate, key=lambda x: len(x[1]))
    test = sorted(test, key=lambda x: len(x[1]))
    return train, validate, test


if __name__ == '__main__':
    if '--data-path' in sys.argv:
        i_arg = sys.argv.index('--data-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --data-path argument!')
        data_base_dir = sys.argv[i_arg]
        if data_base_dir[-1] != '/':
            data_base_dir += '/'
    else:
        raise Exception('No --data-path argument!')
    if '--formulas-path' in sys.argv:
        i_arg = sys.argv.index('--formulas-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --formulas-path argument!')
        formulas_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --formulas-path argument!')
    if '--vocabulary-path' in sys.argv:
        i_arg = sys.argv.index('--vocabulary-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --vocabulary-path argument!')
        vocabulary_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --vocabulary-path argument!')
    if '--images-path' in sys.argv:
        i_arg = sys.argv.index('--images-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --images-path argument!')
        images_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --images-path argument!')
    if '--train-path' in sys.argv:
        i_arg = sys.argv.index('--train-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --train-path argument!')
        train_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --train-path argument!')
    if '--validate-path' in sys.argv:
        i_arg = sys.argv.index('--validate-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --validate-path argument!')
        validate_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --validate-path argument!')
    if '--test-path' in sys.argv:
        i_arg = sys.argv.index('--test-path') + 1
        if i_arg >= len(sys.argv):
            raise Exception('No --test-path argument!')
        test_path = data_base_dir + sys.argv[i_arg]
    else:
        raise Exception('No --test-path argument!')
    new_formulas = None
    if '--new-formulas' in sys.argv:
        i_arg = sys.argv.index('--new-formulas') + 1
        if i_arg < len(sys.argv):
            new_formulas = int(sys.argv[i_arg])

    preprocess(new_formulas, formulas_path, vocabulary_path, [train_path, validate_path, test_path], imgs_path)
