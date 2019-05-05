

class Symbol:

    def __init__(self, bounding_box, token):
        self.bounding_box = bounding_box
        self.token = token

def find_leftmost(symbols):
    return sorted(symbols, key=lambda s: s.bounding_box[0])[0]

def _sort_boxes(boxes):
    return sorted(boxes, key=lambda b: b[1][0])

def parse_bounding_boxes(boxes):
    symbols = set()
    for token, bounding_box in boxes:
        symbols.add(Symbol(bounding_box, token))

    free_symbols = symbols
    connected_symbols = set()

    leftmost_symbol = find_leftmost(symbols)
