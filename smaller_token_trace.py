import pickle
import os


fname = "testing"

data = pickle.load(open(os.path.join('/Users/balazs/token_trace/{}_data.pkl'.format(fname)), 'rb'))
print(len(data))

permitted_tokens = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '\\times', '=', '^', '{', '}', 'x'}

result = []
mx = 0
fm = ""
for formula, tracegroups in data:
    permitted = set(formula) - permitted_tokens == set()
    if permitted:
        for token, _ in tracegroups:
            permitted = token in permitted_tokens
            if not permitted:
                break

    if permitted:
        result.append((formula, tracegroups))
        if mx < len(formula):
            mx = len(formula)
            fm = formula

# pickle.dump(result, open('/Users/balazs/token_trace_small/{}_data.pkl'.format(fname), 'wb'))
vocabulary = sorted(list(permitted_tokens))
vocabulary = {token: index + 1 for index, token in enumerate(vocabulary)}
pickle.dump(vocabulary, open('/Users/balazs/token_trace_small/vocabulary.pkl', 'wb'))
