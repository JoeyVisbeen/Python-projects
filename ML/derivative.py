import sympy

J, w = sympy.symbols('J,w')

J = w**3
#print(J)
# derivative
dJ_dw = sympy.diff(J,w)
print(dJ_dw)

dJ_dw.subs([w,2])
print(dJ_dw)

def split_indices(X, index_feature):
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

# no decisiontree on unstructured data. decision trees are pretty fast.
# tree ensamble is more heavy than single tree
# NN works well on everything
# NN is slower
# Transfer learning