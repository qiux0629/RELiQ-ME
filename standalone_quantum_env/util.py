def one_hot_list(i, max_indices):
    values = [0] * max_indices
    if i >= 0:
        values[i] = 1
    return values
