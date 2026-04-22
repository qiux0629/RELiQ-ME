def one_hot_list(i, max_indices):
    values = [0] * max_indices
    if i >= 0:
        values[i] = 1
    return values


def multi_hot_list(indices, max_indices):
    if len(indices) == max_indices:
        return [1] * max_indices

    values = [0] * max_indices
    for i in indices:
        values[i] = 1
    return values
