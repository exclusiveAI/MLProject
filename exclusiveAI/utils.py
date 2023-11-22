__all__ = ['utils']

def split_batches(inputs, input_label, batch_size):

    """
    Split the inputs into batches of size batch_size.
    """
    for i in range(0, len(inputs), batch_size):
        # if last
        if i + batch_size > len(inputs):
            batch_size = len(inputs) - i
        yield inputs[i:i+batch_size], input_label[i:i+batch_size]
def myzip(*iterables):
    sentinel = object()
    iterators = [iter(it) for it in iterables if it]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)
