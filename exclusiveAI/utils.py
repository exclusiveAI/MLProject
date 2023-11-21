__all__ = ['utils']

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