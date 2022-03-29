


def add2A(A, data, ohe=False):
    A = A.clone()
    print('adding data to A')
    for window in tqdm.auto.tqdm(data):
        if ohe:
            window = window.nonzero(as_tuple=True)[0]
        inds = tr.cartesian_prod(window, window).T
        if inds.shape[1] > 0:
            r = coo(inds, tr.ones(inds.shape[1]),
                    size=size).coalesce().bool().int()
            if A.is_sparse():
                A += r
            else: A += r.to_dense()
    return A