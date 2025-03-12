from gpaw.gpu import cupy as cp
import numpy as np


class CuPyMPI:
    """Quick'n'dirty wrapper to make things work without a GPU-aware MPI."""
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size

    def __repr__(self):
        return f'CuPyMPI({self.comm})'

    def sum(self, array, root=-1):
        assert array.flags.c_contiguous
        if isinstance(array, (float, int)):
            1 / 0
            return self.comm.sum(array, root)
        if isinstance(array, np.ndarray):
            self.comm.sum(array, root)
            return
        a = array.get()
        self.comm.sum(a, root)
        array.set(a)

    def translate_ranks(self, other, ranks):
        return self.comm.translate_ranks(other.comm, ranks)

    def sum_scalar(self, a, root=-1):
        return self.comm.sum_scalar(a, root)

    def min_scalar(self, a, root=-1):
        return self.comm.min_scalar(a, root)

    def max_scalar(self, a, root=-1):
        return self.comm.max_scalar(a, root)

    def max(self, array):
        assert array.flags.c_contiguous
        self.comm.max(array)

    def all_gather(self, a, b):
        if a is not None:
            assert a.flags.c_contiguous
        if b is not None:
            assert b.flags.c_contiguous

        self.comm.all_gather(a, b)

    def gather(self, a, rank, b):
        if a is not None:
            assert a.flags.c_contiguous
        if b is not None:
            assert b.flags.c_contiguous
        if isinstance(a, np.ndarray):
            self.comm.gather(a, rank, b)
        else:
            if rank == self.rank:
                c = np.empty(b.shape, b.dtype)
            else:
                c = None
            self.comm.gather(a.get(), rank, c)
            if rank == self.rank:
                b[:] = cp.asarray(c)

    def scatter(self, fro, to, root=0):
        if fro is not None:
            assert fro.flags.c_contiguous
        assert to.flags.c_contiguous
        if isinstance(to, np.ndarray):
            self.comm.scatter(fro, to, root)
            return
        b = np.empty(to.shape, to.dtype)
        if self.rank == root:
            a = fro.get()
        else:
            a = None
        self.comm.scatter(a, b, root)
        to[:] = cp.asarray(b)

    def broadcast(self, a, root):
        assert a.flags.c_contiguous
        if isinstance(a, np.ndarray):
            self.comm.broadcast(a, root)
            return
        b = a.get()
        self.comm.broadcast(b, root)
        a[...] = cp.asarray(b)

    def receive(self, a, rank, tag=0, block=True):
        assert a.flags.c_contiguous
        if isinstance(a, np.ndarray):
            return self.comm.receive(a, rank, tag, block)
        b = np.empty(a.shape, a.dtype)
        req = self.comm.receive(b, rank, tag, block)
        if block:
            a[:] = cp.asarray(b)
            return
        return CuPyRequest(req, b, a)

    def ssend(self, a, rank, tag):
        assert a.flags.c_contiguous
        if isinstance(a, np.ndarray):
            self.comm.ssend(a, rank, tag)
        else:
            self.comm.ssend(a.get(), rank, tag)

    def send(self, a, rank, tag=0, block=True):
        assert a.flags.c_contiguous
        if isinstance(a, np.ndarray):
            return self.comm.send(a, rank, tag, block)
        b = a.get()
        request = self.comm.send(b, rank, tag, block)
        if not block:
            return CuPyRequest(request, b)

    def alltoallv(self,
                  fro, ssizes, soffsets,
                  to, rsizes, roffsets):
        assert fro.flags.c_contiguous
        assert to.flags.c_contiguous
        a = np.empty(to.shape, to.dtype)
        self.comm.alltoallv(fro.get(), ssizes, soffsets,
                            a, rsizes, roffsets)
        to[:] = cp.asarray(a)

    def wait(self, request):
        if not isinstance(request, CuPyRequest):
            return self.comm.wait(request)
        self.comm.wait(request.request)
        if request.target is not None:
            request.target[:] = cp.asarray(request.buffer)

    def waitall(self, requests):
        self.comm.waitall([request.request for request in requests])
        for request in requests:
            if request.target is not None:
                request.target[:] = cp.asarray(request.buffer)

    def get_c_object(self):
        return self.comm.get_c_object()


class CuPyRequest:
    def __init__(self, request, buffer, target=None):
        self.request = request
        self.buffer = buffer
        self.target = target
