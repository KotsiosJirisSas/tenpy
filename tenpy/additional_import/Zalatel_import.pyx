################## multilayer QH ##############

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, uint8_t
from cpython cimport bool


cimport cython


@cython.wraparound(False)
@cython.boundscheck(False)
def packVmk(list Vmk_list, cnp.ndarray[cnp.float32_t, ndim=6] vmk, cnp.ndarray[cnp.uint8_t, ndim=6] mask, int maxM, int maxK, spec1, spec2, spec3, spec4, float tol):
    cdef int a, b, c, d, m, k, sa, sb, sc, sd
    cdef float v
    cdef int n_a = vmk.shape[0]
    cdef int n_b = vmk.shape[1]
    cdef int n_c = vmk.shape[2]
    cdef int n_d = vmk.shape[3]
    for a in range(n_a):
        sa = spec1[a]
        for b in range(n_b):
            sb = spec2[b]
            for c in range(n_c):
                sc = spec3[c]
                for d in range(n_d):
                    sd = spec4[d]
                    for m in range(0, 2 * maxM + 1):
                        m_loc = m - maxM
                        for k in range(0, 2 * maxK + 1):
                            v = vmk[a, b, c, d, m, k]
                            if abs(v) > tol:
                                Vmk_list.append([m_loc, k - maxK, sa, sc, sd, sb, v, mask[a, b, c, d, m, k]])