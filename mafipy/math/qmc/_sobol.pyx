cimport cython
cimport libc.math as math
from cpython.mem cimport PyMem_Free
from cpython.mem cimport PyMem_Malloc
include '_sobol_constant.pix'


cdef extern from "joe_kuo_d6_21201_initial_number.h":
    (int*)[SOBOL_MAX_DIM] joe_kuo_d6_initial_number


cdef extern from "joe_kuo_d6_21201_irreducible.h":
    int* joe_kuo_d6_irreducibles


cdef int _calculate_degree(int polynomial):
    return int(math.log2(polynomial))


cdef int _get_length(int* initial_numbers):
    for i in range(MAX_BIT):
        if initial_numbers[i] == 0:
            return i
    return MAX_BIT


cdef int* allocate_int(size_t size, str name):
    mem = <int*>PyMem_Malloc(size * sizeof(int))
    if not mem:
        msg = "cannot allocate memory for {0}".format(name)
        raise MemoryError(msg)
    # zero padding
    for i in range(size):
        mem[i] = 0
    return mem


cdef size_t* allocate_size_t(size_t size, str name):
    mem = <size_t*>PyMem_Malloc(size * sizeof(size_t))
    if not mem:
        msg = "cannot allocate memory for {0}".format(name)
        raise MemoryError(msg)
    # zero padding
    for i in range(size):
        mem[i] = 0
    return mem


cdef class Sobol():

    cdef int* primitive_polynomials
    cdef int** direction_numbers
    cdef int* numbers
    cdef list points
    cdef bint is_first
    cdef int counter
    cdef size_t dim

    def __cinit__(self, size_t dim):
        self.dim = dim

        # index for dimension
        cdef int d
        # index for coeff
        cdef int k
        # degrees
        cdef size_t* degrees = allocate_size_t(dim, 'degrees')
        self.is_first = True
        self.counter = 0

        # initialize numbers
        self.numbers = allocate_int(dim , 'numbers')

        # initialize primitive_polynomial up to dimension
        self.primitive_polynomials = allocate_int(dim, 'primitive_polynomial')
        for d in range(dim):
            self.primitive_polynomials[d] = joe_kuo_d6_irreducibles[d]
            degrees[d] = _calculate_degree(self.primitive_polynomials[d])

        # allocate direction_numbers
        self.direction_numbers = <int**>PyMem_Malloc(dim * sizeof(int*))
        if not self.direction_numbers:
            msg = "cannot allocate memory for direction_numbers"
            raise MemoryError(msg)
        cdef size_t* initial_number_sizes = allocate_size_t(dim, 'initial_number_sizes')
        # load direction_numbers
        for d in range(dim):
            initial_number_sizes[d] = _get_length(joe_kuo_d6_initial_number[d])
            self.direction_numbers[d] = allocate_int(MAX_BIT, 'direction_numbers[{0}]'.format(d))
            for k in range(initial_number_sizes[d]):
                self.direction_numbers[d][k] = joe_kuo_d6_initial_number[d][k]
                self.direction_numbers[d][k] <<= (MAX_BIT - (k + 1))
        PyMem_Free(initial_number_sizes)

        cdef int i
        cdef int degree
        cdef size_t summand
        cdef size_t coeff
        # calculate direction numbers
        for d in range(dim):
            degree = degrees[d]
            for k in range(degree, MAX_BIT):
                summand = self.direction_numbers[d][k - degree] >> degree
                for i in range(0, degree):
                    coeff = (self.primitive_polynomials[d] >> (degree - i) & 1)
                    if coeff == 1:
                        summand ^= self.direction_numbers[d][k - i]
                self.direction_numbers[d][k] = summand

        # initialize points
        self.points = [0.0 for d in range(dim)]

        # free
        PyMem_Free(degrees)

    def __dealloc__(self):
        # primitive_polynomial
        PyMem_Free(self.primitive_polynomials)

        # direction_numbers
        for d in range(self.dim):
            PyMem_Free(self.direction_numbers[d])
        PyMem_Free(self.direction_numbers)

        # numbers
        PyMem_Free(self.numbers)

    def next(self):
        if self.is_first:
            self.is_first = False
            return [x for x in self.points]

        # find right most zero bit
        cdef size_t l = 0
        while ((self.counter >> l) & 1) == 1:
            l += 1

        for d in range(self.dim):
            self.numbers[d] ^= self.direction_numbers[d][l]
            self.points[d] = self.numbers[d] * NORMALIZE_FACTOR

        self.counter += 1
        return [x for x in self.points]


def make_sobol(dimension):
    if dimension > SOBOL_MAX_DIM:
        raise ValueError('diemsion <= {0}'.format(SOBOL_MAX_DIM))
    return Sobol(dimension)
