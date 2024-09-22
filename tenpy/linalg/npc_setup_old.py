from distutils.extension import Extension
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import platform
import os
import numpy as np

BLAS_COMPLEX_INNER = 1 #There seems to be a bug in Canopy info for our complex BLAS call; set to 0 in this case before compile.
PARALLEL_TDOT = 1 #Use OpenMP to parallelize over contraction of tensordot blocks
HAS_MKL = 0
MKL_DIRECT_CALL = 1 #Use MKL_DIRECT_CALL

system = platform.system()
extra_compile_args = ["-O3"]
extra_link_args = []
library_dirs = []
include_dirs = [np.get_include()]
libraries = []


CC = os.getenv('CC')

if PARALLEL_TDOT: #really this should be based on compiler type but pain in ass
    if system == 'Darwin':
        print ("OpenMP not standard on OS-X, turning off PARALLEL_TDOT")
        PARALLEL_TDOT=0
    elif system == 'Windows':
        extra_compile_args.extend(['/openmp', '/Ox', '/fp:fast','/favor:INTEL64','/Og'])
    elif CC == 'icc':
        extra_compile_args.extend(["-qopenmp"])
        libraries.extend([ "iomp5"])
    else:
        extra_compile_args.extend(["-fopenmp"]) #assume gcc ?
        #libraries.extend(["pthread", "gomp"])



if CC == 'icc': #Intel compilers - have to link against this library
    libraries.append('irc')
    extra_compile_args.append("-xHost")
else:
    if platform.system() != 'Darwin':
        extra_compile_args.append("-march=native")

#Check for MKL or EPD by looking for these flags in environment variable
MKL_DIR = os.getenv('MKL_DIR')
if MKL_DIR==None:
    MKL_DIR = os.getenv('MKLROOT')
    if MKL_DIR == None:
        MKL_DIR = os.getenv('MKL_HOME')

CONDA_DIR = os.getenv('CONDA_PREFIX')
ATLAS_DIR = os.getenv('ATLAS_DIR')
OPENBLAS_DIR = os.getenv('OPENBLAS_DIR')

print ("MKL, Conda, ATLAS, OpenBLAS:", MKL_DIR, CONDA_DIR, ATLAS_DIR, OPENBLAS_DIR)

if MKL_DIR is not None:
    print( "Using MKL")
    HAS_MKL=1
    with open('cblas.h', 'w') as file: #cblas is included in mkl.h, so make a little dummy file
        file.write('#include "mkl.h"')
    include_dirs.append(MKL_DIR + '/include')
    library_dirs.append(MKL_DIR +'/lib/intel64')
    if MKL_DIRECT_CALL:
        extra_compile_args.append("-DMKL_DIRECT_CALL")
    TKlibraries=['mkl_rt', 'iomp5']

elif CONDA_DIR is not None:
    print ("Using Anaconda")
    if  int(os.path.exists(os.path.join(CONDA_DIR, 'include', 'mkl.h'))):
        HAS_MKL = 1
        with open('cblas.h', 'w') as file: #cblas is included in mkl.h, so make a little dummy file
            file.write('#include "mkl.h"')
        if MKL_DIRECT_CALL:
            extra_compile_args.append("-DMKL_DIRECT_CALL")
    include_dirs.append(CONDA_DIR + '/include')
    library_dirs.append(CONDA_DIR+'/lib')
    TKlibraries=['mkl_rt',  'm', 'iomp5','blas']


elif ATLAS_DIR is not None:
    print ("Using ATLAS")
    include_dirs.append(ATLAS_DIR + '/include')
    library_dirs.append(ATLAS_DIR +'/lib')
    TKlibraries=['blas', 'cblas', 'atlas']

elif OPENBLAS_DIR is not None:
    print ("Using OPENBLAS")
    include_dirs.append(OPENBLAS_DIR + '/include')
    library_dirs.append(OPENBLAS_DIR +'/lib')
    TKlibraries=['openblas']

else:
    include_dirs.append('/usr/include')
    #include_dirs.append('/System/Library/Frameworks/vecLib.framework/Headers/') # < 10.9
    #include_dirs.append('/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers/')
    #include_dirs.append('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers/') #Works with Xcode
    library_dirs.append('/usr/lib')
    TKlibraries=['blas', 'lapack']


ext_modules = [	Extension("npc_helper", ["/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy/tenpy/linalg/npc_helper.pyx"],
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args,
    include_dirs = include_dirs, libraries = libraries+TKlibraries, library_dirs = library_dirs,
    define_macros=[('CYTHON_TRACE', '0')],
    language='c++')
    ]

print('PARALLEL_TDOT', PARALLEL_TDOT)
setup(ext_modules = cythonize(ext_modules, language_level = "2", compile_time_env={'HAS_MKL': HAS_MKL, 'PARALLEL_TDOT': PARALLEL_TDOT, 'BLAS_COMPLEX_INNER':BLAS_COMPLEX_INNER}))


