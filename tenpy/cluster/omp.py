import ctypes
import ctypes.util
from ctypes import CDLL
import threading
import os

"""
if os.environ.get('OS','') == 'Windows_NT':
	Win32	= True
	import psutil
else:
	Win32	= False
	import resource
"""
use_psutil = False
use_resource = False
try:
	import resource
	use_resource = True
except:
	try:
		import psutil
		use_psutil = True
	except:
		print( "resource and psutil didn't import correctly.")
	

""" Fist, I've  wrapped the OpenMP library here so that I can set omp_num_threads during runtime. This is useful if numpy is compiled against MKL. MKL will follow the OpenMP directives, so using these functions you can query and control the number of threads used by mkl-NumPy at runtime. This is useful for load balancing in conjunction with OpenMPI. 

	Second, I have implemented a class which periodically checks if it should change num_threads. 

MPZ """

#Try to find openmp libs. Won't work on OS X because (in 10.7 at least) libgomp is static

libs = ["libiomp5.so", ctypes.util.find_library("libiomp5md"), ctypes.util.find_library("gomp")]

omp = None
for l in libs:
	if l==None:
		continue
	try:
		omp = CDLL(l)
		print ("Loaded " + l +  " for omp")
		break
	except OSError:
		pass

if omp == None:
	print ("Couldn't find omp")


def set_num_threads(n):
	if omp is not None:
		omp.omp_set_num_threads(n)
		
def set_dynamic(n):
	if omp is not None:
		omp.omp_set_dynamic(n)

def set_nested(n):
	if omp is not None:
		omp.omp_set_nested(n)

def set_max_active_levels(n):
	if omp is not None:
		omp.omp_set_max_active_levels(n)

def set_thread_limit(n):
	if omp is not None:
		omp.omp_set_max_active_levels(n)

def get_num_threads():
	if omp is not None:
		return omp.omp_get_num_threads()

def get_max_threads():
	if omp is not None:
		return omp.omp_get_max_threads()

def get_dynamic():
	if omp is not None:
		return omp.omp_get_dynamic()

def get_nested():
	if omp is not None:
		return omp.omp_get_nested()

def get_max_active_levels():
	if omp is not None:
		return omp.omp_get_max_active_levels()


try:
	mkl_rt = ctypes.CDLL('libmkl_rt.so')
except:
	mkl_rt = None

def mkl_get_max_threads():
	if mkl_rt is not None:
		return mkl_rt.mkl_get_max_threads()

def mkl_set_num_threads(n):
	if mkl_rt is not None:
		mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))

def mkl_set_num_threads_local(n):
	if mkl_rt is not None:
		mkl_rt.mkl_set_num_threads_local(ctypes.byref(ctypes.c_int(n)))

def mkl_set_dynamic(n):
	if mkl_rt is not None:
		mkl_rt.mkl_set_dynamic(ctypes.byref(ctypes.c_int(n)))

"""
libs = [ctypes.util.find_library("openblas")]
if 'OPENBLAS_DIR' in os.environ:
	libs.append(os.environ['OPENBLAS_DIR'] + "/bin/libopenblas.dll")

for l in libs:
	if l==None:
		continue
	try:
		openblas = CDLL(l)
		break
	except OSError:
		pass

def openblas_set_num_threads(n):
	if openblas is not None:
		openblas.openblas_set_num_threads(ctypes.byref(ctypes.c_int(n)))
"""


def memory_usage():
	"""Memory usage of the current process in kilobytes."""
	
	"""
	if Win32:
		proc = psutil.Process()
		mem	 = proc.memory_info()
		return  mem.rss/1024.
	else:
		return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
	"""
	if use_psutil:
		proc = psutil.Process()
		mem	 = proc.memory_info()
		return  mem.rss/1024./1024.
	elif use_resource:
		return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.


class nthread_poller(object):
	""" 
		Calls the function 'func' every 'wait' seconds. 'func' should return an integer, the number of threads to use.
	"""

	def __init__(self, func, args = [], kwargs = {}, wait = 120.):
		self.polling = False
		self.wait = wait
		self.func = func
		self.args = args
		self.kwargs = kwargs
		self.timer = None
		self.nthreads = get_max_threads()
		
	def check(self):
		nthreads = self.func(*self.args, **self.kwargs)
		#set_num_threads(nthreads)
		#print "Cur", threading.active_count()
		global target_max_threads 
		target_max_threads = nthreads
		if nthreads != self.nthreads:
			print ("Switched omp_num_threads to", nthreads)
			self.nthreads = nthreads
		
		if self.polling:
			self.timer = threading.Timer(self.wait, self.check)
			self.timer.daemon = True
			self.timer.start()
		
	def start_poll_for_nthreads(self):
		self.polling = True
		self.timer = threading.Timer(self.wait, self.check)
		self.timer.daemon = True
		self.timer.start()
	
	def stop_poll_for_nthreads(self):
		self.polling = False
		if self.timer is not None:
			self.timer.cancel()
			self.timer.join()
		
#set_dynamic(1)
print ("Running with omp_max_threads=", get_max_threads())
target_max_threads = get_max_threads()

