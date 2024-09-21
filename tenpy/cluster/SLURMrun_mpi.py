import time
import os
import imp
import fileinput
from mpi4py import MPI
import sys
import numpy as np
import parse
do_omp_balancing = True
omp_max_threads = 20

if do_omp_balancing:
	import cluster.omp as omp

args = sys.argv[1:]
modelName = args[0]
jobName = args[1]

if len(args) > 2:
	program = args[2]
else:
	program = 'run'

IDlist = np.loadtxt( './simulations/' + modelName + '/RunFiles/j' + jobName + '_IDs.dat', dtype = np.int, ndmin = 1)

#with open('./simulations/' + modelName + '/run.py', 'r') as file:
#	run = imp.load_source('run',file)
run = imp.load_source('run','./simulations/' + modelName + '/' + program + '.py')
WORKTAG = 1
DIETAG = 2
OMPTAG = 3

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print ("Proc", rank, "reporting for duty!")


def load_balance(node_dat, proc):
	old_nthread = node_dat['procs'][proc]
	min_thread = min(node_dat['procs'].viewvalues())
	if old_nthread > min_thread:
		return old_nthread
	
	procs_on_node = len( node_dat['procs'])
	free_cores = node_dat['ncores'] - node_dat['nthreads']
	#print "Balancing", old_nthread, procs_on_node, node_dat['nthreads'], node_dat['ncores'], free_cores
	if free_cores >= old_nthread and 2*old_nthread <= omp_max_threads:
		nthread = 2*old_nthread
		node_dat['procs'][proc] = nthread
		node_dat['nthreads'] += (nthread - old_nthread)
	else:
		nthread = old_nthread

	if nthread - old_nthread > 0:
		print ("Load balancing node", node_dat)
	return nthread     

def master(IDlist):
	T0 = time.time()
	numprocs = comm.Get_size()
	status = MPI.Status()
	jobsOut = 0
	
	try:
		SLURM_JOB_CPUS_PER_NODE = os.environ['SLURM_JOB_CPUS_PER_NODE'].split(',')
		#print "SLURM_JOB_CPUS_PER_NODE",SLURM_JOB_CPUS_PER_NODE
		ncores = []
		for a in SLURM_JOB_CPUS_PER_NODE:
			if a[-1] == ')':
				cpu, rep = parse.parse("{:d}(x{:d})", a)
				ncores.extend([cpu]*rep)
			else:
				ncores.append(int(a))
		#print "Cores / node", ncores	
		SLURM_NODELIST = os.environ['SLURM_NODELIST'].split(',')
		#print "Nodes", SLURM_NODELIST
		nodeList = {}
		for j in xrange(len(SLURM_NODELIST)):
			nodeList[SLURM_NODELIST[j]] = {'ncores':ncores[j], 'nthreads':0, 'procs':{} }
	except KeyError:	
		try:	
			PBS_NODEFILE = os.environ['PBS_NODEFILE']
			with open(PBS_NODEFILE) as file:
				nodefile = file.readlines()
			nodefile = [s.rstrip() for s in  nodefile]
			nodeList = dict( [ (node, {'ncores':nodefile.count(node), 'nthreads':0, 'procs':{} }) for node in set(nodefile) ])
		except KeyError:
			nodeList = {}

	print("NodeList", nodeList)
	procLoc = [None]*numprocs
	print ("Building node list . . . ")
	remaining = numprocs - 1
	if remaining == 0:
		print ("No procs allocated. Quitting")
		return
		
	while remaining:
		node, nthread = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
		proc = status.Get_source()
		procLoc[proc] = node

		if node not in nodeList:
			nodeList[node] = {'nthreads': nthread, 'procs': {proc: nthread}}
		else:
			nodeList[node]['procs'][proc] = nthread
			nodeList[node]['nthreads']+=nthread

		remaining-=1

		
	print ("Proc locations:")
	print (procLoc)
	print ("Node list:")
	print (nodeList)
            
	for dest in xrange(1, numprocs):
		node = procLoc[dest]
		if len(IDlist) > 0:
			print( "Master sending ID", IDlist[0], "to proc", dest)
			comm.send( IDlist[0], dest = dest, tag = WORKTAG)
			IDlist = IDlist[1:]
			jobsOut += 1
		else:
			print ("Master killing proc", dest)
			comm.send(-1, dest = dest, tag = DIETAG)
			nt = nodeList[node]['procs'].pop(dest)
			nodeList[node]['nthreads']-=nt
   
	print( "Initial workload:", nodeList)
	while len(IDlist) > 0:
		ID = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
		proc = status.Get_source()
		tag = status.Get_tag()

		if tag == OMPTAG:
			node = procLoc[proc]
			nthread = load_balance( nodeList[node], proc )
			#print "Master heard proc", proc, "wants to know nthread. Telling him", nthread
			comm.send(nthread, dest = proc, tag = OMPTAG)
			
		else:
			print ("Master heard proc ", proc, "finished ID", ID, "at T = ", time.time() - T0, " sending ID", IDlist[0])
			comm.send( IDlist[0], dest = proc, tag = WORKTAG)
			IDlist = IDlist[1:]
		
	#All the jobs are out for processing now.
	while  jobsOut:
		ID = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)
		proc = status.Get_source()
		tag = status.Get_tag()
		
		if tag == OMPTAG:
			node = procLoc[proc]
			nthread = load_balance( nodeList[node], proc )
			#print "Master hears proc", proc, "wants to know nthread. Telling him", nthread
			comm.send(nthread, dest = proc, tag = OMPTAG)
		else:
			node = procLoc[proc]
			print ("Master heard proc ", proc, "finished ID", ID, "at T = ", time.time() -T0, "killing proc with", jobsOut -1, "jobs still out.")
			comm.send(-1, dest = proc, tag = DIETAG)
			nt = nodeList[node]['procs'].pop(proc)
			nodeList[node]['nthreads']-=nt
			jobsOut = jobsOut - 1

	return

def ask_for_nthread():
	#comm = MPI.COMM_WORLD
	#rank = comm.Get_rank()  
	comm.send(-1, dest = 0, tag = OMPTAG)
	return comm.recv(source = 0, tag = OMPTAG) 

def slave(jobName):
	#WORKTAG = 1
	#DIETAG = 2
	#comm = MPI.COMM_WORLD
	#rank = comm.Get_rank()
	status = MPI.Status()
	name = MPI.Get_processor_name()
	#print "Proc", rank, " is running on node", name
	
	if do_omp_balancing:
		nthreads = omp.get_max_threads()
	else:
		try:
			nthreads = os.environ['OMP_NUM_THREADS']
		except KeyValueError:
			nthreads = 1 
	
	comm.send((name, nthreads), dest = 0)
	
	ID = comm.recv(source = 0, tag = MPI.ANY_TAG, status = status)
	if status.Get_tag() == DIETAG:
		return
	if do_omp_balancing:
		poller = omp.nthread_poller(ask_for_nthread, wait = 500.)
		poller.start_poll_for_nthreads()
	else:
		poller = None

	while True:			
		t0 = time.time()
		#print "Proc", rank, "starting ID", ID
			
		#try:
		run.run(jobName, str(ID))
		#except:
		#	print "ID", ID, "raised exception during run"
					
		#print "Proc", rank, "finished ID", ID, "in time", time.time() - t0
		comm.send(ID, dest = 0)
		ID = comm.recv(source = 0, tag = MPI.ANY_TAG, status = status)
		if status.Get_tag() == DIETAG:
			if poller!=None:
				poller.stop_poll_for_nthreads()
			return

if rank == 0:
    master(IDlist)
else:
    slave(jobName)


print( "Proc", rank, "signing off")
