import numpy as np
import scipy as sp
from time import time
from tenpy.tools.string import joinstr
from tenpy.tools.math import anynan

try:
	import control
	import slycot
	Loaded_Sylcot = True
except:
	print("error import control or slycot")
	Loaded_Sylcot = False
try:
	from algorithms.linalg.more_slicot import more_slicot
	Loaded_more_slicot = True
except ImportError:
	Loaded_more_slicot = False

################################################################################
################################################################################
##	Markov approximations
##
##		Seq_{a,b}  ~  exits_{a,p} . (M^b)_{p,q} . entry_{q}


class Markov_seq(object):
	"""	Markov chain class to generate sequences.
			The sequences are defined as:
				Seq_{a,b}  =  exits_{a,p} . (M^b)_{p,q} . entry_{q}

			The internal variables are
				N: the number of nodes / internal states of the Markov process,
				M: a matrix shaped (N, N),
				entry: a vector shaped (N,),
				exits: a matrix is shaped (# sequence, N),
				dtype: the data type,
				info: a dictionary that contains misc data.
		"""

	def __init__(self, dtype = float, default_seq_info = False):
		self.N = 0		# number of nodes
		self.M = np.zeros([0,0], dtype=dtype)
		self.entry = np.zeros([0], dtype=dtype)
		self.exits = np.zeros([0,0], dtype=dtype)
		self.dtype = dtype
		self.info = {}
		if default_seq_info: self.fill_seq_info(np.zeros([0,0]))


	def MeeInfo(self):
		return self.M, self.entry, self.exits, self.info

	def Mee(self):
		return self.M, self.entry, self.exits


	def check_sanity(self):
		N = self.N
		dtype = self.dtype
		if not (isinstance(N, int) or isinstance(N, long)): raise RuntimeError("N is not an int/long (type=" + str(type(N)) + ")")
		if N < 0: raise RuntimeError("N is negative: N = " + str(N))
	##	M
		if not isinstance(self.M, np.ndarray): raise RuntimeError("M not an np.ndarray: " + str(type(self.M)))
		if self.M.shape != (N, N): raise RuntimeError("incorrect M.shape %s for N = %s" % (self.M.shape, N))
		if self.M.dtype != dtype: raise RuntimeError("M.dtype (%s) does not match dtype (%s)" % (self.M.dtype, dtype))
	##	entry
		if not isinstance(self.entry, np.ndarray): raise RuntimeError("entry not an np.ndarray: " + str(type(self.entry)))
		if self.entry.shape != (N,): raise RuntimeError("incorrect entry.shape %s for N = %s" % (self.entry.shape, N))
		if self.entry.dtype != dtype: raise RuntimeError("entry.dtype (%s) does not match dtype (%s)" % (self.entry.dtype, dtype))
	##	exits
		if not isinstance(self.exits, np.ndarray): raise RuntimeError("exits not an np.ndarray: " + str(type(self.exits)))
		if self.exits.ndim != 2 or self.exits.shape[1] != N: raise RuntimeError("incorrect exits.shape %s for N = %s" % (self.exits.shape, N))
		if self.exits.dtype != dtype: raise RuntimeError("exits.dtype (%s) does not match dtype (%s)" % (self.exits.dtype, dtype))
	##	Misc...
		if not isinstance(self.info, dict): raise RuntimeError("info not a dict: " + str(type(self.info)))


	def generate_seq(self, l):
		"""	Generate sequences (of length l) from M, entry, exits.  """
		M = self.M
		v = self.entry
		exits = self.exits
		seqs = np.zeros([len(exits), l], dtype=exits.dtype)
		for i in xrange(l):
			seqs[:, i] = np.dot(exits, v)
			if i < l - 1: v = np.dot(M, v)
		return seqs
	

	def fill_seq_info(self, seqs):
		diff = self.generate_seq(seqs.shape[1]) - seqs
		self.info['norm_err'] = np.linalg.norm(diff.reshape(-1))
		if diff.shape[0] * diff.shape[1] > 0:
			self.info['max_abs_err'] = np.max(np.abs(diff))
		else:
			self.info['max_abs_err'] = 0.
		self.info['seq_norm'] = np.array([ np.linalg.norm(s.reshape(-1)) for s in seqs ])
		self.info['l'] = seqs.shape[1]
		#self.plot_seqs_diff(seqs)


	def plot_seqs_diff(self, seqs):
	##	TODO, match colours between line & points
		if seqs.shape[1] == 0: return
		import matplotlib.pyplot as plt
		genseq = self.generate_seq(seqs.shape[1] + 1)
		for i,s in enumerate(genseq):
			plt.plot(s)
		for i,s in enumerate(seqs):
			plt.plot(s, 'o')
		diff = self.generate_seq(seqs.shape[1])[:, :seqs.shape[1]] - seqs
		norm_err = np.linalg.norm(diff.reshape(-1))
		plt.title('Fit with %s nodes (error %s)' % (self.N, norm_err))
		plt.show()


	@classmethod
	def fit_to_exp_nodes(cls, seqs, nodes):
		if seqs.dtype == complex or seqs.dtype == np.complex128:
			num_seq, l = seqs.shape
			Mk = Markov_seq(dtype = seqs.dtype)
			Mk.N = len(nodes)
			Mk.M = np.diag(nodes)
			Mk.entry = np.ones(len(nodes))
			Mk.exits, err = fit_exp_coef(seqs, nodes)
			Mk.fill_seq_info(seqs)
			return Mk
		else:
			return Markov_seq.fit_real_to_exp_nodes(seqs, nodes)


	@classmethod
	def fit_real_to_exp_nodes(cls, seqs, nodes):
		"""	Given the nodes (which are complex, but comes in conjugate pairs), find the best fit to the real sequences.
				Uses cos/sin instead of exp to get real numbers.
			"""
		if seqs.dtype == complex or seqs.dtype == np.complex128: raise TypeError
		##	Construct transformation matrix to real
		cp, cp_err = find_conj_pairs(nodes)		# [cp[i]] gives the cc for [i]
		UU = np.eye(len(nodes), dtype=complex)
		sqhf = np.sqrt(0.5)
		for i in xrange(len(nodes)):
			if cp[i] <= i: continue		# only process cp[i] > i cases.
			UU[    i,     i] = sqhf
			UU[    i, cp[i]] = sqhf
			UU[cp[i],     i] = -1j * sqhf
			UU[cp[i], cp[i]] = 1j * sqhf
		UUD = np.conj(UU.transpose())		# UU^\dag

		num_seq, l = seqs.shape
		Mk = Markov_seq(dtype = seqs.dtype)
		Mk.N = len(nodes)
		Mk.M = np.diag(nodes)
		Mk.entry = np.ones(len(nodes))
		Mk.exits, err = fit_exp_coef(seqs, nodes)

		Mk.M = np.dot( np.dot(UU, Mk.M), UUD ).real
		Mk.entry = np.dot(UU, Mk.entry).real
		Mk.exits = np.dot(Mk.exits, UUD).real

		Mk.fill_seq_info(seqs)
		return Mk
		

################################################################################
################################################################################
## Constructors

def seq_approx(seqs, target_err=np.inf, target_frac_err=1e-4, max_nodes=None):
	"""	Turn the sequeneces of numbers in to a Markov process.
			Returns a Markov_seq.
		"""
	if not (target_err >= 0.): raise ValueError(target_err)	# nan not acceptable
	if not (target_frac_err >= 0.): raise ValueError(target_frac_err)
	seqs = atleast_2d_pad(seqs)
	num_seq, l = seqs.shape
	if num_seq == 0: return Markov_seq(default_seq_info = True)
	if max_nodes is None: max_nodes = l
	if max_nodes < 0: raise ValueError(max_nodes)
	target_err = np.min([target_err, np.sum(np.abs(seqs)) * target_frac_err])

	Mk = KungLin_Hankel_approx(seqs, max_nodes=l, svd_cut=target_err)
#	if num_seq == 1:
#		Mk = BM_Hankel_approx(seqs[0], max_nodes=16, svd_cut=target_err)
#	else:
#		nodes_list = []
#		for s in seqs:
#			nodes_list.append(BM_Hankel_approx_nodes(s, max_nodes=16, svd_cut=target_err))
#		nodes = np.hstack(nodes_list)
#		Mk = truncate_approx(seqs, truncate=target_err, max_nodes=max_nodes)
#		if len(nodes) < Mk.N:
#			##	Fit to nodes from the BM_Hankel algorithm applied to each sequence
#			Mk = Markov_seq.fit_to_exp_nodes(seqs, nodes)
			
	Mk.check_sanity()
	return Mk
	
	
def truncate_approx(seqs, truncate=0., max_nodes=None):
	"""	Generate a Markov process that spilts out numbers in order and then stops.
			Uses a trivial truncation scheme.
			Returns a Markov_seq.
		"""
	seqs = atleast_2d_pad(seqs)
	num_seq, l = seqs.shape
	if max_nodes is None: max_nodes = l
	if max_nodes < 0: raise ValueError(max_nodes)
	if num_seq <= 0: return Markov_seq(default_seq_info = True)
	##	Find actual 'l' from truncation
	keeploc = np.nonzero(np.abs(seqs) > truncate)[1]	# columns where seqs > truncate
	if len(keeploc) > 0:
		l = np.max(keeploc) + 1
		l = min(l, max_nodes)
	else:
		l = 0
	##	Construct the trivial Markov chain
	Mk = Markov_seq(dtype = seqs.dtype)
	Mk.N = l
	Mk.M = np.eye(l, k = -1, dtype=seqs.dtype)
	Mk.entry = np.zeros(l, dtype=seqs.dtype)
	if l > 0: Mk.entry[0] = 1
	Mk.exits = seqs[:, :l].copy()
	Mk.fill_seq_info(seqs)
	return Mk
	
	
def BM_Hankel_approx(seq, max_nodes=3, svd_cut=0):
	"""	Aim to fit a single sequence with a set number of nodes.
			If the desired accuracy (svd_cut) can be acheived with fewer nodes, it will.
			Returns a Markov_seq (with # of sequences = 1).
		"""
	seq = np.array(seq)
	nodes = BM_Hankel_approx_nodes(seq, max_nodes, svd_cut)
	#if len(nodes) == 0: return Markov_seq(default_seq_info = True)
	return Markov_seq.fit_to_exp_nodes(np.array([seq]), nodes)


def BM_Hankel_approx_nodes(seq, max_nodes, svd_cut):
	"""	Aim to fit a single sequence with a set number of nodes.
			If the desired accuracy (svd_cut) can be acheived with fewer nodes, it will.
			Returns a list of nodes.
		"""
	l = len(seq)
	if l <= 2: raise NotImplementedError
	##	Construct the Hankel matrix
	N = l // 2 + 1
	Hk = np.zeros([N, N], dtype = seq.dtype)
	for r in xrange(l - N + 1):
		Hk[r] = seq[r:r+N]
	if l + 1 < 2*N: Hk[N-1, :N-1] = seq[N-1:]		# l even = 2N-2
	##	SVD it and find the characteristic polynomial
	U,s,V = np.linalg.svd(Hk)	# U s V, s sorted in decreasing order
	mike = min(np.count_nonzero(s > svd_cut), max_nodes)
	if mike >= N: mike = N - 1	# should do something else?
	polycoef = np.conj(V[mike, :])[-1::-1]
	roots = np.roots(polycoef)
	nodes = roots[np.abs(roots) < 1.]
	if len(nodes) > max_nodes: nodes = nodes[:max_nodes]
	return nodes


def KungLin_Hankel_approx(seqs, max_nodes, svd_cut):
	seqs = np.array(seqs)
	l = seqs.shape[1]
	if l == 0: raise NotImplementedError
	HMx = np.vstack([ sp.linalg.hankel(s) for s in seqs ])
	U,s,V = np.linalg.svd(HMx)	# HMx = U s V
	m = min(np.count_nonzero(s > svd_cut), max_nodes)
	if m >= l: return truncate_approx(seqs)
	polycoef = np.conj(V[m, :])[-1::-1]
	roots = np.roots(polycoef)
	nodes = roots[np.abs(roots) < 1.]
	if len(nodes) > max_nodes: nodes = nodes[:max_nodes]
	#print HMx.shape, m, nodes
	return Markov_seq.fit_to_exp_nodes(seqs, nodes)
	




################################################################################
################################################################################
## Helper functions

def atleast_2d_pad(a, pad_item=0):
	"""	Given a list of lists, turn it to a 2D array (pad with 0), or turn a 1D list to 2D.
			Returns a numpy.ndarray.
			
			Examples:
			>>>	atleast_2d_pad([3,4,0])
				array([[3, 4, 0]])
			>>>	atleast_2d_pad([[3,4],[1,6,7]])
				array([[ 3.,  4.,  0.],
				       [ 1.,  6.,  7.]])
		"""
	iter(a)
	if len(a) == 0: return np.zeros([0,0])
	##	Check if every element of a is a list
	is_list_of_list = True
	for s in a:
		try:
			iter(s)
		except TypeError:
			is_list_of_list = False
			break
	if is_list_of_list:
		a2 = a
	else:
		try:
			return np.array([a])
		except ValueError:
			return [a]
	##	Pad if necessary
	maxlen = max([ len(s) for s in a2 ])
	for r in xrange(len(a2)):
		s = a2[r]
		a2[r] = np.hstack([s, [pad_item] * (maxlen-len(s))])
	return np.array(a2)
	

def fit_exp_coef(seqs, nodes):
	"""	Compute the best fits to
				seqs[a, b] = sum_n coef[a, n] nodes[n]^b
			
			seqs has shape (num_seq, l), coef has shape (num_seq, len(nodes)).
			Return the best fit (coef's), and the residue (norm error)
		"""
	num_seq, l = seqs.shape
	if num_seq == 0 or l == 0 or len(nodes) == 0: return np.zeros([num_seq, len(nodes)]), 0.
	A = np.empty([l, len(nodes)], dtype=nodes.dtype)	# columns are powers
	A[0] = 1
	for r in xrange(1, l):
		A[r] = A[r-1] * nodes
	fit = np.linalg.lstsq(A, seqs.transpose())	# approx Ax = b
	return fit[0].transpose(), np.sqrt(fit[1])
	

def find_conj_pairs(x):
	"""	Given a 1-array x, match up each element with its complex conjugate pair.
			Returns: conj_pair, err.
				x[conj_pair[i]] is the c.c. of x[i].
				err = 2-norm of x[conj_pair] - x
			
			Example:
			>>>	find_conj_pairs( np.array([0,3j,5,5+1j,-3j,5-0.9j]) )
			(array([0, 4, 2, 5, 1, 3]), 0.14142135623730948)
		"""
	y = x.copy()
	if np.any(np.isnan(y)) or np.any(np.isinf(y)): raise ValueError
	conj_pair = np.zeros(len(y), dtype=int)
	err2 = 0.
	for i,v in enumerate(y):
		if np.isinf(v): continue	# already been processed
		cv = np.conj(v)
		ci = np.argmin(np.abs(y - cv))
		conj_pair[i] = ci
		conj_pair[ci] = i
		if i == ci:
			err2 += np.abs((cv - y[ci])**2)
		else:
			err2 += 2 * np.abs((cv - y[ci])**2)
		y[i] = np.inf	# flag these so they don't get used again
		y[ci] = np.inf
	return conj_pair, np.sqrt(err2)



################################################################################
################################################################################
##	MPO_system stores and process the data required to make an exponential approximation to an MPO.
##
##	self.io is a list of dictionaries, which we call 'blocks'.
##	Each block represents a q-in p-out system.
##		block['out'] is a list of length p, the elements are key names
##		block['in'] is a list of length q, the elements are key names
##		block['seq'] is an ndarray shaped (p,q,#r)
##
##	Each block may have additional keys: 'name', 'ignore_r0', 'StrOp'
##	Calling compute_ABC() creates the keys 'A', 'B', 'C' and 'N' for each block.
##		(A matrix is N x N, B is N x q, C is p x N, seq[:,:,r] = C A^r B)
##
##	Calling create_MPO_nodes(MPOgraph) makes the nodes and connect them.  (See function documentation)

class MPO_system(object):
	def __init__(self, dtype=float):
		self.Din = {}
		self.Dout = {}
		self.io = []	# list of dictionaries with keys 'in', 'out', 'seq'
		self.dtype = dtype
		self.info = {}
		self.node_header = ('Mk',)
		self.counter = 0


	def check_sanity(self):
		Din = self.Din
		Dout = self.Dout
		io = self.io
		for i,D in Din.iteritems():
			key = D['index']
			if len(key) != 2: raise RuntimeError(len(key))
			if key[0] < 0 or key[0] >= len(io): raise RuntimeError(key)
			ioblock = io[key[0]]
			if key[1] < 0 or key[1] >= len(ioblock['in']): raise RuntimeError(key)
			if ioblock['in'][key[1]] != i: raise RuntimeError
			if 'conj' in D:
				ic = D['conj']
				if ic not in Din: raise RuntimeError((i, ic))
				if Din[ic]['conj'] != i: raise RuntimeError
		for o,D in Dout.iteritems():
			key = D['index']
			if key[0] >= len(io): raise RuntimeError(key)
			ioblock = io[key[0]]
			if key[1] >= len(ioblock['out']): raise RuntimeError(key)
			if ioblock['out'][key[1]] != o: raise RuntimeError((o, key, ioblock['out']))
			if 'conj' in D:
				oc = D['conj']
				if oc not in Dout: raise RuntimeError((o, oc))
				if Dout[oc]['conj'] != o: raise RuntimeError
		for block_index,b in enumerate(io):
			if b is None: continue
			if not isinstance(b, dict): raise RuntimeError(type(b))
			if ('in' not in b) or ('out' not in b) or ('seq' not in b): raise RuntimeError(b.keys())
			for i in b['in']:
				if i not in Din: raise RuntimeError(i)
				if Din[i]['index'][0] != block_index: raise RuntimeError((block_index, Din[i]['index']))
			for o in b['out']:
				if o not in Dout: raise RuntimeError(o)
				if Dout[o]['index'][0] != block_index: raise RuntimeError((block_index, Dout[o]['index']))
			#	Each seqeuence 'seq' must be a 3-array, shaped (p, q, #)
			seq = b['seq']
			if not isinstance(seq, np.ndarray): raise RuntimeError(type(seq))
			if seq.ndim != 3: raise RuntimeError(seq.shape)
			if seq.shape[0:2] != (len(b['out']), len(b['in'])): raise RuntimeError((seq.shape, len(b['out']), len(b['in'])))
			if seq.dtype != self.dtype: raise RuntimeError((seq.dtype, self.dtype))
		##	Check for optional keys
			if 'N' in b:
				N = b['N']
				#TODO ........
			if 'conj' in b:
				bc_id, in_map, out_map = b['conj']
				bcc_id, bc_in_map, bc_out_map = io[bc_id]['conj']
				if bcc_id != block_index: raise RuntimeError
				if np.any( bc_in_map[in_map] != np.arange(len(b['in'])) ): raise RuntimeError((in_map, bc_in_map, len(b['in'])))
				if np.any( bc_out_map[out_map] != np.arange(len(b['out'])) ): raise RuntimeError((out_map, bc_out_map, len(b['out'])))
				seq_conj = np.conj( io[bc_id]['seq'][out_map, :, :][:, in_map, :] )
				if seq_conj.shape != seq.shape: raise RuntimeError("This really really shouldn't happen.")
				norm_diff = np.linalg.norm(seq_conj - b['seq'])
				if norm_diff > 1e-14:
					raise RuntimeError((block_index, bc_id, np.linalg.norm(seq_conj - b['seq']), np.linalg.norm(b['seq'])))
				#TODO ........
		if not isinstance(self.info, dict): raise RuntimeError
		if not isinstance(self.node_header, tuple): raise RuntimeError(self.node_header)


	def add_io_single_r(self, i, o, r, val):
		if r < 0: raise ValueError((i, o, r, val))
		Din = self.Din
		Dout = self.Dout
		io = self.io
#		if self.counter >= 0: print self.counter, i, o, r, val
		
		##
		if (i not in Din):
			if (o not in Dout):
				ikey = Din[i] = { 'index': [len(io), 0] }
				okey = Dout[o] = { 'index': [len(io), 0] }
				io.append({ 'in': [i], 'out': [o], 'seq': np.zeros([1,1,r+1], dtype=self.dtype) })
			else:		# add new in key
				okey = Dout[o]['index']
				ioblock = io[okey[0]]
				ikey = Din[i] = { 'index': [okey[0], len(ioblock['in'])] }
				ioblock['in'].append(i)
				spad = max(r + 1 - ioblock['seq'].shape[2], 0)
				ioblock['seq'] = np.pad( ioblock['seq'], [(0,0),(0,1),(0,spad)], mode='constant' )
		else:
			ikey = Din[i]['index']
			if (o not in Dout):		# add new out key
				ioblock = io[ikey[0]]
				okey = Dout[o] = { 'index': [ikey[0], len(ioblock['out'])] }
				ioblock['out'].append(o)
				spad = max(r + 1 - ioblock['seq'].shape[2], 0)
				ioblock['seq'] = np.pad( ioblock['seq'], [(0,1),(0,0),(0,spad)], mode='constant' )
			else:
				okey = Dout[o]['index']
				if ikey[0] != okey[0]:
				#	print '\t', ikey, '->', okey
				#	print '\t', ikey[0], io[ikey[0]]['in'], '->', io[ikey[0]]['out']
				#	print '\t', okey[0], io[okey[0]]['in'], '->', io[okey[0]]['out']
					self.combine_ioblock(ikey[0], okey[0])
					ikey = Din[i]['index']
					okey = Dout[o]['index']
				#	print '\t', ikey, okey, io[ikey[0]]['in'], '->', io[ikey[0]]['out']
				ioblock = io[ikey[0]]
				if r + 1 > ioblock['seq'].shape[2]:
					ioblock['seq'] = np.pad( ioblock['seq'], [(0,0),(0,0),(0,r+1-ioblock['seq'].shape[2])], mode='constant' )
					
		##
		ikey = Din[i]['index']
		okey = Dout[o]['index']
		seq = io[ikey[0]]['seq']
		seq[okey[1], ikey[1], r] += val
#		if self.counter >= 0: print self.counter, io[ikey[0]]['in'], '->', io[ikey[0]]['out'], '\n  ', seq
		
		self.counter += 1
#		if self.counter > 7288: quit()


	def combine_ioblock(self, block_index1, block_index2):
		"""	Combine io[block_index1] with io[block_index2], and store in io[block_index1].
				Delete io[block_index2]
			"""
		Din = self.Din
		Dout = self.Dout
		io = self.io

		b1 = io[block_index1]
		b2 = io[block_index2]
		si = len(b1['in'])
		so = len(b1['out'])
		for j,k in enumerate(b2['in']):
			Din[k]['index'] = [block_index1, si+j]
		for j,k in enumerate(b2['out']):
			Dout[k]['index'] = [block_index1, so+j]
		b1['in'] += b2['in']
		b1['out'] += b2['out']
		
		seq1 = b1['seq']
		seq2 = b2['seq']
		seq = np.zeros( [so+seq2.shape[0], si+seq2.shape[1], max(seq1.shape[2],seq2.shape[2])], dtype=self.dtype )
		seq[:so, :si, :seq1.shape[2]] = seq1
		seq[so:, si:, :seq2.shape[2]] = seq2
		b1['seq'] = seq
		io[block_index2] = None
		self.remove_empty_block(block_index2)
		self.check_sanity()


	def remove_empty_block(self, block_index):
		io = self.io
		if block_index >= len(io): raise ValueError((lock_index, len(io)))
		if io[block_index] is not None: raise ValueError((block_index))
		if block_index == len(io) - 1:
			del io[block_index]
			return
		block = io[block_index] = io[len(io) - 1]
		for j,i in enumerate(block['in']):
			self.Din[i]['index'] = [block_index, j]
		for j,o in enumerate(block['out']):
			self.Dout[o]['index'] = [block_index, j]
		del io[len(io) - 1]
		self.check_sanity()


	def pair_conj_blocks(self):
		"""	Given: 'conj' key for each ikey / okey, make block['conj']
				block['conj'] = [ block_conj_id, in_map, out_map ]
					in_map[k] gives the location of in[k]* within block*[in']
			"""
		Din = self.Din
		Dout = self.Dout
		io = self.io
		for i,D in Din.iteritems():
			if 'conj' not in D: continue
			ic = D['conj']
			bindex = D['index'][0]
			io[bindex]['conj'] = [ Din[ic]['index'][0], None, None ]
		for o,D in Dout.iteritems():
			if 'conj' not in D: continue
			oc = D['conj']
			bindex = D['index'][0]
			io[bindex]['conj'] = [ Dout[oc]['index'][0], None, None ]
		for bindex,block in enumerate(io):
			bcindex = block['conj'][0]
			block['conj'][2] = np.array([ Dout[Dout[o]['conj']]['index'][1] for o in block['out'] ])
			block['conj'][1] = np.array([ Din[Din[i]['conj']]['index'][1] for i in block['in'] ])
			
		self.check_sanity()


	def compute_ABC(self, tol, verbose=0, verbose_prepend='\t'):
		"""	Take the 'seq' key of each block and make the A,B,C matrices.
				Require each block to have 'seq' key, and adds the following keys:
					'A'
					'B'
					'C'
					'N'
					'norm_err'
			"""
		for block_index,block in enumerate(self.io):
			if block is None: continue
			seq = block['seq']
			if 'conj' in block and block['conj'][0] < block_index:
				##	Use the previous computation
				bc_id, in_map, out_map = block['conj']
				blockc = self.io[bc_id]
				block['N'] = blockc['N']
				block['A'] = np.conj(blockc['A'])
				block['B'] = np.conj(blockc['B'])[:, in_map]
				block['C'] = np.conj(blockc['C'])[out_map, :]
				block['norm_err'] = blockc['norm_err']
				#seq_diff = ABC_output(seq.shape[2], block['A'], block['B'], block['C'], dtype=self.dtype) - seq
				#if seq_diff.shape[2] > 0 and 'ignore_r0' in block: seq_diff[:, :, 0] *= (1 - block['ignore_r0'])
				#block['norm_err'] = np.linalg.norm(seq_diff.reshape(-1))
				block['num_above_tol'] = blockc['num_above_tol']

			else:
				seq_norm = np.linalg.norm(seq.reshape(-1))
				if verbose > 0:
					print (verbose_prepend + "slycot processing {}/{}: shape={} norm={}tol".format( block_index, len(self.io), seq.shape, int(seq_norm/tol) ))
				if seq_norm < tol:
					block['N'] = 0
					block['A'] = np.zeros([0,0], dtype=seq.dtype)
					block['B'] = np.zeros([0,len(block['in'])], dtype=seq.dtype)
					block['C'] = np.zeros([len(block['out']),0], dtype=seq.dtype)
					seq_comp = np.zeros_like(seq)
				else:
					t0 = time()
					if 'ignore_r0' in block:
						sys = SS_fromV(seq, ignore_r0 = block['ignore_r0'])
					else:
						sys = SS_fromV(seq, cutoff = 1e-14)


					if verbose > 0: print ('v({0:.1f}s)'.format(time() - t0))
					t0 = time()
					if len(sys.A) > 1:
						sys_comp, hsv = SS_compress(sys, tol = tol)
						##sys_comp = SS_eig_canon(sys_comp)
					else:
						sys_comp = sys
					if verbose > 0: print('c({0:.1f}s)'.format(time() - t0))
					block['A'] = sys_comp.A.copy(order = 'C')
					block['B'] = sys_comp.B.copy(order = 'C')
					block['C'] = sys_comp.C.copy(order = 'C')
					block['N'] = len(block['A'])
					seq_comp = SS_output(sys_comp, seq.shape[2])
			
				seq_diff = seq_comp - seq
				if seq_diff.shape[2] > 0 and 'ignore_r0' in block: seq_diff[:, :, 0] *= (1 - block['ignore_r0'])
				block['norm_err'] = np.linalg.norm(seq_diff.reshape(-1))
				block['num_above_tol'] = np.count_nonzero(np.abs(seq) >= tol)
				#print '  ', block['in'], block['out'], block['N']
				if verbose > 0: print ('.')		# terminate line

		self.info['ABC_tol'] = tol
		self.info['total_norm_err'] = np.linalg.norm([ b['norm_err'] for b in self.io ])


	def compute_norm(self):
		n2 = 0.
		for block in self.io:
			if block is None: continue
			seq_flat = block['seq'].reshape(-1)
			block['2norm'] = np.linalg.norm(seq_flat)
			block['infnorm'] = np.max(np.abs(seq_flat))
			n2 += block['2norm'] ** 2
		self.info['2norm'] = np.sqrt(n2)


	def create_MPO_nodes(self, MPOgraph):
		"""	Add nodes to an MPOgrpah given the appropriate data.
				Din items require keys: 'orb', 'src', 'Op'
				Dout items require keys: 'orb', 'dest', 'Op'
				io items have optional keys: 'StrOp', 'name'
			Also need self.node_header
			"""
		for blockid in xrange(len(self.io)):
			self.create_block_MPO_nodes(MPOgraph, blockid)
			

	def create_partial_MPO_nodes(self, MPOgraph, io_create_list, double_unconj_ioblocks=False):
		"""	For each element in io_create_list, add nodes to an MPOgraph.
				io_create_list should be a list of ioblock ids.
				See create_MPO_nodes for requirements.
			"""
		sqrt2 = np.sqrt(2)
		io_create_list = np.sort(io_create_list)
		if np.any(io_create_list < 0) or np.any(io_create_list >= len(self.io)): raise ValueError
		if len(io_create_list) > 1 and np.any(io_create_list[1:] == io_create_list[:-1]): raise ValueError("Duplicate elements.")
		for blockid in io_create_list:
			conj_paired = self.io[blockid]['conj'][0] in io_create_list
			if double_unconj_ioblocks and not conj_paired:
				self.create_block_MPO_nodes(MPOgraph, blockid, inFactor=sqrt2, outFactor=sqrt2)
			else:
				self.create_block_MPO_nodes(MPOgraph, blockid)


	def create_block_MPO_nodes(self, MPOgraph, blockid, inFactor=1., outFactor=1.):
		"""	Add nodes to an MPOgrpah given the appropriate data.
				Din items require keys: 'orb', 'src', 'Op'
				Dout items require keys: 'orb', 'dest', 'Op'
				io items have optional keys: 'StrOp', 'name'
			Also need self.node_header
			"""
		L = len(MPOgraph)
		Din = self.Din
		Dout = self.Dout
		block = self.io[blockid]
		if block is None: return
		if 'name' not in block: block['name'] = blockid		# default name
		head = self.node_header + (block['name'],)
		A = block['A']
		B = block['B'] * inFactor
		C = block['C'] * outFactor
		NMk = block['N']
		if NMk == 0: return
	##	Make the nodes in the loop
		MOp = block.get('StrOp', 'Id')		# operators to connect between Markov-loop nodes
		for n in xrange(NMk):
			for l in xrange(L - 1):
				if head + (n,) in MPOgraph[l]: print ('MPO_system Warning!  %s already exists in MPOgraph[%d]' % (head + (n,), l))
				MPOgraph[l][head + (n,)] = { head + (n,): [(MOp, 1.)] }
			if head + (n,) in MPOgraph[L - 1]: print ('MPO_system Warning!  %s already exists in MPOgraph[%d]' % (head + (n,), L - 1))
			XNode = MPOgraph[L - 1][head + (n,)] = {}
			for m in xrange(NMk):
				if A[m,n] != 0: XNode[head + (m,)] = [(MOp, A[m, n])]		# n -> m at unit cell boundary
	##	Make entries
		AB = np.dot(A, B)
		for b,i in enumerate(block['in']):
			D = Din[i]
			Gorb = MPOgraph[D['orb']]
			for n in xrange(NMk):
				if D['orb'] == L - 1:
					if AB[n, b] != 0: Gorb[D['src']][head + (n,)] = [(D['Op'], AB[n, b])]
				else:
					if B[n, b] != 0: Gorb[D['src']][head + (n,)] = [(D['Op'], B[n, b])]
	## Make exits
		for a,o in enumerate(block['out']):
			D = Dout[o]
			Gorb = MPOgraph[D['orb']]
			for n in xrange(NMk):
				if C[a, n] != 0: Gorb[head + (n,)][D['dest']] = [(D['Op'], C[a ,n])]
				

	def calc_MPOgraph_err(self, MPOgraph, i):
		"""	Retrace the path from i -> o and see if it matches the stored sequence.
			Return p numbers, where p is the number of outs for i.
			"""
		self.check_sanity()
		L = len(MPOgraph)
		Dout = self.Dout
		iD = self.Din[i]
		block = self.io[iD['index'][0]]
		head = self.node_header + (block['name'],)
		N = block['N']
		orb0 = iD['orb']
		MOp = block['StrOp']
		outs = {}
		for a,o in enumerate(block['out']):
			outs[(Dout[o]['dest'], Dout[o]['orb'])] = [a, o, Dout[o]['Op']]
		seq_out = np.zeros([len(block['out']), block['seq'].shape[2]], dtype=self.dtype)
		
		StartNode = MPOgraph[orb0][iD['src']]
		Amp = np.zeros(N, dtype=self.dtype)
		for n in xrange(N):	
			XDest = StartNode[head + (n,)]
			if len(XDest) > 1 or XDest[0][0] != iD['Op']: print("MPO_system Warning!  Entry node has improper form: %s, %s" % (n, XDest))
			Amp[n] = XDest[0][1]
		
		for orb in xrange(orb0 + 1, block['seq'].shape[2] * L):
			NewAmp = np.zeros(N, dtype=self.dtype)
			for n in xrange(N):
				XNode = MPOgraph[orb % L][head + (n,)]
				for Dest,XDest in XNode.iteritems():
					if isinstance(Dest, tuple) and Dest[:-1] == head:
						if len(XDest) > 1 or XDest[0][0] != MOp: print("MPO_system Warning!  Entry node has improper form: [%s]%s, %s" % (orb, n, XDest))
						m = Dest[-1]
						NewAmp[m] += XDest[0][1] * Amp[n]
					elif (Dest, orb % L) in outs:
						a, o, o_Op = outs[(Dest, orb % L)]
						if len(XDest) > 1 or XDest[0][0] != o_Op: print ("MPO_system Warning!  Entry node has improper form: [%s]%s, %s" % (orb, n, XDest))
						#if (orb % L) != o_orb: print "Warning!  %s -> %s orbital %s and %s don't match!" % (head + (n,), Dest, orb, o_orb)
						seq_out[a, orb // L] += XDest[0][1] * Amp[n]
					else:
						print ("MPO_system Warning!  MPOgraph[%s][%s] has unrecognized target %s" % (orb, head + (n,), Dest))
			Amp = NewAmp
		seq_diff = block['seq'][:, iD['index'][1], :] - seq_out
		if 'ignore_r0' in block: seq_diff[:, 0] *= (1 - block['ignore_r0'][:, iD['index'][1]])
		return [ np.linalg.norm(s) for s in seq_diff ]




################################################################################
################################################################################
## Slycot Stuff


def SS_fromV(V, shiftby_1 = True, ignore_r0 = None, cutoff = 1e-16):
	"""
		Input: V[a][b][r], stored either as list of list of rank-1 np.arrays, as single rank-3 np.array, or as a single rank-1 np.array (in which case a=b=0 is implicit).
		ignore_r0 is either array or list of lists - corresponding r=0 responses can be safely ignored.
		Output: Constructs an uncompressed state space  (A, B, C, D) representing V, returned as a control.StateSpace object.
		
		Let space (mu, nu) be p x q. Then:
			shape(D) = p x q
			shape(B) = N x q
			shape(C) = p x N
			shape(A) = N x N
		
		for N to be determined. Suppressing indices [mu][nu],
			V[0] = D
			V[r>0] = C A^(r-1) B
			
		which is succinctly encoded as the upper right pxq block of W^{r+1}, where
			    0 C D
			W = 0 A B
			    0 0 0
			
		of dim (p + N + q, p + N + q).
		
		A, B, C, D can be obtained via sys.A, etc...

		If shiftby_1 = True, shifts the signal later by 1,
			s1_new[r] = s1[r-1],
		padding with a zero.  This way D is irrelevant, and the first measurement is term CB.
		"""

	if isinstance(V, np.ndarray) and V.ndim == 1:
		V = np.reshape(V, (1, 1, -1))
	num_r = len(V)
	num_c = len(V[0])
	if ignore_r0 is not None:
		#if V.shape[0] == 1 or V.shape[1] == 1:
		#	print "Sorry Roger - not proceeding omptimally. Should do shifting strategy in this case"
		print ("This code PROBABLY works, but MPZ couldn't find test case! Now that you have, contact Mike")
		raise NotImplemented
		V = [[ vvv for vvv in r ] for r in V ]
		ignore_r0 = np.array(ignore_r0)
		for r in xrange(num_r):
			for c in xrange(num_c):
				if V[r][c] is not None and ignore_r0[r][c]:
					if len(V[r][c])==1:
						V[r][c]=None
					else:
						V[r][c][0]=V[r][c][1]

	l = np.zeros((num_r, num_c), dtype = np.int)
	v = np.empty((num_r, num_c), dtype = np.object)
	for r in xrange(num_r):
		for c in xrange(num_c):
			if V[r][c] is None:
				V[r][c] = np.array([0.])
			else:
				v[r][c] = V[r][c]
			if shiftby_1:
				v[r][c] = np.pad(v[r][c], (1, 0), mode='constant')
			keep = np.nonzero(np.abs(v[r][c]) > cutoff)[0]
			if len(keep) == 0:
				keep = np.array([0], dtype = np.int)
			v[r][c] = v[r][c][:keep[-1]+1]
			l[r,c] = len(v[r][c])

	l = np.max(l, axis = 1)-1
	l[l<0] = 0
	N = np.sum(l)
	A = np.zeros((N, N))
	B = np.zeros((N, num_c))
	C = np.zeros((num_r, N))
	D = np.zeros((num_r, num_c))
	at = 0
	for r in xrange(num_r):
		C[r, at] = 1.
		for c in xrange(num_c):
			D[r, c] = v[r][c][0]
			if len(V[r][c]) > 1:
				B[at:at+len(v[r][c]) - 1, c] = v[r][c][1:]
	
		for j in xrange(l[r] - 1):
			A[at+j, at+j+1] = 1.
		at = at + l[r]

	sys = control.ss(A, B, C, D, 1)
	
	output = SS_output(sys, np.max(l)+1)
	for r in xrange(len(V)):
		for c in xrange(len(V[r])):
			assert np.linalg.norm(V[r][c] - output[r, c, :len(V[r][c])]) < 1e-15
	return control.ss(A, B, C, D, 1)

def SS_fromV_tff2ss(V, shiftby_1 = True, ignore_r0 = None):
	"""
		Legacy code: uses the control tff2ss to construct ABCD from signal; new version (SS_fromV) is more stable.
	
		Input: V[a][b][r], stored either as list of list of rank-1 np.arrays, as single rank-3 np.array, or as a single rank-1 np.array (in which case a=b=0 is implicit).
		ignore_r0 is either array or list of lists - corresponding r=0 responses can be safely ignored.
		Output: Constructs an uncompressed state space  (A, B, C, D) representing V, returned as a control.StateSpace object.
		
		Let space (mu, nu) be p x q. Then:
			shape(D) = p x q
			shape(B) = N x q
			shape(C) = p x N
			shape(A) = N x N
		
		for N to be determined. Suppressing indices [mu][nu],
			V[0] = D
			V[r>0] = C A^(r-1) B
			
		which is succinctly encoded as the upper right pxq block of W^{r+1}, where
			    0 C D
			W = 0 A B
			    0 0 0
			
		of dim (p + N + q, p + N + q).
		
		A, B, C, D can be obtained via sys.A, etc...

		If shiftby_1 = True, shifts the signal later by 1,
			s1_new[r] = s1[r-1],
		padding with a zero.  This way D is irrelevant, and the first measurement is term CB.
		"""

	if isinstance(V, np.ndarray) and V.ndim == 1:
		V = np.reshape(V, (1, 1, -1))
	num_r = len(V)
	num_c = len(V[0])
	if ignore_r0 is not None:
		#if V.shape[0] == 1 or V.shape[1] == 1:
		#	print "Sorry Roger - not proceeding omptimally. Should do shifting strategy in this case"
		V = [[ vvv for vvv in r ] for r in V ]
		ignore_r0 = np.array(ignore_r0)
		for r in xrange(num_r):
			for c in xrange(num_c):
				if V[r][c] is not None and ignore_r0[r][c]:
					if len(V[r][c])==1:
						V[r][c]=None
					else:
						V[r][c][0]=V[r][c][1]
	N = []	# numerator
	D = []	# denominator
	for r in V:
		n = []
		d = []
		for c in r:
			if c is not None:
				if shiftby_1:
					c = np.pad(c, (1, 0), mode='constant')
				n.append(np.pad(c, (0, 1), mode='constant'))
				den = np.zeros(len(c)+1)
				den[0] = 1
				d.append( den )
			else:
				n.append([0])
				d.append([1])
		N.append(n)
		D.append(d)

	return control.tf2ss(control.tf(N, D, 1.))
	

def ABC_output(r_max, A, B, C, D=None, dtype=float):
	"""Output of sys, p x q x r_max"""
	V = np.zeros((C.shape[0],B.shape[1],r_max), dtype=dtype)
	if D is not None: V[:, :, 0] = D
	X = B
	for r in xrange(r_max):
		V[:, :, r] = np.dot(C, X)
		X = np.dot(A, X)
	return V

def SS_output(sys, r_max):
	"""Output of sys, p x q x r_max"""
	return ABC_output(r_max, sys.A, sys.B, sys.C, dtype=sys.D.dtype)
#	V = np.zeros(sys.D.shape + (r_max,))
#	V[:, :, 0] = sys.D
#	b = sys.B
#	for r in xrange(r_max):
#		V[:, :, r] = np.dot(sys.C, b)
#		b = np.dot(sys.A, b)
#	return V



def SS_plot(sys, r_max):
	from matplotlib import pyplot as plt
	leg = []
	V = SS_output(sys, r_max)
	for i in xrange(sys.D.shape[0]):
		for j in xrange(sys.D.shape[1]):
			plt.plot(V[i, j, :], '.-')
			leg.append( (i, j))
	plt.ylim([np.min(V), np.max(V)])
	plt.legend(leg)
	plt.show()


def SS_canon(sys, full_output=False):
	""" Brings a SS to modal canonical form: A is real and block diagonal, ideally with blocks of size < 2x2. However, if it would be very poorly condition to do so (due to Jordan blocks), returns larger blocks.
	
	"""
	#Bring to real Schur form
	A, B, C, U, wr, wi, info = more_slicot.tb01wd(sys.A, sys.B, sys.C)
	if info!=0:
		raise ValueError(info)
	
	#Bring to block-diagonal Schur form
	A, X, nblcks, blsize, wr, wi, info = more_slicot.mb03rd('U', 'S', 100., A, np.eye(A.shape[0], dtype=float), tol = 0.)
	if info!=0:
		raise ValueError(info)

	if anynan(X):
		print ("NaNs in X, resorting to backup canonicalization")
		return SS_eig_canon( control.StateSpace(A, B, C, sys.D, 1.))

	B = np.linalg.solve(X, B)
	C = np.dot(C, X)
	
	return control.StateSpace(A, B, C, sys.D, 1.)

def SS_eig_canon(sys):
	"""Uses non-symmetric eigendecomposition to block-diagonalize A.
	"""

	u, w = np.linalg.eig(sys.A)
	perm = np.lexsort( ( np.sign(np.imag(u)), np.abs(np.imag(u)), np.real(u), np.abs(u)) )
	u = u[perm]

	w = w[:, perm]
	
	""" u may have complex pairs. We now perform 2x2 rotation for conjugate pairs to make everything real.
	"""
	
	R = np.array( [[1, 1], [1j, -1j]])/np.sqrt(2.)
	Rinv = R.conj().T
	
	A = np.zeros(w.shape, dtype=float) #The resulting block-diagonal A
	
	i=0
	pairs = np.zeros(len(u), dtype=np.bool)
	while i < len(u):
		if np.abs(np.imag(u[i]))>5e-16:
			if np.abs( u[i] - np.conj(u[i+1])) > 5e-16:
				print ("Warning! Non-conjugate pair?:", u[i:i+2])
			w[:, i:i+2] = np.dot(w[:, i:i+2], Rinv).real
			ur, ui = u[i].real, u[i].imag
			A[i:i+2, i:i+2] = [ [ur, ui], [-ui, ur]]
			pairs[i] = True
			i+=2
		else:
			A[i, i] = u[i].real
			i+=1

	w = w.real
	
	cond = np.linalg.cond(w)
	if np.linalg.cond(w) > 1e+3:
		print( "Warning, cond(w):", cond)
		
	err = np.linalg.norm(np.linalg.solve(w, np.dot(sys.A, w)) - A)
	if err > 1e-12:
		print ("Warning, eigen decomposition has error of", err)
	if err > 1e-10:
		raise ValueError( err)
	
	"""
	print "BEGOR", np.linalg.norm(sys.B),np.linalg.norm(sys.C)
	C = np.dot(sys.C, w)
	B = np.linalg.solve(w, sys.B)
	print "After", np.linalg.norm(sys.B),np.linalg.norm(sys.C)
	"""
	i = 0
	while i < len(u):
		if pairs[i]:
			b_nrm = np.linalg.norm(B[i:i+2, :])
			c_nrm = np.linalg.norm(C[:, i:i+2])
			B[i:i+2, :] = B[i:i+2, :]*np.sqrt(c_nrm/b_nrm)
			C[:, i:i+2] = C[:, i:i+2]*np.sqrt(b_nrm/c_nrm)			
			i+=2
		else:
			b_nrm = np.linalg.norm(B[i:i+1, :])
			c_nrm = np.linalg.norm(C[:, i:i+1])
			B[i:i+1, :] = B[i:i+1, :]*np.sqrt(c_nrm/b_nrm)
			C[:, i:i+1] = C[:, i:i+1]*np.sqrt(b_nrm/c_nrm)
			i+=1

	return control.StateSpace(A, B, C, sys.D, 1.)

def SS_compress(sys, N_max=None, tol = 0., equil='S', job = 'B'):
	""" Compression of state machine. Returns SS of no more than N_max nodes (N_max is ignored if None - effectively set to current size of sys).
	
		If tol = 0., attempts an exact reduction.
		If tol > 0., cuts Hankel SVs at tol (effectively this will that the norm of the residual impulse is tol)
		If tol < 0., cuts Hankel SVs at tol*hsv[0] ( effectively this will imply that the relative error is tol)
	
		Returns sys, hsv
	"""
	dico = 'D' #discrete
	n = np.size(sys.A,0)
	m = np.size(sys.B,1)
	p = np.size(sys.C,0)		
	
	if tol >= 0.:
		Nr, Ar, Br, Cr, hsv_ret = slycot.ab09ad(dico,job,equil, n, m, p, sys.A,sys.B,sys.C, nr=N_max, tol=tol)
		return control.StateSpace(Ar, Br, Cr, sys.D, 1.), hsv_ret
	
	#First must find hsv so we know how to scale tol. Not sure there is another way to do this!	
	Nr, Ar, Br, Cr, hsv_ret = slycot.ab09ad(dico,job,equil, n, m, p, sys.A,sys.B,sys.C, nr=N_max, tol=1e-14)
				
	n = np.size(Ar,0)
	if N_max==None:
		N_max = n
	else:
		N_max = min(N_max, n)
	

	tol = -tol*hsv_ret[0]

	#Now try to realize system just based on 'tol'
	Nr, Ar, Br, Cr, hsv = slycot.ab09ad(dico,job,equil, n, m, p, Ar,Br,Cr, nr=None, tol = tol)
	if Nr <= N_max:
		return control.StateSpace(Ar, Br, Cr, sys.D, 1.), hsv_ret

	#Used too many nodes! Or didn't care about tol. Resort to the N_max
	Nr, Ar, Br, Cr, hsv = slycot.ab09ad(dico,job,equil, n, m, p, Ar,Br,Cr, nr=N_max, tol=0.)
	return control.StateSpace(Ar, Br, Cr, sys.D, 1.), hsv_ret


