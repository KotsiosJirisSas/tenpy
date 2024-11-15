import numpy as np
import itertools
import functools
import sys		# for the module command

from tenpy.linalg import np_conserved as npc
from tenpy.tools.string import joinstr
from scipy import linalg
"""Index conventions:


	2-site H.  For sites 1, 2:   ket1 ket2 bra1 bra2 : 'kL', 'kR', 'bL', 'bR'
	
		 MPO:	mpoL mpoR, ket, bra
				
"""

def set_var(pars, key, default, force_verbose=None):
	"""	Look up a value in the dictionary (pars).
			return it if it exists, return default otherwise.
			If 'verbose' key is set, then print this out.

		If pars['verbose'] is set, it'll decide whether to print the value
			force_verbose overrides that option
		
		Examples:
		>>>	from models.model import set_var
		>>>	self.L = set_var(pars, 'L', 1)
		>>>	self.JzCoupling = set_var(pars, 'Jz', 0)
		"""
	verbose = 0
	if 'verbose' in pars.keys(): 
		verbose = pars['verbose']
	if isinstance(force_verbose, int): 
		verbose = force_verbose
	if key in pars.keys():
		if verbose > 0:
			print ("\t", key, "=", pars[key])
		return pars[key]
	else:
		if verbose > 1:
			print ("\t", key, "=", default, "(default)")
		return default



def any_nonzero(model, key_list, verbose_msg = ""):
	"""	Check if anything in key_list is nonzero, returns True/False
		"""
	verbose = 0
	if hasattr(model, 'verbose'): verbose = model.verbose
	if verbose > 0 and verbose_msg != "": print ("\t" + verbose_msg)

	for key in key_list:
		if isinstance(key, str):
		##	Check if it's nonzero
			try:
				v = getattr(model, key)
			except AttributeError:
				continue
			try:
				iter(v)
				if np.any(v != 0):
					if verbose > 1: print( "\t\tmodel.%s = %s != 0" % (key, getattr(model, key)))
					return True
			except TypeError:
				if v != 0:
					if verbose > 1: print ("\t\tmodel.%s = %s != 0" % (key, getattr(model, key)))
					return True

		elif isinstance(key, tuple):
		##	All of them have to be the same
			if len(key) == 0: raise ValueError("empty tuple")
			for i in range(len(key)):
				if isinstance(key[i], str):
					try:
						val = getattr(model, key[i])
					except AttributeError:
						val = 0
				else:
					val = key[i]	# match value (instead of attribute)
				if i > 0:
					if np.any(prev_val != val):
						if verbose > 1: print( "\t\t", key[i], "and", key[i-1], "does not match.")
						return True
				prev_val = val

		else:
			raise ValueError( "No idea what %s is." % key)

	##	Exhaust all the keys
	return False



""" translate_Q1:

	is assumed to return COPY

"""
def create_translate_Q1(tQ_dat):
	"""Set translate_Q1 from translate_Q1 data.
	
	tQ_data is a dictionary with the following keys:
		'module'
		'cls'
		'tQ_func'
		'keywords' (optional)

	This function will return a function, with parameters (Qind, nsites).

	Example ('self' could be a model or an mps object):
	>>>	self.translate_Q1 = create_translate_Q1(self.translate_Q1_data)
		"""
	if tQ_dat is None:
		return None
	print(tQ_dat)
	if tQ_dat['module'][:6+7] != 'tenpy.models.': #CHANGED FROM models to tenpy.models
		raise ValueError
	__import__(tQ_dat['module'])
	tQ_mod = sys.modules[tQ_dat['module']]
	tQ_cls = getattr(tQ_mod, tQ_dat['cls'])
	tQ_func = getattr(tQ_cls, tQ_dat['tQ_func'])
	if 'keywords' in tQ_dat:
		#print 'translate_Q1 = %s, keywords = %s' % (tQ_func, tQ_dat['keywords'])
		return functools.partial(tQ_func, **tQ_dat['keywords'])
	else:
		#print 'translate_Q1 =', tQ_func
		return tQ_func


def identical_translate_Q1_data(tQdat1, tQdat2, verbose=0):
	"""Check if tQdat1 matches tQdat2.  Returns True/False.	"""
	if tQdat1 is None and tQdat2 is None: return True
	if tQdat1['cls'] != tQdat2['cls'] or tQdat1['module'] != tQdat2['module'] or tQdat1['tQ_func'] != tQdat2['tQ_func']:
		if verbose > 0: print ('translate_Q1_data mismatch:\n\t{}\n\t{}\n'.format(tQdat1, tQdat2))
		return False
	if np.any(tQdat1['keywords']['Kvec'] != tQdat2['keywords']['Kvec']):
		if verbose > 0: print ('translate_Q1_data mismatch:\n\t{}\n\t{}\n'.format(tQdat1['keywords']['Kvec'], tQdat2['keywords']['Kvec']))
		return False
	return True

		

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class model(object):
	"""	Template model class.  All other models should inherit from this.

	TODO, make a list of required attributes:
	Attributes:
		L
		d
		grouped
		add_conj
		bc
		translate_Q1_data
		Qp_flat:
		H_mpo ???
		Qmpo_flat
		chi
		"""

	def __init__(self):
		"""	This should be explicitly invoked for each model which derives from it.  For example:
			>>>	super(this_model_name_right_here, self).__init__()
			"""
		self.L = 1
		self.H_mpo = None	# list of MPO ops
		self.H = None	# list of 2-site ops
		self.grouped = 1
		self.add_conj = False 		# If True, does H = (H_mpo + H_mpo^D)/2
		self.bc = 'periodic'	# either 'periodic' or 'finite'
		self.translate_Q1_data = None
		self.dtype = float
		self.verbose = 0
		self.d = None
		
	def translate_Q(self, Q, r):
		if self.translate_Q1 is not None:
			return self.translate_Q1(Q, self.grouped*r)
		return Q.copy()
			
	def check_sanity(self):
	
		if self.H_mpo is not None:
			if len(self.H_mpo)!=self.L:
				raise ValueError
			for h in self.H_mpo:
				h.check_sanity()
				
			for i in range(self.L):
				npc.tensordot_compat(self.getH_mpo(i), self.getH_mpo(i + 1), [[1], [0]] )
				
		if self.H is not None:
			if len(self.H)!=self.L:
				raise ValueError
			for h in self.H_mpo:
				h.check_sanity()
		# This need a lot more checks

	def save_hdf5(self, hdf5_saver, h5gr, subpath):
		"""Export `self` into a HDF5 file.
		
		This method saves the data it needs to **roughly** reconstruct `self` with :meth:`from_hdf5`.
		
		Specifically, it saves the attributes `H_mpo`, `H`, `add_conj`, `bc`, `vL`, `vR`, `translate_Q1_data`,
		`states` as datasets, and `site_ops`, `bond_ops` as dictionaries containing all the operators,
		and `grouped`, `L` as attributes.
		It does *not* save other attributes, in particular it doesn't save the parameters.
		
		It is expected that this function is called after `init_model_for_conservation`.
		
		Parameters
		----------
		hdf5_saver : :class:`~tools.hdf5_io.Hdf5Saver`
		    Instance of the saving engine.
		h5gr : :class`Group`
		    HDF5 group which is supposed to represent `self`.
		subpath : str
		    The `name` of `h5gr` with a ``'/'`` in the end.
		"""
		# save everything defined in __init__
		hdf5_saver.save(self.H_mpo, subpath + "H_mpo")
		hdf5_saver.save(self.H, subpath + "H")
		hdf5_saver.save(self.add_conj, subpath + "add_conj")
		hdf5_saver.save(self.bc, subpath + "boundary_condition")
		hdf5_saver.save(self.vL, subpath + "index_identity_left")
		hdf5_saver.save(self.vR, subpath + "index_identity_right")
		hdf5_saver.save(self.translate_Q1_data, subpath + "translate_Q1_data")
		hdf5_saver.save(self.states, subpath + "states")  # [{name:index} for each site]
		onsite_ops = dict([(name, getattr(self, name)) for name in getattr(self, "site_ops", [])])
		bond_ops = dict([(name, getattr(self, name)) for name in getattr(self, "bond_ops", [])])
		# onsite_ops, bond_ops: {opname: [op for each site/bond]}
		hdf5_saver.save(onsite_ops, subpath + "onsite_operators")
		hdf5_saver.save(bond_ops, subpath + "bond_operators")
		h5gr.attrs["grouped"] = self.grouped
		h5gr.attrs["L"] = self.L
		hdf5_saver.save(self.d, subpath + "dimensions")


	@classmethod
	def from_hdf5(cls, hdf5_loader, h5gr, subpath):
		"""Load instance from a HDF5 file.
		
		This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.
		
		Parameters
		----------
		hdf5_loader : :class:`~tools.hdf5_io.Hdf5Loader`
		    Instance of the loading engine.
		h5gr : :class:`Group`
		    HDF5 group which is represent the object to be constructed.
		subpath : str
		    The `name` of `h5gr` with a ``'/'`` in the end.
		
		Returns
		-------
		obj : cls
		    Newly generated class instance containing the required data.
		"""
		obj = cls.__new__(cls)  # create class instance, no __init__() call
		hdf5_loader.memorize_load(h5gr, obj)

		obj.H_mpo = hdf5_loader.load(subpath + "H_mpo")
		obj.H = hdf5_loader.load(subpath + "H")
		obj.add_conj = hdf5_loader.load(subpath + "add_conj")
		obj.bc = hdf5_loader.load(subpath + "boundary_condition")
		obj.vL = hdf5_loader.load(subpath + "index_identity_left")
		obj.vR = hdf5_loader.load(subpath + "index_identity_right")

		obj.states = hdf5_loader.load(subpath + "states")
		site_ops = hdf5_loader.load(subpath + "onsite_operators")
		bond_ops = hdf5_loader.load(subpath + "bond_operators")
		for name, ops in site_ops.items():
			setattr(obj, name, ops)
		for name, ops in bond_ops.items():
			setattr(obj, name, ops)
		obj.site_ops = list(site_ops.keys())
		obj.bond_ops = list(bond_ops.keys())

		obj.grouped = hdf5_loader.get_attr(h5gr, "grouped")
		obj.L = hdf5_loader.get_attr(h5gr, "L")
		obj.d = hdf5_loader.load(subpath + "dimensions")

		obj.translate_Q1_data = hdf5_loader.load(subpath + "translate_Q1_data")
		obj.translate_Q1 = create_translate_Q1(obj.translate_Q1_data)
		obj.fracture_mpo = False
		obj.mod_q = obj.H_mpo[0].mod_q
		obj.dtype = obj.H_mpo[0].dtype
		obj.check_sanity()
		return obj
		
		
	def set_translate_Q1(self):
		if hasattr(self, 'translate_Q1_data'):
			self.translate_Q1 = create_translate_Q1(self.translate_Q1_data)
		else:
			self.translate_Q1 = None

	
	def increase_L(self, new_L):
		"""Increases size of unit cell by tiling model. If new_L is not a multiple of L, the remainder is ignored.
		
			Note: does not re-sort 'q_ind'
		"""
		
		#MPZ: default behavior is to ignore the remainder.
		rep = new_L / self.L
		if rep == 0:
			raise ValueError( "I'm shrriiiinkking. not allowed. =(")
		if rep*self.L < new_L:
			print ("Warning, new_L was not multiple of L, the remainder will be ignored giving new_L = ", rep*self.L)
		
		L = self.L			
		self.L = self.L*rep
		self.d = np.tile(self.d, rep)
		
		if self.H_mpo is not None:
			self.H_mpo = [ self.H_mpo[i%L].copy() for i in range(self.L) ]
			self.chi = np.tile(self.chi, rep)
			self.vL = np.tile(self.vL, rep)
			self.vR = np.tile(self.vR, rep)
			
		if self.H is not None:
			self.H = [ self.H[i%L].copy() for i in range(self.L) ]
			self.calc_wv()
		
		self.Qp_flat = [ self.Qp_flat[i%L].copy() for i in range(self.L) ]
		self.Qp_ind = [ self.Qp_ind[i%L].copy() for i in range(self.L) ]
		
		if self.translate_Q is not None:
			func = self.translate_Q
			for i in range(L, self.L):
				if self.H_mpo is not None:
					self.H_mpo[i].imap_Q(func, i - i%L)
				if self.H is not None:
					self.H[i].imap_Q(func, i - i%L)
	
				self.Qp_flat[i] = self.translate_Q( self.Qp_flat[i], i - i%L)
				self.Qp_ind[i] = self.translate_Q( self.Qp_ind[i], i - i%L)
		

	def getH_mpo(self, i, copy=False):
		"""	Returns VIEW of H, except that q_ind is copied and translated as necessary.
		Hence modifying the data of the H returned will modify self.H.
		However, if the data is not changed, does not need to be matched with corresponding setB in order to leave self.H unchanged.
			"""
		if self.translate_Q is None or (i - i%self.L)==0:
			Hr =  self.H_mpo[i%self.L]
		else:
			Hr =  self.H_mpo[i%self.L].shallow_map_Q(self.translate_Q, i - i%self.L)
			
		if copy:
			return Hr.copy()
		else:
			return Hr

	def getW(self, i, copy=False):
		return self.getH_mpo(i, copy=copy)
		
	## ============================== TODO ==============================
		

	def setH_mpo(self, i, val, copy=False):
		"""Like above, sets self.B using view of val """
		if copy:
			val = val.copy()
			
		if self.translate_Q is None or (i - i%self.L)==0:
			self.H_mpo[i%self.L] = val
		else:
			self.H_mpo[i%self.L] = val.shallow_map_Q( self.translate_Q, i%self.L - i)
	

	def get_Qp_flat(self, site):
		if self.grouped > 1: raise NotImplementedError
		Q = self.Qp_flat[site % self.L]
		if self.translate_Q1 is not None:
			return self.translate_Q1(Q, site - (site % self.L))
		else:
			return Q


	def get(self, what, i, copy=False):
		"""Given either a list, or the name of an attribute, 'what', 		
		"""
	
		if type(what) == str:
			items = getattr(self, what)
		elif type(what)==list:
			items = what
		else:
			raise ValueError
			
		L = len(items)
				
		if copy:
			item = items[i%L].copy()
		else:
			item = items[i%L]
		
		if self.translate_Q is None or (i - i%self.L)==0:
			return item
		else:
			return item.shallow_map_Q( self.translate_Q, i - i%L)
		

	def set(self, what, i, val):
		
		if type(what) == str:
			items = getattr(self, what)
		elif type(what)==list:
			items = what	
		else:
			raise ValueError
			
		L = len(items)	

		if self.translate_Q is None or (i - i%self.L)==0:
			items[i%L] = val
		else:
			items[i%L] = val.shallow_map_Q( self.translate_Q, i%L - i)
		


	def twocellH_to_H_mpo(self):
		"""Constructs the MPO from the two-cell Hamiltonian.

		Overview
		--------------------
		Required attributes:
			L:  the size of the (mpo) unit cell (int)
			d:  the Hilbert space dimension per site (int)
			H:	 a list (length L) of twocell np.array's shaped (d,d,d,d) with indices corresponding to iL,iR,iL*,iR*
			num_q:  the number of charges [int]
			mod_q:
			Qp_flat:  a list (len L) of charges in flat form, each np.arrays shaped (d,num_q)
		Optional attributes:
			verbose:
			twocellH_svd_cutoff:
			fracture_mpo:

		Attributes set by this function:
			H_mpo		# as an np.array
			vL, vR
			chi
			Qmpo_flat

		One needs to call init_model_for_conservation() afterwards to make everything in to npc.array's

			"""
		#	TODO, currently assumes all the Qp_flats are the same
		#	TODO, check fracture_mpo, and determine how to use the perm below

		L = self.L		# unit cell size
		d = self.d		# Hilbert space dimension of a single site
		verbose = self.verbose
		#svd_cutoff = 1e-13
		#if hasattr(self, "twocellH_svd_cutoff"): svd_cutoff = self.twocellH_svd_cutoff
		svd_cutoff = getattr(self, "twocellH_svd_cutoff", 1e-13)
		Qp_flat = self.Qp_flat[0]		# All the Qp_flats are the same
		Id = np.eye(d, dtype = int)
	##	Check inputs (not an exhaustive check)
		if len(self.H) != L: raise ValueError
		for b in range(L):
			if self.H[b].shape != (d,d,d,d): raise ValueError("self.H[%s] has an awful shape %s != (%s, %s, %s, %s)" % (b, self.H[b].shape, d, d, d, d))

	##	Do svd decomposition on the two cell H, into X,Y,Z's
	##		decompose H = sum_j X_jL * y_j * Z_jR
		XYZ_list = [None] * L	# tuples of X,Y,Z for each bond
		chi = [None] * L		# labelled by bonds
	##	To minimize the MPO chi, we first take out any component of the form Id * A_R on A_L * Id
	##		The A_L / A_R's are the one_site_op's
		one_site_op = [ np.zeros((d,d), self.dtype) for i in range(L) ]
		for b in range(L):
		##	Convert to npc array
			# print 'entry H', self.H[b]

			perm, npc_H = npc.array.from_ndarray_flat(self.H[b], [Qp_flat] * 4, q_conj = [1, 1, -1, -1], mod_q = self.mod_q, sort = False)
			#npc_H_norm_diff 
			npc_H_norm = npc_H.norm(); H_norm = np.linalg.norm(self.H[b].reshape(-1))
			if abs(npc_H_norm - H_norm) > 1e-13: print( "models.twocellH_to_H_mpo() warning!  [bond {}] npc_H_norm ({}) != H_norm ({}), difference = {}".format(b, npc_H_norm, H_norm, npc_H_norm-H_norm))
			#print joinstr(npc_H.to_ndarray())
			# exit(0)
			### TODO, assumes that perm is [[0,1,....,d-1]] * 4
			pipe = npc_H.make_pipe([0, 2])
			XYZ = npc_H.transpose((0,2,1,3)).combine_legs([[0,1], [2,3]], [pipe, pipe])
			# print 'XYZ be4 subtraction \n ', XYZ
			#XYZ.check_sanity()

		##	Project out parts with Id in it
		##		First create the identity matrix (in npc form, with legs combined)
			perm, npc_Id = npc.array.from_ndarray_flat(Id, [Qp_flat] * 2, q_conj = [1, -1], mod_q = self.mod_q)
			npc_vec_Id = npc_Id.combine_legs([[0,1]], [pipe])
			if len(npc_vec_Id.q_dat) > 1: raise RuntimeError("Charge of identity operator is not unique, something is very wrong.")
			q0_i = npc_vec_Id.q_dat[0,0]		# q-index of the zero charge sector
			vec_Id = npc_vec_Id.dat[0]			# the norm of vec_Id should be sqrt(d)
			#print "zero charge sector q-index:", q0_i

		##		Find the zero charge sector in XYZ, since the one_site_op's must have zero-charge
			if len(np.where(XYZ.q_dat == q0_i)[0]) > 0:
				XYZ_q0 = XYZ.dat[np.where(XYZ.q_dat == q0_i)[0][0]]
			##	Subtract out from H
				right_op = np.dot(vec_Id, XYZ_q0) / d		# there's a Id (x) right_op piece in the H that will be dealt with separately
				XYZ_q0 -= np.outer(vec_Id, right_op)
				left_op = np.dot(XYZ_q0, vec_Id) / d		# there's a left_op (x) Id (x) piece in the H that will be dealt with separately
				XYZ_q0 -= np.outer(left_op, vec_Id)
			##	Add to the one_site_op's
				one_site_op[b] += pipe.t_to_mul_ndarray(left_op, qindex=q0_i)
				one_site_op[(b+1) % L] += pipe.t_to_mul_ndarray(right_op, qindex=q0_i)

		##	SVD the stripped down two cell H
			# print 'bond, XYZ \n',XYZ
			try:
				X, Y, Z = npc.svd(XYZ, full_matrices=0, cutoff=svd_cutoff)
				if self.verbose > 1:
					print ("\tsingular values for H[bond=" + str(b) + "] :", Y)
				chi[b] = 2 + npc.shape_from_q_ind(X.q_ind[1])
				XYZ_list[b] = (X, Y, Z)
			except RuntimeError as XXX:
				class DummyX:
					def __init__(self):
						self.q_ind = [0, []]
				if self.verbose > 1:
					print ("\tsingular values for H[bond=" + str(b) + "] :", [])
				chi[b] = 2
				XYZ_list[b] = (DummyX(), None, None)

		if self.verbose > 0:
			print ("\tmpo chi:", chi)

	##	Now that we know all the chi's and X,Y,Z's, we construct the MPO in flat form
	##		H_mpo labelled by sites
		H_mpo = [ np.zeros((chi[(s-1)%L], chi[s], d, d), self.dtype) for s in range(L) ]
		self.Qmpo_flat = [ np.zeros((chi[b],self.num_q), int) for b in range(L) ]
		self.vL = [0] * L
		self.vR = [ chi[s] - 1 for s in range(L) ]
		for b in range(L):
			sL = b
			sR = (b + 1) % L
		##	Plop the one site ops
			H_mpo[sL][0, 0, :, :] = Id
			H_mpo[sL][-1, -1, :, :] = Id
			H_mpo[sL][0, -1, :, :] = one_site_op[sL]

		##	Two sites ops
			X, Y, Z = XYZ_list[b]
			num_qs = len(X.q_ind[1])			# number of charge sectors
			for qi in range(num_qs):
				# np.where returns a tuple of arrays
				Xr = np.where(X.q_dat[:, 1] == qi)[0][0]
				Zr = np.where(Z.q_dat[:, 0] == qi)[0][0]
				for i in range(X.q_ind[1][qi,1] - X.q_ind[1][qi,0]):
					k = X.q_ind[1][qi,0] + i
					#print qi, X.q_ind[1][qi,0] + i, Xr, Zr
					#print X.dat[Xr][:, i], X.q_dat[Xr,:], Z.q_dat[Zr,:]
					#print pipe.t_to_mul_ndarray(X.dat[Xr][:, i], qindex=X.q_dat[Xr,0])
					#print pipe.t_to_mul_ndarray(Z.dat[Zr][i, :], qindex=Z.q_dat[Zr,1])
					H_mpo[sL][0, k+1, :, :] = pipe.t_to_mul_ndarray(X.dat[Xr][:, i], qindex=X.q_dat[Xr,0]) * np.sqrt(Y[k])
					H_mpo[sR][k+1, -1 :, :] = pipe.t_to_mul_ndarray(Z.dat[Zr][i, :], qindex=Z.q_dat[Zr,1]) * np.sqrt(Y[k])
					self.Qmpo_flat[b][k+1, :] = npc.mod_onetoinf(-X.q_ind[1][qi, 2:], self.mod_q)
					#self.Qmpo_flat[b][k+1, :] = -Z.q_ind[0][qi, 2:]
			self.Qmpo_flat[b] = npc.mod_onetoinf(self.Qmpo_flat[b], self.mod_q)
			#print X.to_ndarray()
			#print Y
			#print Z.to_ndarray()
			#print
		self.chi = chi
		self.H_mpo = H_mpo



	def calculate_nsite_h(self, num):
		"""Uses H_mpo to calculate a n-site Hamiltonian.
			num = (i, i+2) will NOT give correct 2-site bond operators because the on-site terms will be double counted.
		
		Returns h in form [d, d,  ... , d, d, ...]
		(for example - two site case is identical to specification in bond_exp )
		
		
		if num is a number, returns h for sites range(0, num)
		if num is a tuple, returns h for sites range(num[0], num[1])
		"""
		
		if type(num)==tuple:
			sites = np.arange(num[0], num[1])
			num = num[1] - num[0]
		else:
			sites = np.arange(num)
	
		L = self.L

		#First calculate right-most mpo, which has right aux bond set to vR 
		
		a = sites[-1]
		Ht = self.getH_mpo(a).copy()
		
		takeR = np.zeros( Ht.shape[1], dtype = np.bool)
		takeR[ self.vR[a%L] ] = True
		Ht.iproject(takeR, 1)#
		Ht.isqueeze(axis = 1)
		#Now attach additional MPOs
		for a in sites[-2:0:-1]:
			Ht = npc.tensordot(self.getH_mpo(a), Ht, axes = (1, 0))
		#Finally calculate left-most MPO, and squeeze result.
		a = sites[0]
		takeL = np.zeros( self.H_mpo[a%L].shape[0], dtype = np.bool)
		
		takeL[ self.vL[(a-1)%L] ] = True
		if len(sites) == 1:
			Ht = Ht.iproject(takeL, 0)#
			Ht.isqueeze(axis = 0)
		else:
		
			Hl = self.getH_mpo(a).copy()
			Hl.iproject(takeL, 0)#
			Hl.isqueeze(axis = 0)
			npc.tensordot_compat(Hl, Ht, axes = [[0], [0]])
			Ht = npc.tensordot(Hl, Ht, axes = [[0], [0]])
			#Transpose to give correct row/column structure
			perm = range(0, 2*num, 2)
			perm.extend(range(1,2*num, 2))
			Ht.itranspose(perm)
		return Ht
		
	def eigensystem(self, L, k = 1):
		"""Calculates, in a dumb (dense) way, the first 'k' eigenvalues of H for L sites, using the MPO to build H.
		"""
		H = self.calculate_nsite_h(L)

		H = H.icombine_legs( [ range(0, L), range(L, 2*L)])

		u, v =  npc.eigh(H)
		print ("Spectrum", np.sort(u)[0:k])
		return u,v 
	
	def calc_H_from_mpo(self):
		"""Calculate two-site gates from MPO"""
		if self.bc == 'segment':
			raise NotImplemented
		
		H1 = [ self.calculate_nsite_h((j, j+1)) for j in range(self.L)]

		H2 = [ self.calculate_nsite_h((j, j+2)) for j in range(self.L - (self.bc=='finite')) ]

			
		for j in range(self.L - 2*(self.bc=='finite')):
			H2[j] = H2[j] - npc.tensordot( self.Id[j], H1[j+1], axes = [[], []]).itranspose((0, 2, 1, 3))/2.

		for j in range(self.bc=='finite', self.L - (self.bc=='finite')):
			H2[j] = H2[j] - npc.tensordot( H1[j], self.Id[j+1], axes = [[], []]).itranspose((0, 2, 1, 3))/2.
			
		if self.bc=='finite':
			H2.append(None)
		return H2
	
	def calc_wv(self):
		"""Diagonalize two-site H for use when calculating 2-site unitaries.
			In addition, caches pipe to view dxdxdxd H as (d^2)x(d^2) H.
				
	Requires:
		self.H
	Sets:
		self.w_bond
		self.v_bond
		self.bond_pipe
		self.diagonal
		"""
		
		self.w_bond = []
		self.v_bond = []
		self.bond_pipe = []
		self.diagonal = [True for ii in range(len(self.H))]
		nhflag = 0
		ii =-1
		
		
		for h in self.H:
			ii += 1
			pipe = h.make_pipe([0, 1])		# pipe to view dxd x dxd H as d^2 x d^2 H

			#hmx = h.to_ndarray().reshape(self.d**2,self.d**2)
			#if np.allclose(hmx,np.transpose(np.conj(hmx))) : # how to do this in npc
			if npc.norm(h - h.conj().transpose([2, 3, 0, 1]))< 1e-15:
				w,v = npc.eigh(h.combine_legs([[0, 1], [2, 3]], pipes = [pipe, pipe])) #Reshape and diagonalize
			else:
				nhflag += 1
				if nhflag == 1 : print( 'Non-Hermitian Hamiltonian, will give non-Unitary (real-)time evolution')
				w,v = npc.eig(h.combine_legs([[0, 1], [2, 3]], pipes = [pipe, pipe])) #Reshape and diagonalize
			
			# check if the diagonalisation worked.			
			Hnp = h.combine_legs([[0, 1], [2, 3]], pipes = [pipe, pipe]).to_ndarray()
			vnp = v.to_ndarray()
			Hp = np.dot(np.dot(vnp,np.diag(w)),np.conj(vnp.T))

			if np.allclose(Hnp,Hp) != True:
				print( 'Diagonalisation of 2-site H did did not work for bond ', ii)
				self.diagonal[ii]  = False
				
							
			self.w_bond.append(w)  #The eigensystem
			self.v_bond.append(v)
			self.bond_pipe.append(pipe) #Save the pipes, because we will need to reshape unitarity
			
	def calc_u(self, delta_t, TrotterOrder, type, E_offset = [] ):
		"""Calculates Suzuki-Trotter decomposition"""
		#ASSUMES calc_wv has been called

		self.Uinfo = {'delta_t':delta_t, 'TrotterOrder':TrotterOrder, 'type': type, 'E_offset':E_offset}
		# subtract the t=0 GS energy for the real time evolution of correlation functions
		dim = len(self.w_bond[0])		
		offsetlist = np.zeros((self.L,dim))
		if len(E_offset) != 0:
			for i_bond in range(self.L):
				offsetlist[i_bond,:] = [E_offset[i_bond] for iii in range(dim)]



		def Ubond_np_expm(h,tstep):
			""" create 2-site time-evolution gate by exponentiating the Hamiltonian """	
			pipe = h.make_pipe([0, 1])		# pipe to view dxd x dxd H as d^2 x d^2 H
			Hnpc  =  h.combine_legs([[0, 1], [2, 3]], pipes = [pipe, pipe])
			Anpc = -1j*tstep*Hnpc
			Anp  = Anpc.to_ndarray() 
			# Usp3 = linalg.expm3(Anp) # Taylor expansion
			Usp = linalg.expm(Anp)   # scaling + Pade ap.
			UU = npc.array.from_ndarray_trivial(Usp)
			
			return UU

		# check if there is any 2-site H that could not be diagonalised
		diagonal_flag = True

		for i_bond in range(self.L):
			if self.diagonal[i_bond] == False:
				diagonal_flag = False
				print ('2-site H at bond ', i_bond ,' could not be diagonalised')
				print ('the U gate will be obtained by matrix exponentiation from numpy')
				print( 'symmetry conservation must be turned off (?)')
				print( 'there is probably a connection between non-diagonalisable H2 and the non-conservation of symmetries ')
				if self.num_q != 0 :
					print( 'is this case even physical ? Some symmetry is conserved, but the 2-site H was not diagonalisable ....\n do matrix exponential for npc matrices. work in progress...')
					print( 'below is a sample code to do Taylor expansion on npc matrices (not working)')
					exit(0)
					#  Taylor expansion, that perserves conserved charges:
					maxiter = 10
					kk = 0
					expAk = npc.eye_like(Anpc)
					expA = expAk
					for kk in range(1,maxiter):
						expAk = npc.tensordot( Anpc,Anpc, axes = [[1],[1]] )
						for ki in range (kk-1):
							print (kk,ki)
							expAk = npc.tensordot( expAk,Anpc, axes = [[1],[1]] )
						expAk *= 1./(np.math.factorial(kk))						
						expAknp = expAk.to_ndarray()
						print (np.allclose(Usp,expAknp))
						print ('Usp = \n', Usp)
						print ('Hnpc = \n', expAknp)
					# and then reattach the pipes ?	
				

		if diagonal_flag == True :

			if (TrotterOrder == 1):
				self.U = [[None]*self.L]
				for i_bond in range(self.L):
					dt = delta_t
				
					if self.diagonal[i_bond] == True:
						if (type == 'IMAGINARY_TIME'):		
							s = np.exp(-dt*self.w_bond[i_bond])
						elif (type == 'REAL_TIME'):
							s = np.exp(-1j*dt*( self.w_bond[i_bond] - offsetlist[i_bond] ))
						else:
							raise ValueError('need to have either real_time or imaginary_time')  
						#U = V s V^dag, s = e^(- tau E )
						Ubond = self.v_bond[i_bond].scale_axis(s, axis = 1)
						Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])				
						self.U[0][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

			elif (TrotterOrder == 2):
				self.U = [[None]*self.L, [None]*self.L]
				for i_bond in range(self.L):
					dt = delta_t/2.
					if (type == 'IMAGINARY_TIME'): 
						s = np.exp(-dt*self.w_bond[i_bond])
					elif (type == 'REAL_TIME'):
						s = np.exp(-1j*dt*( self.w_bond[i_bond] - offsetlist[i_bond] ))
					else:        
						raise ValueError('need to have either real_time or imaginary_time')  

					#U = V s V^dag, s = e^(- tau E )
					Ubond = self.v_bond[i_bond].scale_axis(s, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[0][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

					s = s*s
					#U = V s V^dag, s = e^(- tau E )
					Ubond = self.v_bond[i_bond].scale_axis(s, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[1][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U
					

			elif (TrotterOrder == 4):
				# adapted from Schollwock2011 notation ...
				# U operators in the following order :
				# a  : exp( Hodd t1 / 2   )
				# 2a : exp( Heven t1    )
				# b  : exp(  Hodd t1    )
				# c  : exp(  Hodd (t-3*t1)/2  )
				# d  : exp(  Hodd t3  )
				# 2a and b use the same slot!
			
				self.U = [	[None]*self.L	, [None]*self.L, [None]*self.L, [None]*self.L ]

				for i_bond in range(self.L):
					dt1 = 1. / (4. - 4.**(1/3) ) * delta_t /2.
					dt3 = delta_t - 4* (dt1*2)
				
					if (type == 'IMAGINARY_TIME'): 
						s1 =  np.exp(-dt1*self.w_bond[i_bond])
						s13 = np.exp(- ( dt3 + 2*dt1 )/2.  *self.w_bond[i_bond])
						s3 =  np.exp(-dt3*self.w_bond[i_bond])

					elif (type == 'REAL_TIME'):
						s1 =  np.exp(-1j*dt1*( self.w_bond[i_bond] - offsetlist[i_bond] ))
						s13 = np.exp(-1j*( dt3 + 2*dt1 )/2.*( self.w_bond[i_bond] - offsetlist[i_bond] ))
						s3 =  np.exp(-1j*dt3*( self.w_bond[i_bond] - offsetlist[i_bond] ))
					
					else:        
						raise ValueError('need to have either real_time or imaginary_time')  
		
					#U = V s V^dag, s = e^(- tau E )
				
					# a
					Ubond = self.v_bond[i_bond].scale_axis(s1, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[0][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

					# 2a
					s1 = s1*s1
					Ubond = self.v_bond[i_bond].scale_axis(s1, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[1][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

					# b self.U[2][i_bond] = copy.deepcopy(self.U[1][i_bond])

					# c
					Ubond = self.v_bond[i_bond].scale_axis(s13, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[2][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

					#d
					s3 = s3
					Ubond = self.v_bond[i_bond].scale_axis(s3, axis = 1)
					Ubond = npc.tensordot(Ubond, self.v_bond[i_bond].conj(), axes = [[1], [1]])      
					self.U[3][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U


			else:      
				raise NotImplementedError('Only 4th order Trotter has been implemented')
		# ------
		else:
			if (TrotterOrder == 1):
				self.U = [[None]*self.L]
				for i_bond in range(self.L):
					h = self.H[i_bond]
					Ubond = Ubond_np_expm(h,delta_t)
					self.U[0][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U

			elif (TrotterOrder == 2):
				self.U = [[None]*self.L, [None]*self.L]

				for i_bond in range(self.L):
					h = self.H[i_bond]										
					
					dt = delta_t/2.
					Ubond = Ubond_np_expm(h,dt)			
					self.U[0][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U
			
					dt = delta_t					
					Ubond = Ubond_np_expm(h,dt)			
					self.U[1][i_bond] = Ubond.split_legs([0, 1], leg_pipes = [self.bond_pipe[i_bond]]*2 ) #Reshape U
			else:      
				raise NotImplementedError('Only 2nd order Trotter has been implemented for the direct exponentiation')


	################################################################################
	################################################################################
	##	Convention for MPOgraph
	##
	##	For each site 'i', G[i] encodes that transition  r --> c places some operator Op at site i. It is defined as follows:
	##	Each G[i] is a dictionary whose keys are rows, with values denoting all columns transitioning from the row. These values are themselves dictionaries, whose keys are the columns. The value of the column key is a list of tuples of the form (OpName, OpStrength), which will be added in superposition:
	##		r : { c: [ (OpName1, OpStrength1), (OpName2, OpStrength2) ... ], ... }
	##
	##	Notes:
	##	* r lives in bond i-1 (by Frank's convention) but in G[i] by MPOgraph convention.
	##	* c lives in the following bond.
	##	* all the nodes (r,c,..etc.) should be labelled with tuples, strings or ints (hashable types)
	##	* Ready 'R' and finish 'F' are reserved.
	##	
	##	Here for example is the Transverse-Field-Ising MPO (at ferromagnetic/paramagnetic critical point):
	##			(Sx, Sy, Sz are the spin-operators = Pauli operators/2)
	##		G[0] = { 'R' : { 'R': [('Id',1)] 'F': [('Sx',-2) ], 'ZZ': [('Sz', 2)] },
	##		         'ZZ' : { 'F': [('Sz',2)] },
	##		         'F' : { 'F' [('Id',1)] }
	##		       }

	#TODO : document the convention
	#TODO : document the functions below, what needs to be set for them to work?
	def empty_MPOgraph(self, L=None):
		"""	Return an empty MPOgraph with L sites. """
		if L is None: L = self.L
		return [ { 'R':{ 'R':[('Id',1.)] }, 'F':{ 'F':[('Id',1.)] } } for s in range(L) ]



	def print_MPOgraph(self, MPOgraph=None, multiline=False):
		if MPOgraph is None: MPOgraph = self.MPOgraph
		N = len(MPOgraph)
		dest_delim = ',\n\t\t' if multiline else ',  '
		
		print( "==================== MPO graph ====================")
		for s in range(N):
			for k,v in sorted(MPOgraph[s].items(), key=lambda x:x[0]):
				print ("  %s!%s  -->  " % (s, k)) + dest_delim.join([ '%s!%s: ' % ((s+1)%N,target) \
					+ ' + '.join([ '%s*%s' % (tg[1],tg[0]) for tg in v[target] ]) for target in v ])
		if multiline: print( "===================================================")

	def build_H_mpo_from_MPOgraph(self, G, verbose = 0):
		"""	Builds W-matrices from an abstracted representation 'G' of the MPO.  """
		
		#print(G[0].keys())
		#print(G[0][('Mk', 'AL-5-aL.9', 26)].keys())
		#quit()
		#print(len(G))
		#print(G[0][('Mk', 'aL-6-AL.12', 22)])
		#print(G[0][('Mk', 'aL-6-AL.12', 22)][('_a', np.int64(1), np.int64(12))])
		#quit()
		if verbose >= 1: 
			print( "Building H_mpo (W) from MPO graph...")
		L = self.L
		
		try:
			iter(self.d)
			d = self.d
		except:
			d = [self.d]		
		try:
			site_names = self.site_ops
		except:
			site_names = []
		if self.dtype == np.int64:
			raise ValueError(self.dtype)

		for n in site_names:
			#if n=='Id':
			#	print(n)
			op = getattr(self, n)			
			#if n=='Id':
			#	print(op)
			#INDEED ATTACHES NPC ARRAYS TO THIS STUFF
			#WHY DOES IT THEN ATTACH list to H_mpo?
			if type(op)!=list:
				#print(op)
				op = [op]
				
			setattr(self, n, op)
		#print(self.Id[0])
		self.chi = chi = np.array([len(G[(i+1)%L]) for i in range(L)], dtype = np.int64)
		
		#OK SETS SIZES OF MATRICES CORRESPONDING TO LAYER ETC
		self.W = W = [ np.zeros( (chi[i-1], chi[i], d[i%len(d)], d[i%len(d)] ), dtype = self.dtype) for i in range(L)]
		self.W=W= [[[None for _ in range(chi[i-1])] for _ in range(chi[i])] for i in range(L)]

		
		##	MPO_indices is a dictionary for each bond; ind[key] = index position
		##	MPO_labels is a list for each bond; ind[pos] = key name
		##	(The two objects are reversed lookups of each other)
			#indices = self.MPO_indices = [ dict(itertools.izip(G[(i+1)%L].keys(), range(chi[i]))) for i in range(L) ]
		MPO_labelsR = [ set(G[(s+1)%L].keys()) for s in range(L) ] #formed from rows
		MPO_labelsC = [ set( [ k for r in G[s%L].values() for k in r.keys()]  )   for s in range(L) ] #formed from columns
		
		self.MPO_labels = [ list(MPO_labelsR[s] & MPO_labelsC[s]) for s in range(L)]
		
		indices = self.MPO_indices = [ dict([ (key,i) for i,key in enumerate(self.MPO_labels[s]) ]) for s in range(L) ]

		
		self.vL = np.array([ ind['R'] for ind in indices])
		self.vR = np.array([ ind['F'] for ind in indices])
		for i in range(L):
			g = G[i%len(G)]
			w = W[i]
			ind = indices[(i-1)%L]
			for r, v in g.items(): #row and list of edges
				if r not in ind: #CHECK ME - due to union of MPO labels sometimes we call pruned node
					continue
				ir = ind[r] #where in hilbert space
				for c, U in v.items():
					for u in U:
						op = getattr(self, u[0])
				
						
						op = op[i%len(op)]
				
						try:
						
							if w[ir][indices[i][c]]==None:
								
								w[ir][ indices[i][c]] = u[1]*op
							else:
								w[ir][indices[i][c]] += u[1]*op
						
						except:
							KeyError
		self.H_mpo = W

		#print('site_number=',self.L)
		#print(len(W))
		#print(len(W[1]))
		#print(len(W[1][1]))
		#for i in range(len(W[1][1])):
		#	print(W[1][1][i])
		#quit()
		#print(len(W[0][0][0]))
		#print(len(W[0][0][0][0]))
		#print('WORKS!')
		#quit()
		"""
		DO NOT CONSERVE CHARGE HERE BECAUSE CANNOT LOAD OLD NPC ARRAY
		if self.num_q:
			self.detect_Qmpo_flat(verbose=verbose)

		self.init_model_for_conservation()
		"""



	def detect_Qmpo_flat(self, trim=True, verbose=1):
		"""	Automatically constructs:
					Qmpo_flat, chi
		
			given (for i in range(L)):
				Qp_flat[] - list of Qp_flat (per usual)
				H_mpo[] - list of np.array style W-matrices
				vL[] - list of MPO start positions on left
				vR[] - list of MPO end positions on right
			If trim=True, it will automatically trim "dead" branches (unused MPO nodes). If self.bc=='finite', this criteria assumes only self.L W-matrices, optimizing for finite  DMRG.	
				
		"""
		if verbose >= 1: print( "Detecting MPO charges...")
		L = self.L
		if self.bc=='finite':
			maxL = L
		else:
			maxL = None
			
		Qp_flat = self.Qp_flat
		W = self.H_mpo
		chi = np.array([ w.shape[1] for w in W], dtype=np.int64)
		
		if hasattr(self, 'translate_Q1') and self.translate_Q1 is not None:
				do_translate = True
		else:
			do_translate = False
			self.translate_Q = None
			self.translate_Q1 = None
				
		#Find non-zero edges
		Con = [ np.sum(np.sum(np.abs(w), axis=2), axis=2) for w in self.H_mpo ]
		
		self.Qmpo_flat = Qmpo_flat =  [ np.zeros((chi[i], self.num_q), dtype = np.int64) for i in range(L) ]
		
		#Which indices have been visited
		doneL2R = [ np.zeros(chi[i], dtype=np.bool) for i in range(L) ]
		doneR2L = [ np.zeros(chi[i], dtype=np.bool) for i in range(L) ]
		
		def travel_fromL2R(i1, r):
			"""i1 denotes location of current EDGE (W-matrix)
			"""
			if maxL is not None and i1>=maxL:
				return
				
			i = i1%L
			if do_translate:
				chargeL = self.translate_Q(Qmpo_flat[(i-1)%L][r,:], (i-1) - (i-1)%L)
			else:
				chargeL = Qmpo_flat[(i-1)%L][r,:]				
			Qp1 = Qp_flat[i]
			for c in np.nonzero(Con[i][r, :])[0]:			
				if not doneL2R[i][c]:
					doneL2R[i][c]=True
					op = W[i][r, c]					
					chargeOp = npc.Array.detect_ndarray_charge_flat(op, [Qp1, Qp1], q_conj = [1, -1], mod_q = self.mod_q)
					Qmpo_flat[i][c, :] = npc.mod_onetoinf(chargeOp+chargeL, self.mod_q)
					travel_fromL2R(i1+1, c)
					"""TODO - debugging
					if c == self.vR[i] and Qmpo_flat[i][c, :]
					
					error = travel_fromL2R(i1+1, c)
					if error:
						error.append((i1, r))
						return error
					"""
			
		def travel_fromR2L(i1, c):
			"""i1 denotes location of current edge (W-matrix). Just keep track of nodes visited."""
			if maxL is not None and i1 < 0:
				return
			i = i1%L
			
			for r in np.nonzero(Con[i][:, c])[0]:
				if not doneR2L[(i-1)%L][r]:
					doneR2L[(i-1)%L][r] = True
					travel_fromR2L(i1-1, r)

		#Traverse from left to right - this will determine all charge data
		
		doneL2R[-1][self.vL[-1]] = True
		travel_fromL2R(0, self.vL[-1])
		
		#Traverse from right to left - will help locate additional dead nodes
		doneR2L[L-1][self.vR[L-1]] = True
		travel_fromR2L(L-1, self.vR[L-1])

		if self.bc=='periodic':
			for s in range(L):
				doneL2R[s][self.vL[s]] = True
				doneR2L[s][self.vL[s]] = True
				doneL2R[s][self.vR[s]] = True
				doneR2L[s][self.vR[s]] = True
				

		#TODO optimize this better
		if self.bc=='finite':
			doneL2R[-1][:] = False
			doneR2L[-1][:] = False
			
			for s in range(L):
				doneL2R[s][self.vL[s]] = True
				doneR2L[s][self.vL[s]] = True
				doneL2R[s][self.vR[s]] = True
				doneR2L[s][self.vR[s]] = True
						
		for i in range(L):
			dead = np.nonzero(np.logical_not(np.logical_and(doneL2R[i], doneR2L[i])))[0]
			keep = np.nonzero(np.logical_and(doneL2R[i], doneR2L[i]))[0]

			if len(dead):
				#print "Dead branches of MPO[", i,"] :", dead
				#if oppindices is not None:
				#	print "Names of the Dead"
				#	print [self.MPO_labels[i][d] for d in dead]
				if trim:
					if verbose >= 1: print ("\tTrimming MPO chi[%s]: %s -> %s" % (i, chi[i], len(keep)))

					W[i] = W[i][:, keep, :, :]
					W[(i+1)%L] = W[(i+1)%L][keep, :, :, :]
					Qmpo_flat[i] = Qmpo_flat[i][keep, :]
					chi[i] = len(keep)
				##	Have to figure out new position of vL/R
					self.vL[i] = np.count_nonzero(keep < self.vL[i])
					self.vR[i] = np.count_nonzero(keep < self.vR[i])
				## Invalidate indices
					try:
						self.MPO_indices[i] = None
					except:
						pass
					try:
						self.MPO_labels[i] = [ self.MPO_labels[i][k] for k in keep ]
					except:
						pass
					
				
		self.chi = chi

	def print_MPO_charges(self):
		L = self.L
		for site in range(-1, L):
			if site >= -1:
				print('\tSite %s Charges:' % site)
				tQ = site - (site % L)
				if tQ: print( tQ)
				for Op in self.site_ops:
					print( '\t\t%s: %s' % (Op, self.translate_Q(getattr(self, Op)[site % L].charge, tQ)))

			bond = site
			tQ = bond - (bond % L)
			print ('\tBond %s Charges:' % bond)
			for i in range(len(self.Qmpo_flat[bond % L])):
				print( '\t\t%s: %s' % (self.MPO_labels[bond % L][i], self.translate_Q(self.Qmpo_flat[bond % L][i], tQ)))
		
		


	def init_model_for_conservation(self):
		""" Prepares a model defined using ndarrays for the np_conserved algorithms. Both with and without conserved charges,
			
			M must have the following attributes:
				- M.L = length of unit cell
				- M.H_mpo = list (1/site) of the ndarray MPO at each site
				- M.vL = list (1/site) of the MPO index which gives the identity to the left
				- M.vR = list (1/site) of the MPO index which gives the identity to the right
				
					(For upper triangular form, for instance, vL = 0 and vR = D - 1)
			
			1) For conserving systems, M must have the attributes
			
				M.Qp_flat = a list (1/site) of [d, num_q] 2-arrays denoting the charge of the states on the site
				
				M.Qmpo_flat = a list (1/bond) of [D, num_q] 2-arrays denoting the charge on the bonds of the MPO
			
			2) For non conserving systems, either don't set one of the items mentioned in 1, or set M.num_q = 0. The code will wrap up the operators as npc.array types with num_q = 0
			
					
			Defining states and operators:
			It's convenient for the model to define states (ex.: M.up = 0, M.down = 1), site operators (ex.: M.Sx = [d, d] ) and bond operators (ex.: M.SzSz = [d, d, d, d]). We would like init_ to wrap these operators as npc.arrays. The simplest way to do this in Python is for the model to set a list of names of the operators. For example, 
			>>> M.up = 0
			>>> M.down = 1
			>>> M.states = ['up', 'down']
			>>> 
			>>> M.Sz = np.array([[1, 0], [0, -1]])
			>>> M.site_ops = ['Sz']
			>>> 
			>>> M.SillyOp = np.ones((d, d, d, d))
			>>> M.bond_ops = ['SillyOp']
			
			init_ can then run through the names in M.states, M.site_ops, M.bond_ops and modify the attributes of the model accordingly using getattr and setattr, turning them into npc.array's.
			
			
			So a model can optionally can set any of
				- M.states = a list of names corresponding to any single site states defined. For example, suppose I have an S = 1/2 system and for convenience I want to define names
						M.up = 0, M.down = 1
					in order to build a product state like [M.up, M.down]. Set M.states = ['up', 'down'] and M.up/down will be redefined according to the permuation.
					
									
				- M.site_ops = a list of names of 1 site operators. 
				- M.bond_ops = a list of names of 2 site operators. The operators themselves should have the standard 4 - leg O_{i1 i2 j1 j2} form, with 1 / 2 refering to the two sites.
		"""

		#TODO Should work even if H_mpo isn't provided?
		
		L = self.L

		try:
			num_q = self.num_q
		except AttributeError:
			num_q = self.num_q = 0
						
		if self.num_q == 0: #Provide trivial data for  conservation
			self.mod_q = np.empty( (0,), np.int64)
			self.Qmpo_flat = [ np.empty((self.H_mpo[i].shape[1], 0), np.int64) for i in range(L) ]
			self.Qp_flat = [ np.empty((self.H_mpo[i].shape[2], 0), np.int64) for i in range(L) ]
			do_translate = False
			self.translate_Q = None
			self.translate_Q1 = None
		else:
			mod_q = self.mod_q
			if hasattr(self, 'translate_Q1') and self.translate_Q1 is not None:
				do_translate = True
			else:
				do_translate = False
				self.translate_Q = None
				self.translate_Q1 = None
			for i in range(L):
				self.Qp_flat[i] = npc.mod_onetoinf(self.Qp_flat[i], mod_q)
		
		try:
			fracture_mpo = self.fracture_mpo
		except AttributeError:
			fracture_mpo = self.fracture_mpo = False

		if self.verbose > 0:
			print( "Init model with num_q =", num_q)
		
		Qmpo_flatT = [None]*L # *Temporary* list of the new q_flat form after permutation
		Qp_flatT = [None]*L   # (these will be assigned to self.Qmpo_flat / self.Qp_flat at very end)
		self.Qp_ind = [None]*L #Resulting q_ind for site (just for convenience)
		self.p_perm = [None]*L #permutations used on the physical sites (taking original to new)
		
		for i2 in range(L): #First deal with MPOs and 2-site H ops
			i1 = (i2 - 1)%L
			i3 = (i2 + 1)%L
			np_mpo = self.H_mpo[i2]
			mpo_norm = np.linalg.norm(self.H_mpo[i2].reshape(-1))
			Qp2 = self.Qp_flat[i2]
			Qp3 = self.Qp_flat[i3]
			Qmpo1 = self.Qmpo_flat[i1]
			Qmpo2 = self.Qmpo_flat[i2]
						
			#2-site H, sites at i2, i3
			#If do_translate, when i3 = 0 we need to sort according to 'standard position' (at i3 =  0), but translate charge to L
			
			if num_q > 0:
				p = np.lexsort(Qp3.T) #Sort by "standard" location
			else:
				p = np.arange(Qp3.shape[0])
				
			if do_translate and i3 == 0: #but translate value of charge back
				Qp3 = self.translate_Q(Qp3, L)
				
			if self.H is not None:
				perm, self.H[i2] = npc.array.from_ndarray_flat(self.H[i2], [Qp2, Qp3, Qp2, Qp3 ], q_conj = [1, 1, -1, -1], mod_q = self.mod_q, sort = ['S', p, 'S', p])
				
				self.H[i2].set_labels( ['p0', 'p1','p0*', 'p1*'] )
			#MPOs. Convention: We *always* sort mpo legs, but  bunch/fracture q_ind according to fracture_mpo flag
			
			#If do_translate, when we are at i2 = 0, the Qmpo at i1 = -1 needs to be sorted as if at L-1, but  translated back to -1. Otherwise, nothing special.
			
			if num_q>0:
				p = np.lexsort(Qmpo1.T) #Regardless, sort by "standard" location
			else:
				p = np.arange(Qmpo1.shape[0])	
				
			if do_translate and i2 == 0: #but translate value of charge back
				Qmpo1 = self.translate_Q(Qmpo1, -L)
				 
			if fracture_mpo: #We DONT bunch the MPO index by charge sector
				perm, self.H_mpo[i2] = npc.array.from_ndarray_flat(self.H_mpo[i2], [Qmpo1, Qmpo2, Qp2, Qp2], q_conj = [1, -1, 1, -1], mod_q = self.mod_q, sort = [p, 'S', 'S', 'S'],  bunch = [False, False, True, True], cutoff = 2e-14)
			else: #Bunch MPO index
				perm, self.H_mpo[i2] = npc.array.from_ndarray_flat(self.H_mpo[i2], [Qmpo1, Qmpo2, Qp2, Qp2], q_conj = [1, -1, 1, -1], mod_q = self.mod_q, sort = [p, 'S', 'S', 'S'], cutoff = 2e-14)
			self.H_mpo[i2].set_labels(['w', 'w*', 'p', 'p*'])
			
			Qmpo_flatT[i2] = Qmpo2[perm[1], :] #the charges after sorting
			Qp_flatT[i2] = Qp2[perm[2], :]

			perminv_mpo = np.argsort( perm[1] )
			
			try:
				self.vL[i2] = perminv_mpo[self.vL[i2]]
				self.vR[i2] = perminv_mpo[self.vR[i2]]
			except AttributeError:
				pass
			
			
			self.p_perm[i2] = perm[2]
			self.Qp_ind[i2] = self.H_mpo[i2].q_ind[3]
		
		##	Check if anything elements went missing from charge conservation
			np_mpo = np.take(np.take(np_mpo, perm[0], axis=0), perm[1], axis=1)	# shuffle the un-conserved array according to perm
			np_mpo = np.take(np.take(np_mpo, perm[2], axis=2), perm[3], axis=3)
			diff = (self.H_mpo[i2].to_ndarray() - np_mpo)
			diff_norm = np.linalg.norm(diff.reshape(-1))
			if abs(diff_norm) > 2e-14:
				print ("Warning! some elements from the mpo[" + str(i2) + "] were dropped (norm = " + str(diff_norm) + ")")
				if self.verbose >= 15: 
					print (joinstr(diff))
				if self.verbose >= 1:
					npnpc_diff = np.array(np.nonzero(diff)).transpose()
					if hasattr(self, 'MPO_labels'):
						print ('\tDifferences:')
						labels_L = self.MPO_labels[(i2-1)%L]
						labels_R = self.MPO_labels[i2]
						for rc in npnpc_diff:
							print ('\t\t%s!%s [%s] -> %s!%s [%s] : %f' % (i2, labels_L[perm[0][rc[0]]], rc[0], (i2+1)%L, labels_R[perm[1][rc[1]]], rc[1], np.linalg.norm(np_mpo[rc[0], rc[1]]) ))
					else:
						print (joinstr(['\tDifferences at ', ', '.join(map(str, npnpc_diff))]))
			
		### Generate states, site_ops and bond_ops
		
		#	site_ops initial form:

		## default-to-false is a fairly arbitrary choice. I (CW) make it so the defaultpreserves previous behavior.
		states_uniform = False
		try:
			if type(self.states[0])!=list:
				# self.states= [self.states]
				self.states = [self.states for i in range(L)]
				states_uniform = True
		except: # no states defined
			#Default state naming - 'p0, p1, . . . '
			self.states = [ [ 'p'+str(i) for i in range(len(p)) ] for p in self.p_perm ]
		states = self.states

		try:
			site_names = self.site_ops
		except:
			site_names = []
		try:
			bond_names = self.bond_ops
		except:
			bond_names = []

		site_ops = []
		bond_ops = []
		for names, ops in [ (site_names, site_ops), (bond_names, bond_ops) ]:
			for n in names:
				op = getattr(self, n)			
				if type(op)!=list:
					op = [op]
				ops.append(op)

			
		#TODO By convention, we assume Ops are at site / bond 0.
		#I use silly i1/i2 definition for easy generalization later.

			
		npc_states = [ None ]*L
		npc_site_ops = [ [None]*L for n in site_names]
		npc_bond_ops = [ [None]*L for n in bond_names]

		for i1 in range(L):
			i2 = (i1+1)%L
		
			#States

			state = [ states[i1][j] for j in self.p_perm[i1] ]

			Qp1 = self.Qp_flat[i1]
			Qp2 = self.Qp_flat[i2]
			
			#If i2 = 0, we have wrapped around unit cell, so deal
			if num_q > 0:
				p = np.lexsort(Qp2.T)
			else:
				p = np.arange(Qp2.shape[0])
				
			if do_translate and i2 == 0:
				Qp2 = self.translate_Q(Qp2, L)
			
			#Site operators
			# TODO: put the try statement inside the loop
			for n in range(len(site_names)): #for each op type
				op = site_ops[n][i1%len(site_ops[n])]
				if num_q > 0:
					charge = npc.array.detect_ndarray_charge_flat(op, [Qp1, Qp1], q_conj = [1, -1], mod_q = self.mod_q)
				else:
					charge = None
					
				perm, op = npc.array.from_ndarray_flat(op, [Qp1, Qp1], q_conj = [1, -1], charge = charge, mod_q = self.mod_q)
				
				op.set_labels(['p', 'p*'])
				npc_site_ops[n][i1] = op

			
			#Bond operators	
			# TODO: put the try statement inside the loop
			# TODO: check that permutation (charge assignments) is uniform across sites
			for n in range(len(bond_names)): #for each op type
				op = bond_ops[n][i1%len(bond_ops[n])]
				if num_q > 0:
					charge = npc.array.detect_ndarray_charge_flat(op, [Qp1, Qp2, Qp1, Qp2 ], q_conj = [1, 1, -1, -1], mod_q = self.mod_q)
				else:
					charge = None

				perm, op = npc.array.from_ndarray_flat(op, [Qp1, Qp2, Qp1, Qp2], q_conj = [1, 1, -1, -1], charge = charge, mod_q = self.mod_q, sort = ['S', p, 'S', p])
				npc_bond_ops[n][i1] = op

			#TODO verify leg
			npc_states[i1] = dict([ (state[j],j) for j in range(len(state)) ])
			if states_uniform:
				for k, v in npc_states[0].items():
					setattr(self, k, v)
		
		self.states = npc_states
		for n in range(len(site_ops)):
			setattr(self, site_names[n], npc_site_ops[n])
		for n in range(len(bond_ops)):
			setattr(self, bond_names[n], npc_bond_ops[n])

		self.Qp_flat = Qp_flatT
		



	def combine_sites(self, r):
		"""This function combines together 'r' sites of an MPO, leading to a larger 'site' hilbert space. It should be matched with a corresponding call to the wavefunction.
		"""
		if r == 1:
			return
			
		if self.L%r != 0:
			raise ValueError
		
		if hasattr(self, 'grouped'):
			self.grouped = self.grouped*r
		else:
			self.grouped = r
			
		self.L = self.L//r
		
		H_mpo = [None]*self.L
		vL = [None]*self.L
		vR = [None]*self.L
		for i in range(self.L):
			h = self.H_mpo[i*r]
			for j in range(r-1):
				i2 = r*i + j + 1
				h = npc.tensordot(h, self.H_mpo[i2], axes = [[1], [0]])
				h = h.itranspose([0, 3, 1, 4, 2, 5])
				h = h.icombine_legs([ [0], [1], [2, 3], [4, 5] ], qt_conj = [1, -1, 1, -1])
				
			H_mpo[i] = h
			vL[i] = self.vL[i2]
			vR[i] = self.vR[i2]
		
		for h in H_mpo: 
			h.ipurge_zeros(2*10**(-16))
			#h.ipurge_zeros(0.)
		self.d = np.array( [h.shape[3] for h in H_mpo] )
		self.H_mpo = H_mpo
		self.vL = vL
		self.vR = vR
		
	def prep_for_savemem(self):
		"""In savemem form of DMRG, the loop over the central MPO bond is brought to the very outside of matvec, which means we don't need to store the full Env.MPO object. In the following, we partition the central index into a number of "bins". The maximum number of indices in a bin is set by the largest block size of the index, and bins are created and populated accordingly, in order to minimize the number of bins required.
		
		
			See DMRG_core for explanation of storage form.
		"""		
		Za = [None, None]
		self.p_mpo = [Za[:] for i in range(self.L)] #[ [ (q_dat1, dat1),  (q_dat2, dat2) ...]]
		self.l_mpo = [None]*self.L #Store gutted, tranposed mpos
		self.r_mpo = [None]*self.L
		
		for i in range(self.L):			
			#For each bond
			mpo0l = self.H_mpo[i]
			mpo0r = self.H_mpo[(i+1)%self.L]
			sizes = mpo0r.q_ind[0][:, 1] - mpo0r.q_ind[0][:, 0] #Size of sectors
			perm = np.argsort(sizes) #sort by size
			sizes = list(sizes[perm])
			qs = list(np.arange(len(sizes))[perm]) #inverse permutation
			
			mpol = mpo0l.transpose([1, 2, 3, 0]) #transpose for faster matvec
			mpor = mpo0r.transpose([0, 2, 3, 1])
			
			#Now I determine bins using First Fit Decreasing algorithm
			bin = [] #will store bins of qs
			bin_s = [] #size of each bin
			max_size = np.max([sizes[-1], 16])#max size of bin
			while len(sizes) > 0:
				s = sizes.pop()
				q = qs.pop()
				fit = False
				for k in range(len(bin)): #check if it fits in current bins
					if bin_s[k] + s <= max_size:
						bin[k].append(q)
						bin_s[k] = bin_s[k] + s
						fit = True
						break
						
				if not fit: #Ok, have to make new bin
					bin.append([q])
					bin_s.append(s)	
			
			num_bin = len(bin)				
			print ("Num bins:", len(bin), "Max size:", max_size, "Avg size:", np.mean(bin_s))
			print (bin_s		)
			lpmpo = []
			rpmpo = []

			for b in bin: #Now I for the q_dat and dat for each bin
				do_take = []
				for q in b:
					do_take.extend( [ r for r in range(mpol.q_dat.shape[0]) if mpol.q_dat[r, 0] == q])
				if len(do_take)>0:	
					lpmpo.append([mpol.q_dat[do_take, :], [mpol.dat[r] for r in do_take] ])
					
				do_take = []
				for q in b:
					do_take.extend( [ r for r in range(mpor.q_dat.shape[0]) if mpor.q_dat[r, 0] == q] )
				if len(do_take)>0:
					rpmpo.append([mpor.q_dat[do_take, :], [mpor.dat[r] for r in do_take] ])
			
			
			self.p_mpo[i][0] = lpmpo
			self.p_mpo[(i+1)%self.L][1] = rpmpo
			
			mpol.q_dat = np.empty((0, mpol.rank), dtype = np.uint)
			mpol.dat = []
			self.l_mpo[i] = mpol
			
			mpor.q_dat = np.empty((0, mpor.rank), dtype = np.uint)
			mpor.dat = []
			self.r_mpo[(i+1)%self.L] = mpor








class MPOgraph(object):
	"""A bare-bones MPOgraph to facilitate  adding elements boilerplate"""
	def __init__(self, L):

		self.G = [ { 'R':{ 'R':[('Id',1.)] }, 'F':{ 'F':[('Id',1.)] } } for s in range(L) ]

	def print_MPOgraph(self, multiline=False):
		MPOgraph = self.G
		N = len(MPOgraph)
		dest_delim = ',\n\t\t' if multiline else ',  '
		
		print( "==================== MPO graph ====================")
		for s in range(N):
			for k,v in sorted(MPOgraph[s].items(), key=lambda x:x[0]):
				print ("  %s!%s  -->  " % (s, k)) + dest_delim.join([ '%s!%s: ' % ((s+1)%N,target) \
					+ ' + '.join([ '%s*%s' % (tg[1],tg[0]) for tg in v[target] ]) for target in v ])
		if multiline: print ("===================================================")


	def add(self, i, r, c, op, val, force = False):
		if val!=0. or force:
			self.G[i].setdefault(r, {}).setdefault(c, []).append( (op, val))

