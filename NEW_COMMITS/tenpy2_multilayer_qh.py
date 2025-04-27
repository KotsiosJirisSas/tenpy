""" Multicomponent quantum Hall model.
	Maintained by Mike and Roger.
"""


import numpy as np
import scipy as sp
import scipy.integrate
import sys, time
import itertools, functools
from copy import deepcopy
import cPickle
from itertools import izip
import cluster.omp as omp

from algorithms.linalg import np_conserved as npc
from algorithms.linalg.npc_helper import packVmk
from algorithms.linalg import LA_tools
from models.model import model, set_var, identical_translate_Q1_data
from mps import enviro
from mps.mps import iMPS 
from tools.string import joinstr
from tools.math import perm_sign, toiterable, pad
from algorithms import Markov

"""
	How to specify a multicomponent Hilbert space.

	A typical QH interaction in real space looks like
	
		rho_mu(r1)  V_{mu nu}(r1 - r2) rho_nu(r2)
	
	where mu label 'species' (like spin). Each species can have multiple bands - like the multiple Landau levels of each spin. The properties of a band (its form factor) is encoded in a subclass of a 'band' object. A 'layer' is specified by a (species, band) tuple; there is one layer for each type of orbital in the problem. Throughout, Greek indices denote species and latin the bands.
 
	At the moment, the only bands we have implemented are 2DEG LLs with a possible nematic anisotropy.

	To set the structure of the Hilbert space, we provide a "layers" model parameter:
	
		layers = [ (mu, a), (mu', a'), . . . (species, band = LL#) ]
	For Landau levels, the band type 'a' is specified by an integer: the LL. The species labels 'mu' may be any valid dict key. The layers will be given indices l_i with an ordering in the MPS as determined by the above.
	
	Examples:
	
		layers = [ ('up', 0), ('dn', 0)]   #two spin species, called 'up', 'dn',  in  the LLL
		layers = [ (0, 0), (0, 1), (0, 2)] #one species, called 0, in the n = 0, 1, 2 LLs.
	
	As a shortcut, if only Nlayer is provided, we assume Nlayer distinct species in the LLL:
		layers = [ (0, 0), (1, 0), . . . (Nlayer-1, 0) ]
	
	The code computes two useful representations.
		1) species is a dictionary whose keys are the species, and values are lists of the layer indices owned by the species:
			species = { mu : [l_mu0, l_mu1, ... ], nu : [l_nu0, ... ] , ... }
	
		2) bands is a dictionary of 'bands' objects.
			bands[mu] = bands object for species mu
"""


"""
	How to specify 2-body interactions.
	
	
	Set the Vs model par describe more in make_Vmk
	
	'Vs' = {'eps': 0.0001 (total acceptable truncation of potentials)
			'xiK': 6, (an optional hint giving the rough real-space extent of the potential. 
						Helps scale the overlap integrals properly)
			'potential type' : { options, . . . ,  { (mu, nu) : opts, } } #dict specifying parameters for species mu, nu interaction.
			}
	
"""

class QH_model(model):
	"""	A multilayered quantum hall system on a cylinder.
		
		Parameters when creating model:
			'Lx': circumference
			'particle_type': 'F' or 'B' for fermion/boson (boson not implemented)
			'cons_Q': specifies if charge (and which combo) is conserved. Either, 'total', 'each', False, 0, or a matrix.
				total: total number.
				each: each component seperately.
				species: each species seperately.
				False, 0 : No conservation
				A matrix: M_{i mu}
				
			# Q-mat is a [num_cons_C, N] matrix - each row gives the charges of the layers under the conserved quantity
			
			'cons_K': specified if momentum is conserved.
		"""

	def __init__(self, pars):
		super(QH_model, self).__init__()
		self.model_init_t0 = time.time()
		pars = deepcopy(pars)
		verbose = self.verbose = set_var(pars, 'verbose', 1, force_verbose=0)
		if verbose > 0:
			print "An unambitious Multilayered quantum Hall model"
			print "\tverbose =", verbose
		self.dtype = set_var(pars, 'dtype', np.float)
		
		self.Lx = set_var(pars, 'Lx', 10.)	# circumference
		self.kappa = 2*np.pi/self.Lx
		
	##	Specifying the layers
		if 'layers' in pars:
			if 'Nlayer' in pars: print "!\tWarning, model parameter 'Nlayer' not used.  (Using 'layers' instead.)"
			layers = set_var(pars, 'layers', [[0,0]]) 	# list of pairs: [ (species, band), ... ]
			N = len(layers)
		else:
			N = set_var(pars, 'Nlayer', 1)	# number of layers
			if N < 1 or (not isinstance(N, int)): raise TypeError, (N, type(N))
			layers = [ [l, 0] for l in range(N) ]		# Default - N distinct species in LLL
		self.Nlayer = N
	##	Compute useful layer data (i.e., species, bands, layers)
		self.species = species = {}
		bands = {}
		for j, (mu, a) in enumerate(layers):
			species.setdefault(mu, []).append(j)
			#if mu in bands and hasattr(a,'__iter__'):
			#	raise ValueError, "trying to do LL mixing with a generalized LL, that's not allowed"
			bands.setdefault(mu, []).append(a)
	##set the type of bands depending the second part of the layer tuple
	#if its a list of integers, use the usual LL subclass corresponding to GaAs, 
	#if a list of lists, use the LL_general_combination subclass, written to handle bilayer graphene
		print "Bands", bands
		self.bands={}	
		for mu, b in bands.iteritems():
			"""
			if hasattr(b[0],'__iter__'):
				self.bands[mu]=LL_general_combination(b[0])
			"""
			if hasattr(b[0],'__iter__'):
				print "called gen comb",b
				self.bands[mu]=general_LL(b)
		
			else:		
				self.bands[mu]=LL(b)
			
	##  Anisotropy = {mu1: anisotropy1, mu2: anisotropy2, ... } sets anisotropy of form factors
		for mu, anisotropy in set_var(pars, 'anisotropy', {}).iteritems():
			self.bands[mu].anisotropy = anisotropy
					
		self.layers = layers
		if verbose >= 1: print "\tSpecies {name: orb_inds, ...}:", species
		if verbose >= 1: print "\tBands:", self.bands


	##  Boundary conditions an unit cell:
		# ('periodic'):			with 1-flux unit cell
		# ('periodic', N_phi):	periodic with N_phi flux unit cell
		# ('finite', N_phi):	finite
		if 'L' in pars: print "!\tWarning! 'L' parameter not used in quantum Hall.  Use 'boundary_conditions': ('periodic', NumPhi) instead."
		pars_bc = set_var(pars, 'boundary_conditions', set_var(pars, 'bc', 'periodic'))
		if isinstance(pars_bc, tuple) or isinstance(pars_bc, list):
			self.bc = pars_bc[0]
			self.L = pars_bc[1] * N
		else:
			self.bc = pars_bc
			self.L = N
		if verbose >= 1: print "\tL (number of sites):", self.L
		self.Nflux = self.L // self.Nlayer
		
		self.fracture_mpo = set_var(pars, 'fracture_mpo', False)
		self.p_type = p_type = set_var(pars, 'particle_type', 'F')
		self.N_bosons=set_var(pars,'Nbosons',1) #maximum number of bosons per site
		self.LL_mixing=False
		for mu,b in bands.iteritems():
			if self.bands[mu].n >1: self.LL_mixing=True
		
		if self.p_type=='F' and self.N_bosons>1: raise ValueError,"asking for multiple fermions on the same site, not allowed!"
		if self.p_type=='B' and self.LL_mixing: raise NotImplementedError,"we're not set up to do LL mixing with bosons"

		cons_Q = set_var(pars, 'cons_C', 'total')	# either, 'total', 'each', False, 0, or a matrix
		# Q-mat is a [num_cons_C, N] matrix - each row gives the charges of the layers under the conserved quantity
		if cons_Q == True or cons_Q == 1:
			if N != 1: raise ValueError, "=== Mike, update your code.  This is unacceptable. ==="
			if N == 1: Qmat = np.ones([1, 1], dtype=np.int)	# Fine, Mike, you get a pass here.
		elif cons_Q == 'total' or cons_Q == 't':
			Qmat = np.ones([1, N], dtype=np.int)
		elif cons_Q == 'each' or cons_Q == 'e':
			Qmat = np.eye(N, dtype=np.int)
		elif cons_Q == 'species' or cons_Q == 's':
			Qmat = np.zeros([len(self.species), N], dtype=np.int)
			for spi,sp in enumerate(self.species.iteritems()):
				for layeri in sp[1]:
					Qmat[spi,layeri] = 1
		elif cons_Q == False or cons_Q == 0:
			Qmat = np.empty([0, N], dtype=np.int)
		elif isinstance(cons_Q, np.ndarray) or isinstance(cons_Q, list):
			Qmat = np.atleast_2d(cons_Q)
		else:
			raise ValueError, "invalid cons_Q"
		self.num_cons_Q = Qmat.shape[0]
		mod_q = np.ones(self.num_cons_Q, int)
		self.Qmat = Qmat		# Qmat shaped (num_cons_Q, Nlayer)
		self.num_q = self.num_cons_Q

		self.cons_K = set_var(pars, 'cons_K', 1)		# is K conserved (mod q)?  0 means no, 1 means yes (integer), 2 and higher means mod.
		if self.cons_K > 0:
			self.Kvec = np.array(set_var(pars, 'Kvec', np.ones(self.num_cons_Q, int)))	# K = position * (Q . Kvec)
			self.num_q += 1
			mod_q = np.hstack([[self.cons_K], mod_q])
			if self.cons_K == 1:
				self.translate_Q1_data = { 'module':self.__module__, 'cls':self.__class__.__name__,
					'tQ_func':'translate_Q1_Kvec', 'keywords':{ 'Kvec':self.Kvec, 'Nlayer':self.Nlayer } }
			else:
				self.translate_Q1_data = None
			self.set_translate_Q1()
			#self.translate_Q1 = lambda Q, r: QH_model.static_translate_Q1_Kvec(Q, self.Kvec, self.Nlayer, r)
			#self.translate_Q1 = functools.partial(getattr(self.__class__, self.translate_Q1_data['tQ_func']), **self.translate_Q1_data['keywords'])
			#self.translate_Q1 = functools.partial(QH_model.translate_Q1_Kvec, Kvec = self.Kvec, Nlayer = self.Nlayer)

		self.mod_q = mod_q
		self.root_config = np.array(set_var(pars, 'root_config', np.zeros(self.L))).reshape((-1, N))
		self.mu = set_var(pars, 'mu', None)
		
		self.Vs = deepcopy(pars.get('Vs', {}))
		exp_approx = self.exp_approx = set_var(pars, 'exp_approx', True)
		self.add_conj = self.ignore_herm_conj = set_var(pars, 'ignore_herm_conj', False)
		self.Vabcd_filter = set_var(pars, 'Vabcd_filter', {})
		self.graphene_mixing=set_var(pars,'graphene_mixing',None)


	##	Process stuff from the parameters collected above
		self.make_Hilbert_space()
		if verbose >= 4: print joinstr(['\tQmat: ', self.Qmat, ',  Qp_flat: ', joinstr(self.Qp_flat)])
		#print joinstr([ self.translate_Q1(Qp, self.L) for Qp in self.Qp_flat ])
		
		self.make_Vmk(self.Vs, verbose=verbose)	# 2-body terms
		self.convert_Vmk_tolist()

		if 'Ts' in pars:
			self.make_Trm(pars['Ts'])	# 1-body terms
			self.convert_Trm_tolist()
		
		if pars.get('make_Ht', False):
			self.make_Ht()
			
		# we *have* to call this first before other MPOgraph codes
		if isinstance(exp_approx, tuple) or isinstance(exp_approx, list) and len(exp_approx) >= 1:
			# allow for cases where exp_approx = (instruction, paramters...)
			exp_approx_opt = exp_approx[1:]
			exp_approx = exp_approx[0]
		if exp_approx == "load Markov system from file":
			self.load_Markov_system_from_file(exp_approx_opt[0], verbose=verbose)
		elif len(self.Vmk_list) == 0 or exp_approx == "Don't make MPOgraph from Vmk":
			self.make_empty_MPOgraph1(verbose=verbose)		# No potentials.
		elif exp_approx == False or exp_approx is None or exp_approx == 'none':
			self.make_MPOgraph1_from_Vmk_old(verbose=verbose)
		elif exp_approx == '1in':
			self.make_MPOgraph1_from_Vmk_exp(verbose=verbose)
		elif exp_approx == True or exp_approx == 'slycot':
			if verbose >= 3: print "\tSlycot loaded: %s, more_slicot loaded: %s" % (Markov.Loaded_Sylcot, Markov.Loaded_more_slicot)
			self.make_MPOgraph1_from_Vmk_slycot(verbose=verbose)
		else:
			print "Unrecognized options: 'exp_approx': {}".format(exp_approx)
		self.MPOgraph1_to_MPOgraph(verbose=verbose)

		if self.mu is not None:
			self.add_chemical_pot(self.mu)
			
		self.V3=set_var(pars,'nnn',None)		
		if self.V3 is not None:
			self.add_three_body(self.V3)
			
		if hasattr(self, 'Trm_list'):
			self.add_Trm()

		if verbose >= 7: self.print_MPOgraph()
		self.build_H_mpo_from_MPOgraph(self.MPOgraph, verbose=verbose)
		#if verbose >= 2: print "\tMPO chi:", self.chi
		if verbose >= 2: print "\tBuilding QH_model took {} seconds.".format(time.time() - self.model_init_t0)


	################################################################################
	################################################################################
	##	Measuring stuff
	def measure_density(self, psi, xs, ys, S_munu=None, verbose=2):
		"""	Measures real space "density" psi^D_mu(r) psi_nu(r) at positions [ xs x ys ].
		
			xs and ys are 1D arrays.
			S_munu is a list of tuples (mu,nu)
			
			Returns 3D array, shaped (len(xs), len(ys), Nlayer)
			
			TODO. This function breaks our naming conventions. mu, nu label layers, not species. It does not sum over bands.
			TODO. Uses LLL orbitals.

		"""

		N = self.Nlayer
		xs = np.sort(xs)
		ys = np.sort(ys)
		kappa = self.kappa
		try:
			origin = psi.realspace_origin
		except AttributeError:
			origin = 0
		ys += origin * kappa
		if S_munu is None:
			S_munu = [ [i,i] for i in range(N) ]
		S_munu = np.atleast_2d(np.array(S_munu, int))

		Lcut = 2.5		# cutoff in realspace
		Ocut = int(Lcut/kappa)
		n1 = int(np.floor((ys[0] - Lcut)/kappa))
		n2 = int(np.ceil((ys[-1] + Lcut)/kappa))+1
		if psi.bc == 'segment' or psi.bc == 'finite':
			n1 = np.max([0, n1])
			n2 = np.min([psi.L/N, n2])

		rhozS = []
		for S in S_munu:
			t0 = time.time()
			mu,nu = S
			if self.p_type == 'F':
				if mu == nu:
					rho = psi.correlation_function(self.AOp, self.aOp, sites2 = np.arange(N*n1 + mu, N*n2 + mu, N), sites1 = np.arange(N*n1 + nu, N*n2 + nu, N), OpStr = self.StrOp, hermitian = True)
				else:
					rho = psi.correlation_function(self.AOp, self.aOp, sites2 = np.arange(N*n1 + mu, N*n2 + mu, N), sites1 = np.arange(N*n1 + nu, N*n2 + nu, N), OpStr = self.StrOp)
			else:
				raise NotImplementedError
				rho = psi.correlation_function(self.AOp, self.aOp, sites2 = range(n1, n2), sites1 = range(n1, n2), hermitian = True)
			if verbose:
				print "Correlation function of S_%s took %s seconds." % (S, time.time() - t0)

			rhoz = np.zeros([len(xs), len(ys)], dtype = complex)
			for n in xrange(n2 - n1):
				for m in xrange(n2 - n1):						
					rhoz += np.outer( np.exp(1j*xs*kappa*(n - m)), np.exp(- 1/2.*((n+n1)*kappa - ys)**2 - 1/2.*((m+n1)*kappa - ys)**2)) * (rho[n, m]/self.Lx/np.sqrt(np.pi) )
			if mu == nu: rhoz = rhoz.real
			rhozS.append(rhoz)
		return rhozS
	
	def measure_C(self, psi, z1, z2, Lcut = 3.):
		"""Measures real space Green's function < Psi^T(z1) Psi(z2) > in LLL
			
			z1/2 = [ [x0, y0], [x1, y1], . . . ]
		"""
		
		if self.Nlayers > 1:
			raise NotImplemented
		t = self.kappa
		try:
			origin = psi.realspace_origin
		except AttributeError:
			origin = 0.

		z1[:, 1] += origin * t
		z2[:, 1] += origin * t
		
		y1L = np.min(z1[:, 1])
		y2L = np.min(z2[:, 1])
		y1R = np.max(z1[:, 1])
		y2R = np.max(z2[:, 1])
				   
		n1L = int((y1L - Lcut)/t)
		n1R = int((y1R + Lcut)/t)
		n2L = int((y2L - Lcut)/t)
		n2R = int((y2R + Lcut)/t)
		
		w1 = psi.L
		w2 = np.max( [n2R - n1L, n1R - n2L, w1])
		rho = np.zeros( [w1, w2], dtype = complex)
		t0 = time.time()
		if self.cons_K==1:
			Ns = psi.site_expectation_value(self.nOp)
			for i in xrange(w1):
				rho[i, i] = Ns[i]
			
		else:
			for i in xrange(w1):
				rho[i,:] = psi.correlation_function(self.Aop, self.aOp, sites2 = range(i, i+w2), sites1 = [i], OpStr = self.StrOp)
		print "rho took", time.time() - t0
		C = np.zeros([len(z1), len(z2)], dtype = complex )
		
		for n in xrange(n1L, n1R):
			for m in xrange(n2L, n2R):
				
				if m >= n:
					g = rho[n%w1, m-n]
				else:
					g = np.conj(rho[m%w1, n - m])
				
				C += np.outer(np.exp(1j*t*z1[:,0]*n - 1/2.*(n*t - z1[:,1])**2 ), np.exp( -1j*t*z2[:,0]*m - 1/2.*(m*t - z2[:,1])**2  ))*(g/self.Lx/np.sqrt(np.pi))
		return C


	def species_densities(self, psi=None, verbose=0):
		"""Return a dictionary {species:orbital filling density}	"""
		if psi is None:
			exp_n = self.root_config.reshape(-1)
			exp_n = np.mean(exp_n.reshape((-1,self.Nlayer)), axis = 0)
		else:
			if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
			exp_n = psi.site_expectation_value(self.nOp)
			exp_n = np.mean(exp_n.reshape((-1,self.Nlayer)), axis = 0)
		densities = {}
		for s,orb in self.species.items():
			densities[s] = np.sum([ exp_n[i] for i in orb ])
			
		return densities


	def species_orbital_densities_x(self, psi, Fourier=False, verbose=0):
		"""Measures the orbital density of each species.
	Returns a dictionary {species: Ds}, where Ds is a 3-array with elements Ds_{a,b,m} = <c_{m,a}^dag c_{m,b}>, where m is the site."""
		Ds = {}
		Nlayer = self.Nlayer
		if psi.L % Nlayer: raise ValueError, (psi.L, Nlayer)
		Nflux = psi.L // Nlayer
		if Fourier:
			Ds['qy_list'] = self.Lx / float(Nflux) * np.arange(Nflux)		# pray 'qy_list' is not a species label.
		for spec,orb in self.species.items():
			Nb = len(orb)
			exp_n = np.zeros((Nb, Nb, Nflux), dtype=psi.dtype)
			for x in range(Nflux):
				osx = x * Nlayer
				sites = x * Nlayer + np.arange(Nlayer)
				#for b in range(Nb):
				#	exp_n[b, b, x] = psi.site_expectation_value(self.nOp, sites=[osx+b])[0]
				#for b1 in range(Nb):
				#	for b2 in range(Nb):
				#		if b1 < b2:
				#			exp_n[b1, b2, x] = psi.correlation_function(self.AOp, self.aOp, [osx+b2], [osx+b1], OpStr=self.StrOp)
				exp_n[:, :, x] = psi.correlation_function(self.AOp, self.aOp, sites2=sites, sites1=sites, OpStr=self.StrOp, hermitian=True)
			if Fourier:
				exp_n = np.fft.ifft(exp_n, axis=2)
			Ds[spec] = exp_n
		return Ds
	

	def pair_correlation(self, psi, max_m = None, xi = 20., verbose=0):
		"""
			S[mu][nu](qx, qy, dqy) = < :rho_mu(qx, qy) rho_nu(-qx, -qy+dqy): >
			
			max_m is the number of qx modes to compute.
			xi is estimate of correlation length (in the y-dir)
			
			returns a dictionary with stuff S[mu][nu](qx, qy, dqy), qx, qy, dqy
			
			
			S_munu(-qx, qy, dqy)= 	<rho_mu(-qx, qy)rho_nu(qx, -qy+dqy)>
							=	<rho_nu(qx, -qy+dqy) rho_mu(-qx, qy)>
							=	S_numu(qx, dqy-qy, dqy)
							
			
			S_munu(-qx, qy, dqy)= 	<rho_mu(-qx, qy)rho_nu(qx, -qy+dqy)>
				=	<rho_nu(-qx, qy-dqy) rho_mu(qx, -qy)>^ast
				=	S_numu(-qx, qy-dqy, dqy)
		"""
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		Nflux = psi.L // self.Nlayer 	# length of unit cell
		kappa = self.kappa
		Lx = self.Lx

		max_k = int(xi/kappa/Nflux + 1)*Nflux	# choose based on correlation length; commensurate with flux
		if max_m is None: max_m = int(Lx / 1. + 1)	#
		
		#Temporarily suppress band indices,  qx = kappa*m
		#S(qx, qy, qy+dqy) 	= <:rho(qx, qy) rho(-qx, -qy+dqy):>
		#					= sum_{x, k} exp(-1j*[ qy*(x + k + m/2) - (qy-dqy)*(x+m/2) ]*kappa)
		#					= sum_{x, k} exp(-1j*[ qy*k + dqy*(x+m/2) ]*kappa)
		#Amounts to FFT in x, k --> dqy, qy followed by a twiddle = exp(-1j*dqy*m/2*kappa)

		#S_munu(-qx, qy, dqy)= 	<rho_mu(-qx, qy)rho_nu(qx, -qy+dqy)>
		#					=	<rho_nu(qx, -qy+dqy) rho_mu(-qx, qy)>
		#					=	S_numu(qx, dqy-qy, dqy)

		##	--- I have instead ---
		##	S(qx, qy, dqy) = <: rho_ab(qx, qy) rho_cd(-qx, -qy+dqy) :>
		##	               = sum < pd_{x+k,a} pd_{x+m,c} p_{x,d} p_{x+m+k,b} > exp[ i kappa [(x+k+m/2)qy + (x+m/2)(-qy+dqy)] ]
		##	               = sum <correlator> exp[ i kappa qy k + i kappa dqy x ] exp[ i kappa dqy m/2 ]
		##	and m = -qx / kappa.
		##
		##	On the other hand, if I pick m = qx / kappa, then
		##	S(qx, qy, dqy) = sum <correlator> exp[ -i kappa qy k + i kappa dqy (x + k + m/2) ]
		
		##	two_body_correlator_pbc() returns dictionary with keys
		##	'C':         7-array with indicies [a,b,c,d,x,m,k] and shape (Nlayer, Nlayer, Nlayer, Nlayer, Nflux, max_m+1, 2*max_k+1).
		##	'k_offset':  gives the location of k=0 for the k-index (leg #6).
		D = self.two_body_correlator_pbc(psi, max_m = max_m, max_k = max_k, remove_disconnected_part = False, compute_negative_m = True, verbose = verbose)
		k_offset = D['k_offset']
		m_offset = D['m_offset']
		if m_offset!=max_m:
			raise ValueError
		if k_offset!=max_k:
			raise ValueError
		
		C = C_abcdxmk = D['C']
		
		C = np.roll(C, -m_offset, axis = 5)		# bring m = 0 to index 0
		C = np.roll(C, -k_offset, axis = 6)		# bring k = 0 to index 0
		C = np.fft.fft2(C, axes = (4, 6)) / Nflux #FFT in x, k, scale away unit cell.
		C = np.concatenate( [ C[:, :, :, :, 1:, :, :], C], axis = 4 ) #Add recicprically related copy of dqy
		
		qx = np.arange(-m_offset, m_offset + 1) * kappa			# quantized in units of 2pi/L
		qx = np.roll(qx, -m_offset)
		qy = np.linspace(0., Lx, C.shape[6], endpoint=False)	# BZ has size L
		qy = np.mod(qy + Lx/2., Lx) - Lx/2.
		dqy = np.linspace(0., Lx, Nflux, endpoint=False)
		dqy = np.concatenate( [dqy[1:]-Lx, dqy]) #Add recicprically related copy of dqy
		
		#exp(-1j*dqy*m/2*kappa)
		twiddle = np.exp( np.outer(dqy, -1j*qx/2))
		C = C.transpose([0, 1, 2, 3, 6, 4, 5])*twiddle
		C = C.transpose([0, 1, 2, 3, 5, 6, 4])
		#Order: a b c d dqy qx qy
	
		S = {}
		#STEP 1. Contract ab->mu using Fab
		Qx, Qy1 = np.meshgrid(qx, qy, indexing='ij')
		for mu, b1  in self.bands.iteritems():
			F1=b1.F
			pre_F1=b1.pre_F
			n_mu = b1.n
			s = 0.
			for a, b in itertools.product(xrange(n_mu), xrange(n_mu)):
				Fab = pre_F1(a, b)*F1(a, b, Qx, Qy1)
				a = self.species[mu][a] #get layer index
				b = self.species[mu][b]
				s+= C[a, b]*Fab
			S[mu] = s
			
		#STEP 2. Contract cd->nu using Fcd
		dQy, Qx, Qy = np.meshgrid(dqy, qx, qy, indexing='ij')
		Qy2 = Qy - dQy
		
		for mu, b1  in self.bands.iteritems():
			s = {}
			for nu, b2 in self.bands.iteritems():
				F2=b2.F
				pre_F2=b2.pre_F
				n_nu = b2.n
				s[nu] = 0.
				for c, d in itertools.product(xrange(n_nu), xrange(n_nu) ):
						Fcd = pre_F2(c, d)*F2(c, d, -Qx, -Qy2)

						c = self.species[nu][c]
						d = self.species[nu][d]
						s[nu]+= S[mu][c, d]*Fcd
				s[nu] = s[nu].transpose([1, 2, 0])
				
				
			S[mu] = s
		C = C.transpose([0, 1, 2, 3, 6, 4, 5])
		return {'S':S, 'qx':qx, 'qy':qy, 'dqy':dqy, 'C':C, 'C_abcdxmk':C_abcdxmk}

	def convert_S2g(self, y0, dat):
		""" Compute g( (x0=0, y0),  (x, y))  from S.
		
			dat = output of pair_correlation = {'S':S, 'qx':qx, 'qy':qy, 'dqy':dqy, 'C':C} 
			
			Returns {'G':G, 'x':x, 'y':y}
		
		"""
		G = {}
		for mu in dat['S'].keys():
			G[mu] = {}
			for nu in dat['S'][mu].keys():
				g = 0.
				S = dat['S'][mu][nu]            
				#Stack together to add -qx components
				for j, dqy in enumerate(dat['dqy']):                          
					g+= np.exp(1j*dqy*y0)*np.fft.fft2(S[:, :, j])*dat['qy'][1]/self.Lx/2/np.pi/2/np.pi
					
				G[mu][nu] = g.real
		
		x = np.linspace( -np.pi/dat['qx'][1], np.pi/dat['qx'][1], len(dat['qx'])  )
		y = np.linspace( -np.pi/dat['qy'][1], np.pi/dat['qy'][1], len(dat['qy'])  )
		   
		return {'G':G, 'x':x, 'y':y}
	
		
	def structure_factor(self, psi, mu, a, xi = 30,nu=None,b=None):
		""" Diagonal structure  of species 'mu' projected into band 'a' with 'nu' projected into band 'b'
		
				S(q) = < P_a rho_mu(-q) P_a  P_b rho_nu(q) P_b > / N_e
			
			Calculated for q = (0, qy), which is good enough if near-rotationally invariant.
			Removes any part associated with long-range ordered CDW.
			
			Assumes correlations important out to xi in real-space
			
			Returns pair of 1D arrays S, qy, with qy > 0
			
			Algorithm:
			
			
				kappa = 2 pi / L
				rho_{mu, a}(0, qy) =  F_{mu a; mu a}(0, qy) \sum_x exp(-i kappa qy x ) n_{mu, a, x}  ( x in Ints)
				
				S(qy) = <rho_{mu, a}(0, -qy) rho_{mu, a}(0, qy)> / N_e = | F_{mu a; mu a}(0, qy) |^2 \sum_{x, y} exp(-i kappa qy (y - x) )  < n_{mu, a, x} n_{mu, a, y}> / nu
			
			
		"""
		Nlayer = self.Nlayer
		if nu==None: nu=mu
		if b==None: b=a
		l1 = self.species[mu][a] #which orbital
		l2= self.species[nu][b]
		
		Lx = self.Lx
		max = int(2*xi*Lx/(2.*np.pi))
		oldmax=xi
		G = np.zeros(max, dtype = np.float)
		
		n1 = psi.site_expectation_value(self.nOp, range(l1, psi.L, Nlayer)).real #densities
		n2 = psi.site_expectation_value(self.nOp, range(l2, psi.L, Nlayer)).real #densities
		#n = n - np.mean(n)
		n1 = np.tile(n1, max + psi.L/Nlayer)
		n2 = np.tile(n2, max + psi.L/Nlayer)
		
		for j in range(l1, psi.L, Nlayer): #average across unit cell
			G+= ( psi.correlation_function(self.nOp, self.nOp, range(j+l2-l1, j+l2-l1+max*Nlayer, Nlayer), [j]).real - n1[j:j+max]*n2[j+l2-l1:j+l2-l1+1] )  #take only connected component

		if b==a and mu==nu: G[0]=-G[0]
		G *= (1.*Nlayer/psi.L)
		if np.abs(G[-1])> 1e-7:
			print "Warning: last Gs", G[-3:]
		if len(G) < Lx*5:
			G = np.pad(G, (0, int(Lx*5 - len(G)) ), mode = 'constant')
			print G
	
		#G = np.hstack([G, G[-1:0:-1]])
		#pad with zeros to ensure spacing of dk < 0.1

		S = np.fft.rfft(G[1:]).real #Returns only positive freq component
		#S = S/np.mean(n1) #structure factor conventionally normalized by Ne, not NPhi
		max = (len(G) + 1)/2
		qy = np.arange(max, dtype = np.float)/(2*max-1)*Lx
		
		S*=self.bands[mu].pre_F(a, a)*self.bands[mu].pre_F(b, b)  #Form factor
		S*=self.bands[mu].F(a, a, 0, qy)*self.bands[nu].F(b, b, 0, -qy)
		
		return S, qy

	def density_density(self,psi,mu,a,nu=None,b=None,xi=30):
		"""computes sum_{x1,x2} rho(x1,y1)rho(x2,y2), which is just the Fourier transform of the structure factor (i think)
			can also be computed a different way, as sum_{n,m} e^{-(y-kappa(n-m))^2} <rho(n)rho(m)>"""
		Nlayer = self.Nlayer
		if nu==None: nu=mu
		if b==None: b=a
		l1 = self.species[mu][a] #which orbital
		l2= self.species[nu][b]

		Lx = self.Lx
		max = int(2*xi*Lx/(2.*np.pi))
		G = np.zeros(max, dtype = np.float)

		for j in range(l1, psi.L, Nlayer): #average across unit cell
			G+= ( psi.correlation_function(self.nOp, self.nOp, range(j+l2-l1, j+l2-l1+max*Nlayer, Nlayer), [j]).real )
	
		G *= (1.*Nlayer/psi.L)
		if len(G) < Lx*5:
			G = np.pad(G, (0, int(Lx*5 - len(G)) ), mode = 'constant')
			print G

		out = np.zeros(max, dtype = np.float)
		for j in range(0,max*Nlayer,Nlayer):
			out[j]=np.sum([ (np.exp( -0.5*(2*np.pi/Lx)**2*(j-k)**2 )+np.exp( -0.5*(2*np.pi/Lx)**2*(j+k)**2 ) )*G[k] for k in range(0,max*Nlayer,Nlayer)])
		return out		

	def structure_factor_real(self,psi,mu,a,nu=None,b=None,xi=30):
		"""Fourier transforms the structure factor from the previous function to get a real-space density correlator"""
		from scipy.special import jn
			
		S,qy=self.structure_factor(psi,mu,a,nu=nu,b=b,xi=xi)
		rs=np.arange(0,10,0.1)
		out=jn(0,np.outer(rs,qy))
		out=np.tensordot(out,qy*S,(1,0))
		cons=(1./(len(qy)*2.-1.))*2*np.pi*self.Lx/4./np.pi
		out=cons*out+1/9.
		return rs,out

	def structure_factor_kx(self,psi,mu,a,nu=None,b=None,xi=20):
		"""Same as structure factor, but it works for a non-zero k_x, and it only works for the LLL, and for electrons in the same layer
			right now it calls bond_correlation function so it only works for the smallest non-zero kx, kx=2pi/L"""
		print ""
		Nlayer = self.Nlayer
		if nu==None: nu=mu
		if b==None: b=a
		if a!=0 or b!=0 or mu!=nu:
			print "nonzero kx structure only implemented for LLL and electrons in the same layer"
			raise NotImplementedError
		
		l1 = self.species[mu][a] #which orbital

		Lx = self.Lx
		max = int(2*xi*Lx/(2.*np.pi))
		G1 = np.zeros(max-2, dtype = np.float)
		G2 = np.zeros(max-2, dtype = np.float)
		G0 = 0
		CdaggerC=npc.tensordot(psi.get(self.AOp,0),psi.get(self.aOp,1),0 )
		CCdagger=npc.tensordot(psi.get(self.aOp,0),psi.get(self.AOp,1),0 )
		CdaggerC=CdaggerC.transpose([0,2,1,3])
		CCdagger=CCdagger.transpose([0,2,1,3])

		for j in range(l1, psi.L, Nlayer): #average across unit cell
			G1+=  psi.bond_correlation_function(CdaggerC, CCdagger, range(j+2*Nlayer, j+max*Nlayer, Nlayer), [j]).real 
			#G2+=  psi.bond_correlation_function(CCdagger, CdaggerC, range(j+2*Nlayer, j+max*Nlayer, Nlayer), [j]).real 
			#G0+=psi.correlation_function(self.nOp,self.nOp,[j],[j]).real
		
		G1 *= (1.*Nlayer/psi.L)
		#for s in G1:
		#	print s,'#orbit'
		S=np.fft.rfft(G1).real			
		max=len(G1)
		qy = np.arange(max/2+1, dtype = np.float)/max*Lx
		print len(S),len(qy)
		S*=np.exp(-0.5*(qy**2+(2*np.pi/Lx)**2))
		return S,qy
		
	def particle_hole_order(self,psi,mu,a):
		"""measures the order parameter nnn-(1-n)(1-n)(1-n), which detects the breaking of particle-hole symmetry.  -SG"""
		out=0
		l1=self.species[mu][a]
		Nlayer=self.Nlayer
		##There are a lot of different ways one can think to calculate this
		def n_minus_one_half():
			out=0
			for j in range(psi.L):
				G=psi.correlation_function(self.nOp_shift,self.nOp_shift,[2*Nlayer+j],[j],OpStr=self.nOp_shift)			
				out+=G**2
			return out/(1.*psi.L)

		def nnn_minus_phc():
			out=0
			for j in range(psi.L):
				G=2*psi.correlation_function(self.nOp,self.nOp,[2*Nlayer+j],[j],OpStr=self.nOp)-1
				G+=psi.site_expectation_value(self.nOp,[j])[0]+psi.site_expectation_value(self.nOp,[Nlayer+j])[0]+psi.site_expectation_value(self.nOp,[2*Nlayer+j])[0]	
				G-=psi.correlation_function(self.nOp,self.nOp,[Nlayer+j],[j])+psi.correlation_function(self.nOp,self.nOp,[2*Nlayer+j],[j])
				G-=psi.correlation_function(self.nOp,self.nOp,[2*Nlayer+j],[Nlayer+j])
				out+=G**2
			return out
			
		cordir={}
		def single_nnn(pos):			
			pos=tuple(sorted(pos))
			if pos in cordir: return cordir[pos]
			else:
				if pos[0]==pos[1]:
					temp=psi.site_expectation_value(self.nOp_shift,[pos[2]])[0]
				elif pos[1]==pos[2]:	
					temp=psi.site_expectation_value(self.nOp_shift,[pos[0]])[0]
				else:	
					LEnv=enviro.make_env(psi)
					LEnv.initLE_site_Op(pos[0],self.nOp_shift)
					LEnv.updateLE_all_seq(pos[1]-pos[0]-1)
					LEnv.updateLE_all_by_one_site(self.nOp_shift)
					LEnv.updateLE_all_seq(pos[2]-pos[1]-1)
					REnv=enviro.make_env(psi)
					REnv.initRE_site_Op(pos[2],self.nOp_shift)
					temp=enviro.contract(LEnv,REnv,bonds=pos[2]-1)[0]				
				cordir[pos]=temp
				#print pos,temp
				return temp

		def single_nnn_minus_phc(pos):			
			pos=tuple(sorted(pos))
			if pos in cordir: return cordir[pos]
			else:
				if pos[0]==pos[1] or pos[1]==pos[2]:
					temp=0#np.sum(psi.site_expectation_value(self.nOp,[pos[0],pos[2]]))-1
				else:	
					temp=2*psi.correlation_function(self.nOp,self.nOp,pos[2],pos[0],OpStr=[self.Id]*(pos[1]-pos[0]-1)+[self.nOp]+[self.Id]*(pos[2]-pos[1]-1) )-1
					temp+=np.sum(psi.site_expectation_value(self.nOp,pos))
					temp-=psi.correlation_function(self.nOp,self.nOp,[pos[2]],[pos[0]])+psi.correlation_function(self.nOp,self.nOp,[pos[0]],[pos[1]])
					temp-=psi.correlation_function(self.nOp,self.nOp,[pos[2]],[pos[1]])
				cordir[pos]=temp
				return temp

		def derivs_y():	
			#calculates rho(0) (d/dy)^2 rho(0) (d/dy)^4 rho(0)
			if Nlayer>1: raise NotImplementedError 
			pl=2*np.pi/self.Lx
			expcutoff=1e-5; ncutoff=20			
			temp= np.sum([np.exp( -pl**2*(n1**2+n2**2+n3**2) )*(4*pl**2*n2**2-2)*(16*pl**4*n3**4-48*pl**2*n3**2+12)*(single_nnn((n1,n2,n3))+single_nnn((n1+1,n2+1,n3+1)) ) \
				for n1 in range(-ncutoff,ncutoff+1) for n2 in range(-ncutoff,ncutoff+1) for n3 in range(-ncutoff,ncutoff+1) if abs(np.exp(-(n1**2+n2**2+n3**2)*pl**2))>expcutoff])
			return temp

		return n_minus_one_half()

	def three_body_local(self,psi,mu,a,cut=10):
		"""computes rho(r) (d/dy)^2 rho(r) (d/dy)^4 rho(r) 
		given by F*\sum_{M,N,lam,mu,nu} delta(M+N-lam-mu-nu) c^dag_0 c^dag_M c^dag_N c_lam c_mu c_nu 
		where F=M*N*(M-N)*(mu-nu)*(lam-mu)*(nu-lam)*exp(-(2pi/L)^2/2*[(sum X^2)-1/6.*(sum X)^2], 
		where X runs overall indices M,N,lam,mu,nu
		This is computed for everywhere in the unit cell, and then such things are summed from -cut to 3*cut, with the position kept track of in the argument of the exponent
		"""
		from collections import OrderedDict
		Nlayer=self.Nlayer
		p=self.species[mu][a]
		assert psi.bc=='periodic'; assert psi.L%Nlayer==0
		fluxl=np.arange(0,psi.L,Nlayer)
		LEnv=enviro.make_env(psi)
		REnv=enviro.make_env(psi)
		out=0
		piL=2.*np.pi/self.Lx
		#loop over all values of M,N,mu,nu,lam, allowing no duplicates and keeping 0<M<N, mu<nu<lam
		for M in range(1,cut+1): 
			for N in range(M+1,cut+1):
				for mu in range(-cut,cut+1):
					for nu in range(mu+1,cut+1):
						lam=M+N-mu-nu
						if lam>cut or lam<=nu: continue
						permute_sign=1
						#as long as 0<M<N and mu<nu<lam, our signs are almost right. The only subtlety is sometimes we have to 
						#move a c and c^dagger together to make a number operator. How many - signs do we pick up doing this?
						#by enumerating all cases one can show that each time 0==nu,M==mu,M=lam,N=nu one - sign is added
						if(nu==0 or nu==N): permute_sign*=-1;
						if(M==mu or M==lam): permute_sign*=-1; 

						#we now have a list of operators and their positions. Sort this list
						terms=[(0,self.AOp),(M,self.AOp),(N,self.AOp),(mu,self.aOp),(nu,self.aOp),(lam,self.aOp)]
						terms=sorted(terms,key=lambda t:t[0])
						newterms=[]
						i=0
						#combine c^dag c into n-1/2 if they are on the same site
						while i<len(terms)-1:
							if terms[i][0]==terms[i+1][0]:
								assert terms[i][1]!=terms[i+1][1]
								newterms.append( (terms[i][0],self.nOp_shift))
								i+=2
							else: 
								newterms.append(terms[i])
								i+=1
						if i==len(terms)-1: newterms.append(terms[i])

#print list of operators
#						for term in newterms:
#							print term[0],
#							if term[1]==self.AOp: print 'A'
#							elif term[1]==self.aOp: print 'a'
#							elif term[1]==self.nOp: print 'n'
#							else: print 'error'		

						#initialize left environment with first operator
						LEnv.initLE_site_Op(fluxl+p+newterms[0][0]*Nlayer,newterms[0][1])#do I need to mod into the unit cell?
						if newterms[0][1]==self.nOp_shift: maybeStrop=self.Id #keep track of if we need to start putting in string operators
						else: maybeStrop=self.StrOp
						#loop through elements of operator list, applying them in the right place
						for i in range(1,len(newterms)-1):
							LEnv.updateLE_all_seq((newterms[i][0]-newterms[i-1][0]-1)*Nlayer*[maybeStrop]) 
							LEnv.updateLE_all_by_one_site(newterms[i][1])
							if newterms[i][1]!=self.nOp_shift:
								if maybeStrop==self.Id: maybeStrop=self.StrOp
								else: maybeStrop=self.Id
						LEnv.updateLE_all_seq((newterms[-1][0]-newterms[-2][0]-1)*Nlayer*[maybeStrop])
						#put on last operator
						REnv.initRE_site_Op(fluxl+p+newterms[-1][0]*Nlayer,newterms[-1][1])

						#multiply by prefactor and add onto out
						val=enviro.contract(LEnv,REnv,fluxl+p+newterms[-1][0]*Nlayer)
#						print val[0],val[1]
						prefactor=permute_sign*piL**6*M*N*(M-N)*(lam-mu)*(mu-nu)*(nu-lam)
						sumx=M+N+mu+nu+lam
						prefactor*=np.exp(piL**2*sumx**2/12.)*np.exp(-0.5*piL**2*(M**2+N**2+mu**2+nu**2+lam**2)) 
						out+=prefactor*np.sum(val)
						#print M,N,mu,nu,lam,prefactor*np.sum(val)
		return out

	def three_body_local_test(self,psi,species,a,cut=10):
		"""computes rho(r) (d/dy)^2 rho(r) (d/dy)^4 rho(r) 
		given by F*\sum_{M,N,lam,mu,nu} delta(M+N-lam-mu-nu) c^dag_0 c^dag_M c^dag_N c_lam c_mu c_nu 
		where F=M*N*(M-N)*(mu-nu)*(lam-mu)*(nu-lam)*exp(-(2pi/L)^2/2*[(sum X^2)-1/6.*(sum X)^2], 
		where X runs overall indices M,N,lam,mu,nu
		This is computed for everywhere in the unit cell, and then such things are summed from -cut to 3*cut, with the position kept track of in the argument of the exponent
		It seems that (at least for the CFL), we need cut>20. This leads to 20^4 terms that need to be calculated. The naive way to do this would require 10^5 MPS contractions
		But that naive way would duplicate a lot of effort, since there are many terms whose first part is the same
		To be more efficient, we construct a tree, each node of which contains an operator and one site it goes on
		Each element of the sum is a distinct leaf of the tree, and we can save all the environments on the path to that leaf, and then reuse them for similar leaves
		"""

		#part 1: build the tree
		from collections import OrderedDict
		from copy import deepcopy
		class Node:
			def __init__(self,data):
				self.data=data
				self.children=[]
				self.sign=0
			def add_child(self,obj):
				self.children.append(obj)
			def __repr__(self, level=0):
				ret = "\t"*level+repr(self.data)+"\n"
				for child in self.children:
					ret += child.__repr__(level+1)
				return ret
			
			
		root=Node(None)
		node_list=root
		#loop over all values of M,N,mu,nu,lam, allowing no duplicates and keeping 0<M<N, mu<nu<lam
		for M in range(1,cut+1): 
			for N in range(M+1,cut+1):
				for mu in range(-cut,cut+1):
					for nu in range(mu+1,cut+1):
						lam=M+N-mu-nu
						if lam>cut or lam<=nu: continue
						permute_sign=1
						#as long as 0<M<N and mu<nu<lam, our signs are almost right. The only subtlety is sometimes we have to 
						#move a c and c^dagger together to make a number operator. How many - signs do we pick up doing this?
						#by enumerating all cases one can show that each time 0==nu,M==mu,M=lam,N=nu one - sign is added
						if(nu==0 or nu==N): permute_sign*=-1;
						if(M==mu or M==lam): permute_sign*=-1; 

						#we now have a list of operators and their positions. Sort this list
						terms=[(0,self.AOp),(M,self.AOp),(N,self.AOp),(mu,self.aOp),(nu,self.aOp),(lam,self.aOp)]
						terms=sorted(terms,key=lambda t:t[0])
						newterms=[]
						i=0
						#combine c^dag c into n-1/2 if they are on the same site
						while i<len(terms)-1:
							if terms[i][0]==terms[i+1][0]:
								assert terms[i][1]!=terms[i+1][1]
								newterms.append( (terms[i][0],self.nOp_shift))
								i+=2
							else: 
								newterms.append(terms[i])
								i+=1
						if i==len(terms)-1: newterms.append(terms[i])
				
						#add this element to the tree
						current_node=root
						for term in newterms:
							#search this level to try to find matching node
							found=False
							for node in current_node.children:
								if node.data==term:
									current_node=node
									found=True
									break
					
							if(not found):
								#if we didn't find a matching node, have to create a new one
								temp=Node(term)
								current_node.add_child(temp)
								current_node=temp		
						current_node.sign=permute_sign							

		#part 2: traverse the tree to construct all the different elements
		Nlayer=self.Nlayer
		p=self.species[species][a]
		assert psi.bc=='periodic'; assert psi.L%Nlayer==0
		fluxl=np.arange(0,psi.L,Nlayer)
		copy_env=enviro.make_env(psi)
		REnv=enviro.make_env(psi)
		piL=2.*np.pi/self.Lx
		def traverse(current_node,env=None,sites=([],[]),strop=self.Id,oldsite=None):
			if current_node.data!=None:
				copy_sites=deepcopy(sites)
				if(current_node.data[1]==self.nOp_shift or current_node.data[1]==self.AOp): copy_sites[0].append(current_node.data[0])
				if(current_node.data[1]==self.nOp_shift or current_node.data[1]==self.aOp): copy_sites[1].append(current_node.data[0])
				if len(current_node.children)==0: #we are at the end of the tree
					#make right environment
					copy_env=env.shallow_copy()
					copy_env.updateLE_all_seq([strop]*(current_node.data[0]-oldsite-1)*Nlayer)
					REnv.initRE_site_Op(fluxl+p+current_node.data[0]*Nlayer,current_node.data[1])
					#contract
					val=enviro.contract(copy_env,REnv,fluxl+p+current_node.data[0]*Nlayer)
					#compute prefactor
					prefactor=piL**6*current_node.sign;
					sumx=0
					for i in range(2):
						prefactor*=(copy_sites[i][0]-copy_sites[i][1])*(copy_sites[i][1]-copy_sites[i][2])*(copy_sites[i][2]-copy_sites[i][0])
						for j in range(3):
							prefactor*=np.exp(-0.5*piL**2*copy_sites[i][j]**2)
							sumx+=copy_sites[i][j]
					prefactor*=np.exp(piL**2*sumx**2/12.)
				#	print copy_sites[0],copy_sites[1],prefactor*np.sum(val)
					return prefactor*np.sum(val)
				else:
					if oldsite==None:
						#copy_env=[current_node.data]
						copy_env=env.shallow_copy()
						copy_env.initLE_site_Op(fluxl+p+current_node.data[0]*Nlayer,current_node.data[1])#do I need to mod into the unit cell?
					else:
						#deep copy env
						copy_env=env.shallow_copy()
						copy_env.updateLE_all_seq([strop]*(current_node.data[0]-oldsite-1)*Nlayer)
						#update env
						copy_env.updateLE_all_by_one_site(current_node.data[1])
					if(current_node.data[1]==self.AOp or current_node.data[1]==self.aOp): 
						if(strop==self.Id): strop=self.StrOp
						else: strop=self.Id
					out=0
					for child in current_node.children:
						out+=traverse(child,copy_env,copy_sites,strop,current_node.data[0])
					return out
			else:
				out=0
				for child in current_node.children:
					out+=traverse(child,env,sites,strop,oldsite)
				return out
									

		return traverse(root,copy_env)	

	def interpret_two_body_correlator(self, psi, max_m=2, x_limit=10, derivatives=[1], filename='structure', predict=False, verbose=0, pred_order=10, pred_f=5, pred_skips=10):
		"""takes the output of two_body_correlator_pbc for a single species and 
			- (optional) performs linear prediction to extend the data to larger k
			- fourier transforms
			- computes derivative fourier transforms based on the input field 'derivatives'
			xlimit- number of sites to compute out to, this gets translated into max_k 
			the pred_* parameters are used in the linear prediction only
			"""

		#compute real-space correlators
		cordict=self.two_body_correlator_pbc(psi, max_m=max_m, max_k=int(2*x_limit*self.Lx/(2.*np.pi)), verbose=verbose)
		corfunc=np.mean(cordict['C'][0,0,0,0,:,:,:],0)
		corfunc=corfunc[:,corfunc.shape[1]//2+1:]

		#print for testing purposes
		if verbose>0:	
			for i in range(corfunc.shape[1]):
				print corfunc[0,i],"#raw_real"

	
		#perform linear prediction
		if(predict):
			from algorithms.linalg import lpc
			n=corfunc.shape[1]
			predicted=np.hstack([corfunc,np.zeros((max_m+1,pred_f*n))])

			for m in range(max_m+1):
				if np.sum(np.abs(np.diff(predicted[m,:n]))) < 10**(-8):
					predicted[m,:] = predicted[m,0]
				else: 		
					c = lpc.lpc_ref(predicted[m,pred_skips:n],pred_order)
					for i in range(n*pred_f):
						predicted[m,n+i] = np.sum(-c[-1:0:-1]*predicted[m,n+i-pred_order:n+i])

			if verbose>0:
				for i in range(predicted.shape[1]):
					print predicted[0,i],"#raw_pred"

		else:
			predicted=corfunc

		#fourier transform
		xi_adjusted=predicted.shape[1]/(1.*self.Lx)
		for m in range(max_m+1):
			outfile=open(filename+str(m),'w')
			print >>outfile,'#0',
			for eta in derivatives: print >>outfile,eta,   #print header with a list of what derivative is what
			print >>outfile
		 	temp=np.fft.rfft(predicted[m,:]).real
			S0=np.zeros((len(derivatives)+1,len(temp)))
			S0[0,:]=temp
			for i,eta in enumerate(derivatives):
				S0[i+1,:]=np.fft.rfft(np.arange(predicted.shape[1])**eta*predicted[m,:]).real
			for i in range(S0.shape[1]):
				print >>outfile,i/xi_adjusted,			
				for j in range(S0.shape[0]): print >>outfile,S0[j,i],
				print >>outfile	

	def two_body_correlator_pbc(self, psi, max_m=2, max_k=4, remove_disconnected_part=True, compute_negative_m=False, verbose=0):
		r"""Calculates a bunch of 2-body (4 operators) correlators.

	Computes the expectation values C[a,b,c,d ; x,m,k] defined to be
		pd_{x+k,a} pd_{x+m,c} p_{x,d} p_{x+m+k,b}
	where
		p = \psi (annihilation) and pd = \psi^\dag (creation),
		m goes from 0 to max_m,
		k goes from -max_k to +max_k,
		(a,b,c,d) label the layer indicies (0 to Nlayer-1),
		x labels each flux in the unit cell of psi (0 to Nflux, where Nflux = psi.L / Nlayer).
	
	If remove_disconnected_part is True, compute the normal-ordered connected-correlator between:
		(pd_{x+m,c} p_{x,d})  and  (pd_{x+k,a} p_{x+m+k, b})
		I.e., subtract <pd_{x+m,c} p_{x,d}> <pd_{x+k,a} p_{x+m+k, b}> from the piece above.
	
	If compute_negative_m is True, then the range of m goes from -max_m to +max_m.
	
	Returns a dictionary with keys:
		'C':         the (total or connected) correlator;
		             a 7-array with indicies [a,b,c,d,x,m,k] and shape (Nlayer, Nlayer, Nlayer, Nlayer, Nflux, m_offset+max_m+1, 2*k_offset+1).
		'm_offset':  gives the location of m=0 for the m-index (leg #5).
		'k_offset':  gives the location of k=0 for the k-index (leg #6).
		'1body':     the 1-body expectation values <pd_{x+m,c} p_{x,d}>;
		             a 4-array with indicies [c,d,x,m] and shape (Nlayer, Nlayer, Nflux, max_m+1),
		'time':      computation time (in seconds).
	
	Notes:
	-	psi must have periodic boundary conditions.
	-	The m = k = 0 cases are computed separately, and probably works (have not been tested yet).
	-	Only non-negative m's are computed.  The correlator with negative m are related via:
			C[a,b,c,d ; x,m,k] = C[b,a,d,c ; x+m,-m,k]*
	-	The (total) correlators also obey the following symmetries
			C[a,b,c,d ; x,m,k] = -C[c,b,a,d ; x,k,m]		(minus sign for fermions)
			C[a,b,c,d ; x,m,k] = C[d,c,b,a ; x+k,m,-k]*
			C[a,b,c,d ; x,m,k] = C[c,d,a,b ; x+m+k,-m,-k]
	-	While the code doesn't require momentum to be conserved, without this assumption there would be
		a lot more nonzero 2-body correlators not captured in C[a,b,c,d ; x,m,k].
	"""
		t0 = time.time()
		assert psi.bc == 'periodic'
		assert type(max_m) == int and max_m >= 0
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		N = self.Nlayer
		assert psi.L % N == 0
		Nflux = psi.L // N
		if max_k < max_m + 1: max_k = max_m + 1
		sgn = -1		# for fermions
		kOS = max_k		# k_offset
		mOS = max_m if compute_negative_m else 0
		C = np.zeros((N, N, N, N, Nflux, mOS + 1 + max_m, 2 * kOS + 1), dtype=psi.dtype)
		one_body = np.zeros((N, N, Nflux, max_m+1), dtype=psi.dtype)
		for m in range(max_m + 1):
			Dm = self.two_body_correlator_pbc_m(psi, m, max_l = max_k-m, verbose = verbose-1)
			if m == 0:
				m0_sym_error = np.max(np.abs(Dm['Cm'][:,:,:,:,:,1:] - Dm['Cm'][:,:,:,:,:,1:].transpose(1,0,3,2,4,5)))
				if m0_sym_error > 1e-14: print "Warning!  two_body_correlator_pbc(): m0_sym_error too large: " + str(m0_sym_error)
			##	m <= k <= max_k:  Use l = k - m
			C[:, :, :, :, :, mOS+m, kOS+m:kOS+max_k+1] = Dm['Cm']
			##	0 <= k < m:  Use C[a,b,c,d ; x,m,k] = -C[c,b,a,d ; x,k,m]
			C[:, :, :, :, :, mOS+m, kOS:kOS+m] = sgn * C[:, :, :, :, :, mOS:mOS+m, kOS+m].transpose(2,1,0,3,4,5)
			##	-max_k <= k < 0:  Use C[a,b,c,d ; x,m,k] = C[d,c,b,a ; x+k,m,-k]
			for k in range(-max_k, 0):
				C[:, :, :, :, :, mOS+m, kOS+k] = np.roll(C[:, :, :, :, :, mOS+m, kOS-k], -k, axis=4).transpose(3,2,1,0,4).conj()
			##	For -m <= k < 0, we could have also used C[a,b,c,d ; x,m,k] = -C[b,c,d,a ; x-|k|,|k|,m]*
			one_body[:, :, :, m] = Dm['1body']
		
		if remove_disconnected_part:
			for k in range(-max_k, max_k + 1):
				##	Subtract <pd_{x+m,c} p_{x,d}> <pd_{x+k,a} p_{x+m+k, b}>
				##	one_body[c,d,x,m] stores <pd_{x+m,c} p_{x,d}>
				##	hence one_body[b,a,x+k,m]* = <pd_{x+k,a} p_{x+k+m,b}>
				X_ab = np.roll(one_body[:, :, :, :], -k, axis=2).conj()
				X_ab = np.tensordot(X_ab, np.ones((N, N), dtype=np.float), axes=0).transpose(0,1,4,5,2,3)
				X_cd = np.tensordot(np.ones((N, N), dtype=np.float), one_body, axes=0)
				C[:, :, :, :, :, mOS:, kOS+k] -= X_ab * X_cd
		
		if compute_negative_m:
			for m in range(-max_m, 0):		# negative m's
				C[:,:,:,:, :, mOS+m, :] = np.roll(C[:,:,:,:, :, mOS-m, :], -m, axis=4).transpose(1,0,3,2,4,5).conj()
		
		#C[:, :, :, :, :, mOS, kOS] = 0.			# elements m = k = 0
		D = {'C': C, 'k_offset': kOS, 'm_offset': mOS, '1body': one_body, 'time': time.time()-t0}
		return D


	def two_body_correlator_pbc_m(self, psi, m, max_l=2, verbose=0):
		r"""Calculates a bunch of 2-body (4 operators) correlators, for a specific m (~qx).

	Computes the expectation values of:
		pd_{x+m+l,a} pd_{x+m,c} p_{x,d} p_{x+2m+l,b}
	where p = \psi (annihilation), pd = \psi^\dag (creation).
		l goes from 0 to max_l, and (a,b,c,d) label the layer indicies (0 to Nlayer-1).
		x labels each flux in the unit cell of psi (0 to Nflux, where Nflux = psi.L / Nlayer).
		m must be non-negative.
	
	Returns a dictionary with keys:
		'Cm':     6-array with indicies [a,b,c,d,x,l] and shape (Nlayer, Nlayer, Nlayer, Nlayer, Nflux, max_l+1).
		'1body':  Expectation value of < pd_{x+m,c} p_{x,d} >, in a 3-array indexed by [c,d,x].
		'm':      m.
		'time':   computation time (in seconds).
	
	Notes:
	-	l here is k-m in the usual notation (see two_body_correlator_pbc docstring).
	-	Does not compute m = l = 0 case, those elements are set to np.nan.
	"""
		assert psi.bc == 'periodic'
		assert type(m) == int and m >= 0
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		t0 = time.time()
		N = self.Nlayer
		psiL = psi.L
		assert psiL % N == 0
		Nflux = psiL // N
		corr = np.empty((N, N, N, N, Nflux, max_l+1), dtype=psi.dtype)
		corr.fill(np.nan)		# prefill with nan's
		LEnv = np.zeros((N, N), dtype=object); LEnv.fill(None)		# indices c,d
		REnv = np.zeros((N, N), dtype=object); REnv.fill(None)		# indices a,b
		range_NN = [ pq for pq in itertools.product(range(N), range(N)) ]
		fluxl = np.arange(0, psiL, N)
		##	Construct the starting left/right environments with operators pd_c p_d / pd_a p_b
		##	Later, we will update the environments (with Id) to put some distance between the left/right operators
		if m == 0:
			corr[:,:,:,:,:,0] = self.two_body_correlator_pbc_ml0(psi, verbose=verbose)		# the m = l = 0 case.
			##	Case a > b, c > d
			for p in range(1, N):
				for q in range(p):		# p > q
					RE = REnv[p, q] = enviro.make_env(psi)
					RE.initRE_site_Op(fluxl + p, self.AOp)
					for jj in range(q + 1, p):
						RE.updateRE_all_by_one_site(self.StrOp)
					RE.updateRE_all_by_one_site(self.aOp)
					for jj in range(q):
						RE.updateRE_all_by_one_site()
					LE = LEnv[p, q] = enviro.make_env(psi)
					LE.initLE_site_Op(fluxl + q, self.aOp)
					for jj in range(q + 1, p):
						LE.updateLE_all_by_one_site(self.StrOp)
					LE.updateLE_all_by_one_site(self.AOp)
					for jj in range(p + 1, N):
						LE.updateLE_all_by_one_site()
			##	Case a = b, c = d
			for l in range(N):
				LE = LEnv[l, l] = enviro.make_env(psi)
				LE.initLE_site_Op(fluxl + l, self.nOp)
				LE.updateLE_all_seq(Op_list = N - l - 1)		# up to bond N-1
				RE = REnv[l, l] = enviro.make_env(psi)
				RE.initRE_site_Op(fluxl + l, self.nOp)
				RE.updateRE_all_seq(Op_list = l)		# up to bond -1
			##	Note: We don't have to make all the REnv's
			##	- we have the symmetry Cm[:,:,:,:,:,1:] = Cm[:,:,:,:,:,1:].transpose(1,0,3,2,4,5)
			##	Case a < b, c < d
			for q in range(1, N):
				for p in range(q):		# p < q
					RE = REnv[p, q] = enviro.make_env(psi)
					RE.initRE_site_Op(fluxl + q, self.aOp)
					RE.updateRE_all_seq(Op_list = (q - p - 1) * [self.StrOp])
					#for jj in range(p + 1, q):
					#	RE.updateRE_all_by_one_site(self.StrOp)
					RE.updateRE_all_by_one_site(self.AOp)		# @ p
					RE.updateRE_all_seq(Op_list = p)		# up to bond -1
					LE = LEnv[p, q] = enviro.make_env(psi)
					LE.initLE_site_Op(fluxl + p, self.AOp)
					#for jj in range(p + 1, q):
					#	LE.updateLE_all_by_one_site(self.StrOp)
					LE.updateLE_all_seq(Op_list = (q - p - 1) * [self.StrOp])
					LE.updateLE_all_by_one_site(self.aOp)		# @ q
					LE.updateLE_all_seq(Op_list = N - q - 1)		# up to bond N-1
			bonds_to_contract = np.arange(N - 1, psiL, N)
		
		else:		# m > 0 case
			###SG: I commented out this assert, and added code at line 1252 and line 1259
			assert self.p_type == 'F' or (self.p_type=='B' and self.LL_mixing==False)
			##	Slap the first operator on (aOp @ q), and the second operator (AOp @ p)
			for p,q in range_NN:
				LE = LEnv[p, q] = enviro.make_env(psi)
				LE.initLE(fluxl + q - 1)
				LE.updateLE_all_by_one_site(self.aOp)
				for jj in range(q + 1, m * N + p):
					LE.updateLE_all_by_one_site(self.StrOp)
				LE.updateLE_all_by_one_site(self.AOp)
				RE = REnv[p, q] = enviro.make_env(psi)
				RE.initRE(fluxl + q)
				RE.updateRE_all_by_one_site(self.aOp)
				for jj in range(p + 1, m * N + q):
					RE.updateRE_all_by_one_site(self.StrOp)
				RE.updateRE_all_by_one_site(self.AOp)
			##	Compute for the l = 0 case
			##	the operators are:  p_d, pd_c, pd_a, p_b  (@ x, x+m, x+m, x+2m respectively)
			for a in range(N):
				bonds_to_contract = np.mod(fluxl + a - 1 + m*N, psiL)
				for c in range(a):		# a > c
					for d in range(N):
						for b in range(N):
							corr[a,b,c,d,:,0] = enviro.contract(LEnv[c,d], REnv[a,b], bonds=bonds_to_contract, verbose=0)							
							if self.p_type=='F': corr[c,b,a,d,:,0] = -corr[a,b,c,d,:,0]
							else: corr[c,b,a,d,:,0] = corr[a,b,c,d,:,0]
						LEnv[c, d].updateLE_all_by_one_site()	# this gets called for every (a) larger than (c) = N-1-a times.
				for b in range(N):
					for d in range(N):
						if(b!=d or b!=a): corr[a,b,a,d,:,0] = 0 #this operator preserves layer
						else:
							if self.p_type=='F' : corr[a,b,a,b,:,0] = 0		# a = c: the psi^dag's overlap
							else: #boson case
								LE2=enviro.make_env(psi)
								LE2.initLE(fluxl+q-1)
								LE2.updateLE_all_by_one_site(self.aOp)
								for jj in range(q + 1, m * N + p):
									LE2.updateLE_all_by_one_site(self.StrOp)
								LE2.updateLE_all_by_one_site(self.AAOp)
								RE2 = enviro.make_env(psi)
								RE2.initRE(fluxl + q)
								RE2.updateRE_all_by_one_site(self.aOp)
								for jj in range(p + 1, m * N + q):
									RE2.updateRE_all_by_one_site(self.StrOp)
								corr[a,b,a,b,:,0] = enviro.contract(LE2, RE2, bonds=bonds_to_contract, verbose=0)							
								
							
			##	Prepare for the l > 0 cases
			for a,b in range_NN:
				REnv[a, b].updateRE_all_seq(Op_list = a)
			bonds_to_contract = np.mod(np.arange(N-1, psiL, N) + m*N, psiL)

		##	Currently, LEnv[c, d].getLE(bonds_to_contract[x]) stores pd_{x+m,c} p_{x,d}
		##	  REnv[a, b].getRE(N * x - 1) stores pd_{x,a} p_{x+m,b}
		if verbose >= 2:
			for c,d in range_NN:
				if LEnv[c, d] is None: print '  LEnv', (c,d)
				if LEnv[c, d] is not None: print '  LEnv', (c,d), LEnv[c,d].str_age_array()[0]
			for a,b in range_NN:
				if REnv[a, b] is None: print '  REnv', (a,b)
				if REnv[a, b] is not None: print '  REnv', (a,b), REnv[a,b].str_age_array()[1]
			print 'bonds_to_contract', bonds_to_contract

		##	Check the environments are of the right age
		from tools.math import assert_np_identical
		for p,q in range_NN:
			if REnv[p, q] is not None:
				assert_np_identical( REnv[p, q].age[1], np.tile([-1]*(N-1) + [max(p,q+m*N) + 1], Nflux), name='REnv[m={},{},{}] ages'.format(m,p,q) )
			if LEnv[p, q] is not None:
				assert_np_identical( LEnv[p, q].age[0], np.tile([-1]*(N-1) + [(m+1)*N - min(q,p+m*N)], Nflux), name='LEnv[m={},{},{}] ages'.format(m,p,q) )

		if verbose >= 1: print "Taking correlation functions (m = {})".format(m),
		##	Compute the 1-body terms:  <pd_{x+m,c} p_{x,d}>
		one_body = np.zeros((N, N, Nflux), dtype=psi.dtype)
		for c,d in range_NN:
			if LEnv[c, d] is None: continue
			one_body[c, d] = LEnv[c, d].contract_LE(bonds_to_contract)
		##	Contract the left/right environments while extending the REnv's to get the 2-body terms.
		for l in range(1, max_l + 1):
			if l > 1:
				for e in REnv.reshape(-1):
					if e is not None: e.updateRE_all_seq(Op_list = N)		# update by N sites
			if verbose >= 1: sys.stdout.write('.')
			for a,b in range_NN:
				if REnv[a, b] is None: continue
				for c,d in range_NN:
					if LEnv[c, d] is None: continue
					corr[a,b,c,d,:,l] = enviro.contract(LEnv[c,d], REnv[a,b], bonds=bonds_to_contract, verbose=0)
		t = time.time() - t0
		if verbose >= 1: print "({} seconds)".format(t)
		return {'m': m, 'Cm': corr, '1body': one_body, 'time': t}


	def two_body_correlator_pbc_ml0(self, psi, verbose=0):
		r"""Helper function for two_body_correlator_pbc_m above, specifically for the case m = l = 0.
	That is, pd_{x,a} pd_{x,c} p_{x,d} p_{x,b} .
	Returns a 5-array C[a,b,c,d,x], with shape (Nlayer, Nlayer, Nlayer, Nlayer, Nflux).
	"""
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		N = self.Nlayer
		psiL = psi.L
		assert psi.bc == 'periodic' and psiL % N == 0
		Nflux = psiL // N
		fluxl = np.arange(0, psiL, N)
		corr = np.empty((N, N, N, N, Nflux), dtype=psi.dtype)
		corr.fill(np.nan)		# prefill with nan's
		##	Only compute b > d, a > c for fermions.  With this restriction, there are no signs from fermion -> bosons + string
		###SG: removed this assertion, added sign at 1376 and computed a=c, b=d terms
		assert self.p_type == 'F' or (self.p_type=='B' and self.LL_mixing==False)
		if self.p_type=='F': sgn=-1
		else: sgn=1
		oringal_Op = np.array(['A','A','a','a']); acdb = np.array(['a','c','d','b'])
		for a in range(N):
			for b in range(N):
				if self.p_type=='B':
					crange=range(a+1)
					drange=range(b+1)
				else:
					crange=range(a)
					drange=range(b)

				for c in crange:
					for d in drange:
						#print '(a,c,d,b) =', (a,c,d,b)
						loc = np.array([a, c, d, b])
						order = np.argsort(loc)		# find the stable ordering of the operators from smallest (left-most) to largest
						loc = loc[order].tolist(); Op = oringal_Op[order].tolist()
						sign = -1 if ((Op[1] == 'A') ^ (Op[3] == 'A')) else +1
						assert sign == perm_sign(order) or self.p_type=='B'
						#print '\t', Op, acdb[order], '@', loc, perm_sign(order)
						if c==a or d==b:
							if d!=b or d!=a or d!=c or self.p_type=='F': 
								corr[a,b,c,d,:]=corr[c,b,a,d,:]=corr[a,d,c,b,:]=corr[c,d,a,b,:]=0
								continue
							else:
								loc=[a]
								Op=['AAaa']
						else:
							if loc[2] == loc[3]:
								assert Op[2] == 'A' and Op[3] == 'a'
								loc = loc[:3]; Op = Op[:2] + ['n']
							if loc[1] == loc[2]:
								assert Op[1] == 'A' and Op[2] == 'a'
								loc = [loc[0]] + loc[2:]; Op = [Op[0], 'n'] + Op[3:]
							if loc[0] == loc[1]:
								assert Op[0] == 'A' and Op[1] == 'a'
								loc = loc[1:]; Op = ['n'] + Op[2:]
						OpList = [None] * (loc[-1] - loc[0] + 1)
						OpString = [None] * len(OpList)
						par = 0; prevloci = 0
						for i in range(len(Op)):
							loci = loc[i] - loc[0]
							OpList[loci] = getattr(self, Op[i]+'Op')
							OpString[loci] = Op[i]+'Op'			# just for printing purposes
							for s in range(prevloci+1, loci):
								OpList[s] = self.StrOp if par == 1 else self.Id
								OpString[s] = 'StrOp' if par == 1 else 'Id'			# just for printing purposes
							if Op[i] != 'n': par = 1-par
							prevloci = loci
						#print '\t', Op, '@', loc, OpString
						##	If at this point we cache the Op+loc, we would be able to reduce some work.  But alas, no.
						LE = enviro.make_env(psi)
						LE.initLE_site_Op(fluxl + loc[0], OpList[0])
						LE.updateLE_all_seq(Op_list = OpList[1:])
						corr[a,b,c,d,:] = LE.contract_LE(fluxl + loc[-1])
						corr[c,b,a,d,:] = sgn*corr[a,b,c,d,:]
						corr[a,d,c,b,:] = sgn*corr[a,b,c,d,:]
						corr[c,d,a,b,:] =  corr[a,b,c,d,:]
						#for x in range(Nflux):		# The n_site_expectation_value is shit.
						#	#corr[a,b,c,d,x] = psi.n_site_expectation_value(OpString, loc[0]+x*Nflux)
				if self.p_type=='F':
					corr[a,b,a,:,:] = 0		# for a = c
					corr[a,b,:,b,:] = 0		# for b = d
		#print corr.reshape(N**4, Nflux)
		return corr



	def four_body_correlator(self,psi,bR=1,bL=1,cR=0,cL=0,kx=0,muR=0,aR=0,muL=0,aL=0,max_l=10):
		"""a general four-body correlator
		correlators take the form O=C_{a}c_{a+c}c{a+b-c+kx}C{a+b}, and we compute <O(a)O(a')> for all a,a'
		b,c are different between the two O's but the kx is common
		mu are species, a are landau level indices"""
	
		#initialize
		try: iter(bR)
		except:	bR=[bR]
		try: iter(bL)
		except:	bL=[bL]
		try: iter(cR)
		except:	cR=[cR]
		try: iter(cL)
		except:	cL=[cL]

		N=self.Nlayer
		cors=np.zeros((len(bL),len(cL),len(bR),len(cR),psi.L/N,max_l),dtype=np.float)
		fluxl=np.arange(0,psi.L,N)
		bonds_to_contract=np.arange(N-1,psi.L,N)

		LEnv = np.zeros((len(bL), len(cL)), dtype=object); LEnv.fill(None)		# indices c,d
		REnv = np.zeros((len(bR), len(cR)), dtype=object); REnv.fill(None)		# indices a,b		

		#right operators
		r1=self.species[muR][aR]

		for bi,b in enumerate(bR):
			for ci,c in enumerate(cR):
				RE=REnv[bi,ci]=enviro.make_env(psi)
				if(c==kx): RE.initRE_site_Op(fluxl+r1,self.nOp_shift)
				elif c>kx:
					RE.initRE_site_Op(fluxl+r1,self.aOp)
					RE.updateRE_all_seq( (N-1+(c-kx-1)*N)*[self.StrOp] )
					RE.updateRE_all_by_one_site(self.AOp)
				else:
					RE.initRE_site_Op(fluxl+r1,self.AOp)
					RE.updateRE_all_seq( (N-1+(kx-c-1)*N)*[self.StrOp] )
					RE.updateRE_all_by_one_site(self.aOp)
				#work out how many steps to take between parts
				spaces=b-1
				if c>0: spaces-=c
				if c>kx: spaces-=c-kx
				if spaces<0:
					print "bad parameters right",b,c
					RE=None
					continue
				RE.updateRE_all_seq(N-1+spaces*N)	
				if(c==0):
					RE.updateRE_all_by_one_site(self.nOp_shift)
				elif c>0:
					RE.updateRE_all_by_one_site(self.AOp)
					RE.updateRE_all_seq( (N-1+(c-1)*N)*[self.StrOp] )
					RE.updateRE_all_by_one_site(self.aOp)
				else:
					RE.updateRE_all_by_one_site(self.aOp)
					RE.updateRE_all_seq( (N-1+(-c-1)*N)*[self.StrOp] )
					RE.updateRE_all_by_one_site(self.AOp)

		#left operators
		l1=self.species[muL][aL]
		for bi,b in enumerate(bL):
			for ci,c in enumerate(cL):
				LE=LEnv[bi,ci]=enviro.make_env(psi)
				if(c==0): LE.initLE_site_Op(fluxl+l1,self.nOp_shift)
				elif(c>0):
					LE.initLE_site_Op(fluxl+l1,self.AOp)
					LE.updateLE_all_seq( (N-1+(c-1)*N)*[self.StrOp])
					LE.updateLE_all_by_one_site(self.aOp)
				else:
					LE.initLE_site_Op(fluxl+l1,self.aOp)
					LE.updateLE_all_seq( (N-1+(-c-1)*N)*[self.StrOp])
					LE.updateLE_all_by_one_site(self.AOp)
	
				#work out how many spaces to leave between the two parts of the operator		
				spaces=b-1
				if c>0: spaces-=c
				if c>kx: spaces-=c-kx
				if spaces<0:
					print "bad parameters left",b,c
					LE=None
					continue
				LE.updateLE_all_seq(N-1+spaces*N)

				if(c==kx):
					LE.updateLE_all_by_one_site(self.nOp_shift)
				elif c>kx:
					LE.updateLE_all_by_one_site(self.aOp)
					LE.updateLE_all_seq( (N-1+(c-kx-1)*N)*[self.StrOp] )
					LE.updateLE_all_by_one_site(self.AOp)
				else:
					LE.updateLE_all_by_one_site(self.AOp)
					LE.updateLE_all_seq( (N-1+(kx-c-1)*N)*[self.StrOp] )
					LE.updateLE_all_by_one_site(self.aOp)
				LE.updateLE_all_seq(N-1)

		for l in range(1,max_l+1):#l is the number of identities between the correlators +1, maybe a different definition is needed though
			if l>1: 
				for e in REnv.reshape(-1):
					e.updateRE_all_seq(N)
			for bl in range(len(bL)):
				for cl in range(len(cL)):
					for br in range(len(bR)):			
						for cr in range(len(cR)):
							if LEnv[bl,cl] is not None and REnv[br,cr] is not None:
								cors[bl,cl,br,cr,:,l-1]=enviro.contract(LEnv[bl,cl],REnv[br,cr],bonds=bonds_to_contract,verbose=2)

		return cors
		#then convert to 0,y correlators
#		spacing=4
#		real_cors=np.zeros((max_l),dtype=np.float)
#		for y in range(2,max_l+2):
#			for a1 in range(-spacing,spacing+1):
#				for a2 in range(y-spacing,y+spacing+1):
#					real_cors[y]+=np.exp(-a1**2-(a1+1)**2)*np.exp(-(a2-y-1)**2-(a2-y)**2)

		#then fourier transform

	def composite_fermion_correlator0(self,psi,Nflux=2,max_l=2):
		"""inserts Nflux flux and 1 charge in an x-independent way, which can be used to measure composite fermion correlators for the case where q_x=0
		returns results for cf operators l apart, with l in (0,max_l)
		currently only works for single component problems"""
		assert psi.bc== 'periodic'
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		psiL = psi.L
		assert psiL % N == 0
		Nflux = psiL // N
		assert self.Nlayer==1		
		corr=np.empty((Nflux,max_l+1),dtype=np.complex)
		corr.fill(np.nan)
		LEnv=np.zeros(Nflux,dtype=object); LEnv.fill(None)
		REnv=np.zeros(Nflux,dtype=object); REnv.fill(None)
		#original environment for part of system before/after where I am measuring
		cleanEnv=enviro.make_env(psi)
		
		#make an MPS which is identity on virtual bonds and 0 or 1 on physical bonds
		MPSfull=np.zeros(psi.getB(0).shape,dtype=np.complex)
		MPSfull[1,:]=np.eye((psi.getB(0).shape[1:]))
		MPSempty=np.zeros(psi.getB(0).shape,dtype=np.complex)
		MPSempty[0,:]=np.eye((psi.getB(0).shape[1:]))
		#TODO: set charges on these MPS and convert them to npc arrays
		
		shiftedpsi=psi.copy()
		shiftedpsi.translate(Nflux)
		
		for n in range(Nflux):
			LE=LEnv[n]=enviro.make_env(psi,shiftedpsi)		
			RE=REnv[n]=enviro.make_env(psi,shiftedpsi)		
			LE.copy_LE_from(cleanEnv)
			RE.copy_RE_from(cleanEnv)
			#tensor LE with bras
			#tensor LE with X
			#tensor RE with ket
			#tensor RE with X
			
			corr[n,l]=enviro.contract(LE,RE,bonds=np.arange(psiL),verbose=0)
		
		return corr
		
	def SMA(self, k, S, qy, mu, a):
		"""	GMP Single mode approximation for the roton mode in layer (mu, a) assuming rotational invariance.
			
			Currently only implemented for single LL.
			
			rho(q) is projected density operator
			
			S(q) = < rho(-q) rho(q) >
			
			Ansatz: |k> = rho(k) | psi > / sqrt{ S(q) }
			
			180-degree reflection invariance gives dispersion as double commutator:
			
			E(k) - E0 = < [rho(-k), [H, rho(k)] ] > / 2 S(k)
			
			For H = int dq^2 / (2 pi)^2  V(q) rho(-q) rho(q),
			
			Using [rho_q, rho_k] = 2 i sin(q x k) rho_{q + k} * F(q) F(k) / F(q + k),
			
			fk = |F(k)|^2
			E(k) - E0 	= fk int dq^2 / (2 pi)^2 S(q) ( 1 - cos(k x q) ) [ V(q) - f_{q-k} / fq V(q - k)] / 2 S(k)
						= fk int dq^2 / (2 pi)^2 V(q) ( 1 - cos(k x q) ) [ S(q) - fq / f_{q+k} S(q + k)] / 2 S(k)


			Strategy: evaluate 2nd form in polar coordinate for q to ensure V(q) singularity well behaved.
		"""
		raise NotImplemented

	def LL_mixing(self, psi, omega, Ncut):
		"""
			Calculates energy shift for perturbative LL mixing. Assumes a single species in a single LL.
			
			
			dE = \sum_{n!=0}    <Psi| V | n > < n| V | Psi> / (E_n - E_Psi)
			
			Keeps energy denominators <= Ncut*omega
			
			
		"""
		if self.Nlayer > 1:
			raise NotImplemented

		LL = self.bands.iteritems()[0]
		
		Lmin = np.max([0, LL-Ncut])
		Lmax = LL + Ncut
		
		dE = 0
		model_par = deepcopy(self.model_par)

		for a in range(Lmin, Lmax):
			for b in range(a, Lmax):
				
				dN = np.abs(a - LL) + np.abs(b - LL)
				
				if dN > Ncut:
					continue
	
				
				if a < b and b < LL:
					#Double scattering into distinct LL
					model_par['layers'] = [ (0, LL), (0, a), (0, b)]
					filter = np.zeros((3, 3, 3, 3))
					filter[0, 1, 0, 2] = True
					filter[0, 2, 0, 1] = True
					proj = [None, 1, 1]
				elif a < LL and LL < b:
					#Double scattering into distinct LL
					model_par['layers'] = [ (0, LL), (0, a), (0, b)]
					filter = np.zeros((3, 3, 3, 3))
					filter[0, 1, 2, 0] = True
					filter[0, 0, 2, 1] = True
					filter[2, 1, 0, 0] = True
					filter[2, 0, 0, 1] = True
					proj = [None, 1, 0]
				elif  LL < a  and a < b :
					#Double scattering into distinct LL
					model_par['layers'] = [ (0, LL), (0, a), (0, b)]
					filter = np.zeros((3, 3, 3, 3))
					filter[1, 0, 2, 0] = True
					filter[2, 0, 1, 0] = True
					proj = [None, 0, 0]
				elif a==b and b < LL:
					#Single scattering into a < LL
					model_par['layers'] = [ (0, LL), (0, a)]
					filter[0, 1, 0, 1] = True
					proj = [None, 1]
				elif a == b  and b == LL:
					pass
				elif a==b and  LL < b :
					#Single scattering into a < LL
					model_par['layers'] = [ (0, LL), (0, a)]
					filter[1, 0, 1, 0] = True
					proj = [None, 0]
				elif a == LL and LL < b:
					#Single scattering into b > LL
					model_par['layers'] = [ (0, LL), (0, b)]
					filter = np.zeros((2, 2, 2, 2))
					filter[0, 0, 1, 0] = True
					filter[1, 0, 0, 0] = True
					proj = [None, 0]
				elif b == LL and a < LL:
					model_par['layers'] = [ (0, LL), (0, a)]
					filter = np.zeros((2, 2, 2, 2))
					filter[0, 0, 0, 1] = True
					filter[0, 1, 0, 0] = True
					proj = [None, 1]


				VV = calc_vev(psi, V)
				dE -= VV / (omega*dN)

		return dE

	def Vq0_energy(self, psi=None, verbose=0):
	#TODO, check why passing in psi doesn't work.
		"""	Gives the energy per unit FLUX from V(q=0) component.	"""
		if psi is None:
			exp_n = np.mean(self.root_config.reshape((-1,self.Nlayer)), axis = 0)
			densities = {}
			for s,orb in self.species.items():
				densities[s] = np.sum([ exp_n[i] for i in orb ])
		else:
			densities = self.species_densities(psi, verbose=verbose)
	
		E0 = 0.
		for s1,s2 in self.Vq0:
			E0 += densities[s1] * densities[s2] * self.Vq0[(s1, s2)]
		return E0

	
	def species_charge(self):
		"""	Compute the charges of each species.

			Return a tuple of three objects:
				list of species,
				dictionary maping species to indices (in the list above),
				and 2-array shape (# species, # charges)
			"""
		Sp_list = []
		Sp_dict = {}
		Q_Sp = np.zeros((len(self.species), self.num_cons_Q), dtype=int)
		for spi,sp in enumerate(self.species.iteritems()):
			Sp_list.append(sp[0])
			Sp_dict[sp[0]] = spi
			Q = self.Qmat[:, np.array(sp[1])]		# columns are the charges
			if Q.shape[1] > 1 and np.any(Q[:, 1:] - Q[:, :-1] > 0): raise RuntimeError, str(Q)
			Q_Sp[spi] = Q[:, 0]
		return Sp_list, Sp_dict, Q_Sp


	def momentum_polarization(self, psi, verbose=0):
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		OrbSpin = np.zeros(self.Nlayer, dtype=float)
		for Sp,band_index in self.species.iteritems():
			for i,ly in enumerate(band_index):
				OrbSpin[ly] = self.bands[Sp].orbital_spin(i)
		q = self.root_config.size		# This should be the qDenom
		if self.p_type == 'F':
			return self.wf_mompol(psi, bonds=None, Lx=self.Lx, OrbSpin=OrbSpin, PhiX=0.5, verbose=verbose)
		else:
			print 'momentum_polarization:  This code probably doesn\'t work with p_type != F.  But I\'m going to try anyway.'
			return self.wf_mompol(psi, bonds=None, Lx=self.Lx, OrbSpin=OrbSpin, PhiX=0, verbose=verbose)


	@classmethod		
	def wf_mompol(cls, psi, bonds=None, Lx=None, PhiX=0.5, OrbSpin=0.5, qDenom=None, verbose=0):
		"""	Return the real-space momentum polarization.
					psi:  wavefunction
					bond:  which bonds to compute mompol for, by default it'll compute bonds -1 to L-1 (the boundary of the unit cell are computed twice)
					Lx:  circumference.  (Needed to correctly get the shift)
					PhiX:  flux in the cylinder.  Related to the spatial position to take the real-space cut.  (0 cuts through middle of site, 0.5 cuts through middle of bond.)
					qDenom:  normalization for the K quantum number.
			"""
		if verbose >= 1: print 'Momentum Polarization:'
		spec = psi.entanglement_spectrum()
		L = psi.L
		Nlayer = psi.translate_Q1_data['keywords']['Nlayer']
		if L % Nlayer: raise "AHHHHHHHHHHHHHHHH"
		M = L // Nlayer		# number of orbitals/flux in psi
		Kvec = psi.translate_Q1_data['keywords']['Kvec'] 
		try:
			iter(OrbSpin)
		except TypeError:
			OrbSpin = [OrbSpin] * Nlayer
		if L % len(OrbSpin): raise ValueError
		num_q = psi.num_q 		#num_cons_Q = num_q - 1
		if num_q < 1: raise ValueError

		Kvec_L = Kvec * (L // Nlayer)
		Qp_flat = np.array([ B.q_ind[0][:2, 2:] for B in psi.B ])		# first two q-indices corresponds to [empty, occupied].
		site_Q = (Qp_flat[:, 1, :] - Qp_flat[:, 0, :])		# the charge/particle on each site.  array shaped (L, num_q)

		if qDenom is None:
			from tools.math import gcd_array
			qDenom = gcd_array(site_Q)		# make a guess
		if bonds is None:
			site_K = site_Q[:, 0]
			bonds = np.arange(-1, L, Nlayer)
			return_as_scalar = False
		else:
			try:
				iter(bonds)
				return_as_scalar = False
			except TypeError:
				bonds = [bonds]
				return_as_scalar = True
			for b in bonds:
				if (b+1) % Nlayer: raise ValueError, (b, Nlayer)		# make sures it's on orbital boundary

		spec_weight = [ np.array([ (list(Q) + [np.sum(np.exp(-2*EV))]) for Q,EV in sp ]) for sp in spec ]
		total_weight = np.array([ np.sum(sw[:, -1]) for sw in spec_weight ])
		if np.any(np.abs(total_weight - 1) > 1e-13): raise ValueError
		bond_avgQ = [ np.sum(sw[:, :-1].T * sw[:, -1], axis=1) for sw in spec_weight ]
		bond_avgQ = np.array([bond_avgQ[-1]] + bond_avgQ)		# 2-array (L+1, num_q), from bonds -1 to L-1
		bond_avgQ[0, 0] -= np.dot(bond_avgQ[0, 1:], Kvec_L)		# manually translate_Q
		site_avgQ = bond_avgQ[1:, :] - bond_avgQ[:-1, :] - Qp_flat[:, 0, :] + np.array([ B.charge for B in psi.B ])
		site_avgN = np.sum(site_avgQ * site_Q, axis=1) / np.sum(site_Q * site_Q, axis=1)		# shaped (L,)
		unit_site_avgN = np.mean(site_avgN.reshape((M, Nlayer)), axis=0)		# shaped (Nlayer,)
		background_K = unit_site_avgN * (PhiX**2 - PhiX + 1./6) / 2.		# mompol from average (static) charge
		if Lx is not None:
			for i in range(Nlayer):
				background_K[i] -= unit_site_avgN[i] * (Lx/np.pi)**2 / 8. * OrbSpin[i % Nlayer]
		if verbose >= 1: print '\t<n> =', site_avgN, unit_site_avgN
		
		MomPol = []
		for b in bonds:
			KCw = spec_weight[b%L][np.argmax(spec_weight[b%L][:, -1])]		# find charge with largest weight
			K = KCw[0] + (b // L) * np.dot(Kvec_L, KCw[1:-1]) + PhiX * np.dot(Kvec, KCw[1:-1])
			avgQ = bond_avgQ[(b+1)%L].copy()
			avgQ[0] += (b // L) * np.dot(Kvec_L, avgQ[1:]) + PhiX * np.dot(Kvec, avgQ[1:])
			cdw_K = 0.
			##	The <K> from non-uniform charge orbital component is given by s (s - M) / 2 (M-1).
			##	The (M-1) is because we assumed that <<C>> = average charge in unit cell = 0.
			##	I.e., there is a background charge everywhere and a delta function spike at site s.
			if M > 1:
				for s in range(b+1, L+b+1):
					sloc = (s - b - 1) // Nlayer + PhiX
					#print s, site_avgN[s % L] - unit_site_avgN[s % Nlayer], sloc, sloc*(sloc-M)/2./M
					cdw_K += (site_avgN[s % L] - unit_site_avgN[s % Nlayer]) * (sloc * (sloc-M) / 2. / (M-1))
			
			if verbose >= 1: print '\tbond', b, avgQ, K, KCw, (K-avgQ[0])/qDenom, (K-avgQ[0])/qDenom + np.sum(background_K), ', site loc', (b + 1) // Nlayer + PhiX, cdw_K
			MomPol.append( (K-avgQ[0])/qDenom + np.sum(background_K) + cdw_K )
		
		if return_as_scalar: 
			return MomPol[0]
		else:
			return np.array(MomPol)


	def charge_weights(self, psi, combine_K=True, weight_func='prob', output_form='dict', verbose=0):
		""" Return the charge distribution of the entanglement cuts.
			The elements of the (output) list corresponds to bond -1, Nlayer-1, 2*Nlayer-1, ..., psi.L-1.  Note the first/last element are the same.
			
			Depending on output_form, the list will consist of dictionaries or more lists.
			Elements of the list corresponds to bonds (starting with -1) that crosses an orbital.
			If output_form == 'dict', then
				each dictionary (corrpesonding to a bond) is of the form charge:weight
			If output_form == 'list', then
				each list (corrpesonding to a bond) is of the form [ q_1, ..., q_n, weight ]
			The weight_func tells the function what function to apply to the entnanglement spectrum.
			"""
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		if self.cons_K > 0 and combine_K:
			Qi = 1
		else:
			Qi = 0
		return self.wf_charge_weights(psi, Nlayer=self.Nlayer, Qi=Qi, weight_func=weight_func, output_form=output_form, verbose=verbose)


	@classmethod
	def wf_charge_weights(cls, psi, Nlayer=None, Qi=0, weight_func='prob', output_form='dict', verbose=0):
		""" Return the charge distribution of the entanglement cuts.  See charge_weights() for documentation.
			
			Nlayer is the number of layers.  Only bonds at X * Nlayer - 1 for integer X will be considered.
			Qi is how many conserved charges to ignore.
			"""
		if Nlayer is None:
			try:
				Nlayer = psi.translate_Q1_data['keywords']['Nlayer']
			except:
				raise ValueError, "We don't know what Nlayer is."
		spectrum = psi.entanglement_spectrum()
		bonds = np.arange(-1, psi.L, Nlayer)		# -1 is included first (and last)
		if isinstance(weight_func, str):
			weight_func = {
						'prob': lambda x: np.sum(np.exp(-2*x)),
						'min': lambda x: np.min(x),
					}[weight_func] 	# if weight_func is a string, convert it to a function
		
		Qweights_list = []
		for b in bonds:
			spec = spectrum[b]
			Qspec = {}
			for s in spec:
				Q = s[0][Qi:]
				Qspec.setdefault(Q, []).append(s[1])
			
			if output_form == 'dict':
				Qweights = {}
				for Q,ws in Qspec.iteritems():
					Qweights[Q] = weight_func(np.hstack(ws))
			elif output_form == 'list':
				Qweights = []
				for Q in sorted(Qspec.keys()):
					ws = Qspec[Q]
					Qweights.append( list(Q) + [weight_func(np.hstack(ws))] )
			Qweights_list.append(Qweights)
		
		return Qweights_list

	def charge_polarization(self,psi):
		""" returns the charge polarization on every bond on the chain. 
			in the form of a [num_charges,num_bonds] 2-array. Only does physical charges, not k's"""

		Nlayer=self.Nlayer

		P = psi.probability_per_charge(bonds=np.arange(-1, psi.L-1, Nlayer)) #get the (unnormalized) probabilities

		#the charges in the bonds are are multiplied by some integer charge quanta, we need to divide this out
		#we can find this quantity by subtracting two different charges on a bond, but sometimes there is only one charge of a given type on a bond
		#so we check all bonds in order to find the right amount of charge
		#at small chi there is still occasionally the case that for all bonds, all entries on that bond are the same. This will lead to a charge polarization of NaN
		#I've only seen this happen for bond dimension 10
		charges=np.zeros(Nlayer)
		for p in P:
			diffs=p[self.cons_K:-1,1:]-p[self.cons_K:-1,:-1]	
			for i in range(Nlayer): 
				try:
					n=min(abs(x) for x in diffs[i,:] if abs(x)>0)
					if(n>charges[i]): charges[i]=n
				except ValueError:
					continue

		print "charges",charges
		out=np.empty((len(P),Nlayer))
		for j,p in enumerate(P):			
			#need to subtract off the piece with the highest weight
			k=np.argmax(p[-1,:])
			print p[self.cons_K:-1,k,np.newaxis]
			p[self.cons_K:-1,:]=p[self.cons_K:-1,:]-np.multiply(p[self.cons_K:-1,k,np.newaxis],np.ones([Nlayer,p.shape[1]]))

			#now take the inner product which gives the charge polarization
			out[j,:]=[np.inner(p[self.cons_K+i,:],p[-1,:])/charges[i] for i in range(Nlayer)]

		
		return out


	################################################################################
	################################################################################
	##	Now code for displaying shit
	@classmethod		
	def plot_spectrum(cls, psi, Nsector, bond=-1, catch_error_type=Exception, save_fig=None, subplot=None, limits=(None,None), print_spectrum=False, print_spectrum_cutoff=None, plot_kwargs={}, label=True, title=True, plot_pts_filter=None):
		"""Plot the entanglement spectrum (vs. momentum) for one particular charge sector.

	Parameters:
		psi:  mps wavefunction (should have at least two conserved charges, momentum being the first one).
		Nsector:  The charge sector.  It may be a tuple of numbers representation a charge sector, an option: 'absmin' or 'maxweight'.
		bond:  The bond to display the spectrum.
		
		subplot:
			An optional tuple (nr, nc, j) passed to plt.subplot(), or a plot module object.
			In either case, this function delays calling plt.show().
						
		Style options:
			plot_marker:  No longer an option!!!
			
			plot_kwargs : additional kwargs to plt.plot() 
			
			label: 
					False or None: do not label axes
					True: default labels
			title:
					False or None: no title
					True: default title
					<type str>: put the string as title
		
		print_spectrum:  Also print the entanglement spectrum
		print_spectrum_cutoff:  If printing the spectrum, only print energies < cutoff.
		
		plot_pts_filter: A function that processes the data points.  It should take in a (#,2)-array, and return a (#',2)-array.
			- It may be used to offset data (e.g. zero the minima).
			- It may be used to eliminate data points outside of plot range (to reduce file size).
	
	Returns a dictionary with random tidbits of information."""
		info_dict = {}		# accumulates information as the function progresses
		try:
			import matplotlib
			spec = psi.entanglement_spectrum(bond)[0]
			if len(spec) == 0:
				print "plot_spectrum: No spectrum to plot!"
				return info_dict

			wh, wv = 1, 0.1
			plt_kwargs = {'markersize':8, 'color':'red', 'marker':[(-wh, -wv), (wh, -wv), (wh, wv), (-wh, wv)], 'linewidth':0.}#navy
			plt_kwargs.update(plot_kwargs)
			
		##	Figure out the charge sector
			if Nsector == 'absmin':
				minEE = np.array([ list(sec[0]) + [ np.min(sec[1]) ] for sec in spec ])
				N = minEE[np.argmin(minEE[:, -1]), 1:-1]		# pick only the number sectors
			elif Nsector == 'maxweight':
				minEE = np.array([ list(sec[0]) + [ np.sum(np.exp(-2*sec[1])) ] for sec in spec ])
				N = minEE[np.argmax(minEE[:, -1]), 1:-1]		# pick only the number sectors
			else:
				N = np.array(Nsector, dtype=int)
			info_dict['N sector'] = N
			print('The CHARGE SECTOR IS',N,'========================================')
			pts = []
			for sec in spec:
				if np.all(sec[0][1:] == N):
					pts = pts + [ [sec[0][0], p] for p in sec[1] ]
			pts = np.array(pts)
			if len(pts) == 0: pts = np.zeros((0, 2), dtype=int)
			pts[:, 0] /= float(psi.L)
			if plot_pts_filter is not None:
				pts = plot_pts_filter(pts)
			if len(pts) == 0: print "plot_spectrum: No points to plot!"
			if print_spectrum:
				cls.print_spectrum(psi, bond=bond, charge_filter=tuple(N.tolist()), spec_cutoff=print_spectrum_cutoff)
			
			if subplot is None:
				import matplotlib.pyplot as plt
			elif type(subplot) == list or type(subplot) == tuple:
				plt = matplotlib.pyplot
				plt.subplot(*subplot)
			else:
				plt = subplot
			plt.plot(pts[:, 0], pts[:, 1], **plt_kwargs)
			
			if label == True:
				if hasattr(plt, 'set_xlabel'):plt.set_xlabel(r'$K$', fontsize=18)
				if hasattr(plt, 'xlabel'): plt.xlabel(r'$K$', fontsize=18)
				if hasattr(plt, 'set_ylabel'): plt.set_ylabel(r'$E_i$', fontsize=18, rotation='horizontal')
				if hasattr(plt, 'ylabel'): plt.ylabel(r'$E_i$', fontsize=18, rotation='horizontal')
			
			if title == True or isinstance(title, str):
				plot_title = title
				info_dict['title'] = plot_title
				if title == True: plot_title = "ES in N = {} sector".format(N)
				if hasattr(plt, 'title') and (not isinstance(plt.title, matplotlib.text.Text)): plt.title(plot_title)
				if hasattr(plt, 'set_title'): plt.set_title(plot_title)
			
			if limits[0] is not None:
				if hasattr(plt, 'set_xlim'): plt.set_xlim(limits[0])
				if hasattr(plt, 'xlim'): plt.xlim(limits[0])
			if limits[1] is not None:
				if hasattr(plt, 'set_ylim'):plt.set_ylim(limits[1])
				if hasattr(plt, 'ylim'):plt.ylim(limits[1])
			
			if save_fig is not None:
				plt.savefig(save_fig)
				plt.clf()
			elif subplot is None:
				plt.show()
		
		except catch_error_type as XXX:		# shit happens
			print "Plot spectrum failed!"
			print '\t', type(XXX)
			print '\t', XXX.args
			info_dict['Error'] = XXX

		return info_dict
		

	@classmethod
	def print_spectrum(cls, psi, bond = -1, charge_filter = None, print_spec = True, print_counting = False, spec_cutoff = None):
		"""	Print the entanglement spectrum.
				
			Parameters:
				psi: the wavefunction
				bond (default -1): bond
				charge_filter (default None): None, or a tuple of numbers.  Only charges whoses last index match the filter will be shown.
				print_spec (default True): print the entanglement spectrum
				print_counting (default False): print the number of states in the spectrum.
			"""
		if bond == 'all':
			bond = np.arange(-1, psi.L - 1)
		try:
			iter(bond)
		except TypeError:
			bond = [bond]
		specs = psi.entanglement_spectrum(bond)
		if charge_filter is not None:
			if not isinstance(charge_filter, tuple): raise ValueError
			if len(charge_filter) > psi.num_q: raise ValueError
			Qi = psi.num_q - len(charge_filter)
		
		s = ''
		for b,spec in itertools.izip(bond,specs):
			s += "Entanglement spectrum at bond {}:\n".format(b)
			if charge_filter is not None:
				s += "Charge filter {}\n".format(charge_filter)
			entspec = spec
			Q = np.array([ e[0] for e in entspec ])
			perm = np.lexsort(Q.transpose())
			entspec = [ entspec[i] for i in perm ]
			for e in entspec:
				if charge_filter is not None:
					if e[0][Qi:] != charge_filter: continue
				if spec_cutoff is not None:
					e1 = e[1][np.nonzero(e[1] < spec_cutoff)]
				else:
					e1 = e[1]
				if len(e1) == 0: continue
				slist = ['\t', e[0], ": "]
				if print_counting: slist.append( "(%i) " % len(e1) )
				if print_spec: slist.append(e1),
				s += joinstr(slist) + '\n'
			s += '\n'
		print s

	##real space decomposition for use when calculating entanglement
	## SG copied this from quantum_hall.py and modified it for multiple layers and bosons
	def realspace_decomposition(self, psi, truncation_par=None, start_site=0, end_site=-1, verbose=0):
		"""	Take psi, split every site in to two, move them all to left / right.
			Returns a copy.
			
			The sites start_site to end_site (inclusive) are decomposed,
				anything to the left of start_site or right of end_site will remain as is.
			"""
		if psi.bc != 'segment' and psi.bc != 'finite': raise NotImplementedError
		
		L = psi.L
		a = start_site % L	# first site to split
		b = end_site % L + 1		# past the last site to split
		if b < a: raise ValueError
		if verbose > 2: psi.norm_test2(verbose = 1)
 		circum = self.Lx
		if truncation_par is None:
			trunc_par = {}
		else:
			trunc_par = truncation_par.copy()	# don't mess with the original
		if 'tol' not in trunc_par: trunc_par['tol'] = 1e-7
		if verbose > 0:
			print "quantum_hall realspace_decomposition: %i sites with L = %s" % (L, circum)
			print "\tchi (before) = " + str([psi.chi[-1]] + psi.chi[:-1].tolist())
		
	##	Split sites
		psi2 = self.split_sites(psi, trunc_par, start_site=start_site, end_site=end_site, verbose=verbose)
		if verbose > 2: psi2.norm_test2(tol = np.inf, verbose = 1)
		if verbose > 0:
			print "\tchi (split) = " + str([psi2.chi[-1]] + psi2.chi[:-1].tolist())
		
	##	Swap the sites
	##		There's (b - a) pairs of unit cells
	##		sites (a) to (2b-a-1) have been split
		swapOp = 'f' if self.p_type == 'F' else None
		if verbose == 1: print "\tswap",
		for start_i in range(a + 1, b):	# (b-1,b) will be the last pair to be swapped
			for i in range(start_i, 2 * b - start_i, 2):
				oldchi = psi2.chi[i]
				s_trunc = psi2.swap_sites(i, truncation_par = trunc_par, swapOp = swapOp)
				if verbose == 1: print "(%i-%i, %i->%i)" % (i, i+1, oldchi, psi2.chi[i]),
				if verbose > 1: print "\tswap (%i, %i), chi %i -> %i, s_trunc = %s (%s mb)" % (i, i+1, oldchi, psi2.chi[i], s_trunc, round(omp.memory_usage()))
			#psi2.norm_test2()
		if verbose > 0: print
		
		#print joinstr(["\tpsi2.B = "] + psi2.B, delim = ' | ')
		if verbose > 1: print joinstr(["\tphysical charges: "] + [Bs.q_ind[0] for Bs in psi2.B])
		#print joinstr(["\t"] + psi2.s), "\n"
		
		if verbose > 0:
			print "\tchi (after) = " + str([psi2.chi[-1]] + psi2.chi[:-1].tolist())
		
		psi2.bond_cut = b - 1	# the bond to measure things from
		psi2.check_sanity()
		if verbose > 2: psi2.norm_test2(verbose = 1)
		return psi2

	def split_sites(self, psi, truncation_par = None, start_site=0, end_site=-1, verbose = 0):
		"""	Take psi, split every site in to two.
			Returns a copy. """
		L = psi.L
		d = psi.d
		circum = self.Lx
		if not np.array_equiv(psi.form, np.array([0., 1.])): raise NotImplementedError
		if psi.bc != 'segment' and psi.bc != 'finite': raise NotImplementedError
		if truncation_par is None:
			trunc_par = {'tol': 1e-7}
		else:
			trunc_par = truncation_par.copy()	# don't mess with the original
		a = start_site % L	# first site to split
		b = end_site % L + 1		# past the last site to split
		if b < a: raise ValueError
		try:
			origin = psi.realspace_origin/(1.*self.Nlayer)
		except:
			origin= (b+a)/(2.*self.Nlayer)

	##	1) Set up the new mps
		splitpsi = iMPS(L + b - a, form = 'B', bc = 'segment')
		splitpsi.d = d
		splitpsi.B = [None] * splitpsi.L
		splitpsi.s = [None] * splitpsi.L + [psi.s[-1]]
		splitpsi.chi = np.zeros(splitpsi.L + 1, int)
		splitpsi.chi[-1] = psi.chi[-1]		# left most bond
		if verbose > 0: print "\tquantum_hall split_sites: sites [%i,%i) with circumference = %s" % (a, b, circum)
		
	##	2) Copy the mps for sites that aren't split
	##		For each site, copy chi and s of the right bond
		for i in range(a):
			splitpsi.B[i] = psi.B[i].copy()
			splitpsi.s[i] = psi.s[i].copy()
			splitpsi.chi[i] = psi.chi[i]
			if verbose > 1: print "\t  copy  %2s -> %2s" % (str(i), str(i))

	##	3) Fill in the new mps, looping through sites
	##		erf(z) is defined as 2/sqrt(pi)*integral(exp(-t**2), t=0..z), with limits -1 and 1 as z -> -inf and inf.
	##		erfc(z) = 1 - erf(z).  (limits 2 and 0 respectively)
		for i in range(a, b):

			x = (i//self.Nlayer - origin) * 2. * np.pi / circum
			if verbose > 1:
				probL = scipy.special.erfc(x) / 2
				probR = scipy.special.erfc(-x) / 2
				print "\t  split %2s -> [%2s,%2s], x = % 2.3f, amp = (%.9f, %.9f)," % (str(i), str(2*i-a), str(2*i-a+1), x, np.sqrt(probL), np.sqrt(probR)),
			charge_dist = 'L' if i//self.Nlayer < origin else 'R'
			Bleft, Bright, Y, s_trunc = self.split_site(psi, i, x, trunc_par, charge_dist = charge_dist, verbose = 0)
			splitpsi.B[2 * i - a] = Bleft
			splitpsi.B[2 * i - a + 1] = Bright
			splitpsi.s[2 * i - a] = Y
			splitpsi.chi[2 * i - a] = len(Y)
			splitpsi.s[2 * i - a + 1] = psi.s[i]
			splitpsi.chi[2 * i - a + 1] = psi.chi[i]
			if verbose > 1: print "chi = %s, s_trunc = %s (%s mb)" % (len(Y), s_trunc, round(omp.memory_usage()))

	##	4) Finish copying the remaining sites
		for i in range(b, L):
			splitpsi.B[i + b - a] = psi.B[i].copy()
			splitpsi.s[i + b - a] = psi.s[i].copy()
			splitpsi.chi[i + b - a] = psi.chi[i]
			if verbose > 1: print "\t  copy  %2s -> %2s" % (str(i), str(i+b-a))

		splitpsi.check_sanity()
		return splitpsi


	def split_site(self, psi, site, x, trunc_par, charge_dist = 'L', verbose = 0):
		"""	Split one site in an MPS.
			
			Parameters:
				psi: the MPS
				site: the site to split
				x: realspace position of the site
				trunc_par: truncation parameters to be passed to svd_theta
				charge_dist: 'L' or 'R' based on which side gets all the charge
			
			Returns:
				tuple (Bleft, Bright, s (in between), s_trunc)
			
			Note: this function does not alter self or psi.
			"""
		L = psi.L
		sLen = len(psi.s)
		d = self.d
		try:
			iter(d)
			d=self.d[0]
			if(np.any(self.d != d)):
				raise NotImplementedError, "ds must be the same for all layers"
		except:
			pass
				
		if site >= L or site < 0: raise ValueError
		if not np.array_equiv(psi.form, np.array([0., 1.])): raise NotImplementedError

	##	The probabilities on each side is given by the error function.
	##		erf(z) is defined as 2/sqrt(pi)*integral(exp(-t**2), t=0..z), with limits -1 and 1 as z -> -inf and inf.
	##		erfc(z) = 1 - erf(z).  (limits 2 and 0 respectively)
		probL = scipy.special.erfc(x) / 2
		probR = scipy.special.erfc(-x) / 2
		if verbose > 0: print "\tsplit %2s, x = % 2.3f, amp = (%.9f, %.9f)," % (str(site), x, np.sqrt(probL), np.sqrt(probR)),
		sLeft = psi.s[(site - 1) % sLen]		# s on the left
		sB = psi.B[site].scale_axis(sLeft, axis = 1)		# TODO assumes B form
	
	##	Form the 3-tensor that converts two-particle filling to single particle filling
		partition = np.zeros((d,d,d), float)	# legs L, R, together
		for fL in range(d):
			for fR in range(d):
				if fL + fR < d:
					partition[fL, fR, fL+fR] = sp.misc.comb(fL+fR,fR)*probL**(fL/2.) * probR**(fR/2.)	# gives the amplitude

	##	Decide how to distribute the charge left and right
		qind0 = sB.q_ind[0]
		if charge_dist == 'L':
			qindL = qind0
			qindR = qind0.copy()
			qindR[:, 2:] -= qind0[0, 2:]
		elif charge_dist == 'R':
			qindR = qind0
			qindL = qind0.copy()
			qindL[:, 2:] -= qind0[0, 2:]
		else:
			raise ValueError
		q_conj = sB.q_conj[0]
		p = npc.array.from_ndarray( partition, [qindL, qindR, qind0], q_conj = [q_conj, q_conj, -q_conj], mod_q = sB.mod_q )

	##	Split the site into two
		theta = npc.tensordot(p, sB, axes=([2],[0])).transpose((0, 2, 1, 3))
		pipeL = theta.make_pipe([0, 1], qt_conj = 1)
		pipeR = theta.make_pipe([2, 3], qt_conj = -1)
		XYZ = theta.combine_legs([[0, 1], [2, 3]], pipes = [pipeL, pipeR])
		X, Y, Z, s_trunc = LA_tools.svd_theta(XYZ, trunc_par)
		Bright = Z.split_legs([1], [pipeR]).transpose((1, 0, 2))
		X.iscale_axis(Y, axis = 1)
		Bleft = X.split_legs([0], [pipeL]).iscale_axis(sLeft**(-1), axis = 1)
		
		if verbose > 0: print "chi = %s, s_trunc = %s (%s mb)" % (len(Y), s_trunc, round(omp.memory_usage()))
		return Bleft, Bright, Y, s_trunc
	

	###############################################################
	############################### Zaletel's Grotto ##############
	###############################################################			
	########################### Enter at your own risk! ###########
	###############################################################			
	def make_Trm(self, Ts):
		""" Generates single particle `hopping' elements. 
			List of lists for tunneling from species mu->nu
			T[mu][nu][a, b, r, m]
		
			H = sum_{r, m} T[mu][nu]_{a, b, r, m} psi^D_{mu a; r} psi_{nu b; r - m}
			
		HOWEVER, Trm only stores values for m>=0. The remainder are infered by conjugation.
		
		(r, m) part has shape self.L, maxM + 1
		
		If T.shape = [L1, L2], then T_{r + L1, c} = T_{r, c} - this is just a periodicity of 'L1' along the chain. We call L1 the 'unit cell'
		
		T is generated from higher level description 'Ts'. Ts is a dictionary which can have the following keys:
		
		'pins'. List of pinning potentials: see t_pins
		
		'tunneling'. Inter-layer tunneling
		
		'Tq'. Periodic potential
		
		'background_x': background potential rho(x), interacting via Vq[mu][mu]
		
		'Tx'/ 'Ty' real space potentials in x or y
		
		"""
		species = self.species
		bands = self.bands
		maxM = np.ceil( 10./self.kappa)
		############### 23 Apr 2024
        # maxM should be an integer, not float!!!
		maxM = int(maxM)
		verbose = self.verbose
		if verbose >= 2: print '\tmake_Trm:'
		Trm = self.Trm = {}
		for mu in species.keys():
			Trm.setdefault(mu, {})
			for nu in species.keys():
				Trm[mu].setdefault(nu, None)
		
		def get_trm(mu, nu):
			n_mu = bands[mu].n
			n_nu = bands[nu].n
			if Trm[mu][nu] is None:
				t = np.zeros((n_mu, n_mu, self.Nflux, maxM+1), self.dtype)
				Trm[mu][nu] = t
				return t
			else:
				return Trm[mu][nu]
		
		if 'pins' in Ts:
			for mu, t in Ts['pins'].iteritems():
				if verbose >= 2: print "\t\tMaking pins: mu = %s, t = %s" % (mu, t)
				trm = get_trm(mu, mu)
				trm += self.t_pins(t, maxM, bands[mu])
		
		if 'Tq' in Ts:
			for mu, t in Ts['Tq'].iteritems():
				if verbose >= 2: print "\t\tMaking egg carton: mu = %s, t = %s" % (mu, t)
				trm = get_trm(mu, mu)
				self.t_q(t, trm, bands[mu])	
		if 'tunneling' in Ts:
			"""For each (mu, nu), gives
			
				[  t_{mu nu} \int d^2r  psi^D_mu(r) \psi_nu(r)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
			"""
			for (mu, nu), t in Ts['tunneling'].iteritems():
				if mu==nu:
					print "Warning: mu = nu tunneling"
				if verbose >= 2: print "\t\tMaking tunneling terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				trm = get_trm(mu, nu)				
				self.t_tunneling(t, trm, bands[mu], bands[nu])
				trm = get_trm(nu, mu)
				self.t_tunneling(t, trm, bands[nu], bands[mu])
		
		if 'background_x' in Ts:
			""" For each (mu, nu), gives
			
				[   \int d^2r  n_mu(x, y) V_{mu mu}(x, y) t(x)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for mu, t in Ts['background_x'].iteritems():
				if verbose >= 2: print "\t\tMaking background terms: (mu,) = (%s,), t = %s" % (mu, t)
				trm = get_trm(mu, mu)
				self.background_x(t, trm, mu)

				
		if 'Tx' in Ts:
			""" For each (mu, nu), gives
			
				[   \int d^2r  t_{mu nu}(y) psi^D_mu(x, y) \psi_nu(x, y)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for (mu, nu), t in Ts['Tx'].iteritems():
				if verbose >= 2: print "\t\tMaking Tx terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				
				trm = get_trm(mu, nu)
				self.t_x(t, trm, bands[mu], bands[nu])
				trm = get_trm(nu, mu)
				self.t_x(t, trm, bands[nu], bands[mu], conj=True)
				
		if 'Ty' in Ts:
			""" For each (mu, nu), gives
			
				[   \int d^2r  t_{mu nu}(y) psi^D_mu(x, y) \psi_nu(x, y)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for (mu, nu), t in Ts['Ty'].iteritems():
				if verbose >= 2: print "\t\tMaking tunneling terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				trm = get_trm(mu, nu)
				self.t_y(t, trm, bands[mu], bands[nu])
				trm = get_trm(nu, mu)
				self.t_y(t, trm, bands[nu], bands[mu], conj = True)
			
		self.Trm = Trm

	#(0,) : { 'tx':tx, 'Vs':{} }

	def t_x(self, t, trm, band1, band2, conj=False):
		"""
			t is a function, t(x), giving potential at 'x'.
			
		"""
		if not band1==band2:
			raise NotImpemented
		maxM = trm.shape[-1]-1
		 
		#Sample 4x as many points as maxM
		dx = self.Lx/4./maxM
		xs = np.arange(maxM*4)*dx
		tq = np.fft.rfft(t(xs))[:maxM+1]/(4.*maxM)

		if np.linalg.norm(np.real(tq) - tq) > 1e-10:
			print(np.linalg.norm(np.real(tq) - tq))
			print "Warning: only even part of t_x kept"
		tq = np.real(tq)
		for a in range(band1.n):
			for b in range(band1.n):
				for m in range(maxM+1):
					#print(tq[m]*band1.pre_F(a, b)*band1.F(a,b, -m*self.kappa, 0)*np.exp(-0.5*(m*self.kappa)**2))
					trm[a, b, :, m] += tq[m]*band1.pre_F(a, b)*band1.F(a,b, -m*self.kappa, 0)*np.exp(-0.5*(m*self.kappa)**2)###why not e^{-q^2/4} rather than e^{-q^2/2}??
				

	def t_y(self, t, trm, band1, band2, conj = False):
		"""
			t is a function, t(y), giving potential at 'y'.
			
		"""
		Ly = self.Nflux * self.kappa
		
		mul = int(np.ceil(self.kappa*16))
		dy = self.kappa/mul
		ym = 8.
		ys = np.arange(0, ym, dy)
		ym = ys[-1]
		
		y_V = np.arange( -ym, ym + Ly - self.kappa +  1e-6, dy)
		V = np.conj(t(y_V))

		for a in range(band1.n):
			phi_a = band1.phi(a, ys)
			for b in range(band2.n):
				phi_b = band2.phi(b, ys)
				
				f = phi_a*phi_b
				f = np.concatenate( [ band1.eta(a)*band2.eta(b)*f[-1:0:-1], f])
				
				s = sp.convolve(f, V, mode='valid')[::mul]
				trm[a, b, :, 0] += s*dy



	def t_q(self, t, trm, band):
		""" 
			t = [ (qx, qy, V[q]), . . . ]

				where
				
			H = -int d^2x V(x) rho(x)
			  = -\sum_i int d^2x V[q_i] e^{i q_i r) rho(r)
			  = -\sum_i V[q_i] rho(-q_i)
			  
			NOTE 1: only the hermitian component of H is kept;
			
				1/2 ( V[q_i] rho(-q_i) + V[-q_i] rho[ q_i ] )

				where V[-q] = conj( V[q] )
				
			NOTE 2: only the part symmetric under  x -> -x is kept, implying
			
					V[qx, qy] = V[-qx, qy]
					
			Together, we can safely add in the form
			
				[ t rho(qx, qy) + conj(t) rho(-qx, qy) + t rho(-qx, qy) + conj(t) rho(qx, -qy) ] /4  
					
		"""
			
		maxM = trm.shape[-1]-1
		for i in range(len(t)):
			qx = t[i][0]
			qy = t[i][1]
			tq = t[i][2]
			#Use symmetry to bring to quadrant qx, qy >=0
			if qy < 0:
				qx*=-1
				qy*=-1
				tq = np.conj(tq)
			if qx < 0:
				qx*=-1
				
			m = qx*self.Lx/2./np.pi
			if abs( m - round(m)) > 10**(-10) or round(m) > maxM:
				print "nx:", m
				raise ValueError
			m = int(round(m))
				
			n = qy*self.L/self.Lx
			if abs( n - round(n)) > 10**(-10):
				print "ny:", n, "L:", self.L,"qy:",qy 
				raise ValueError	
			n = int(round(n))
			
			q2 = qx*qx+qy*qy
			ks = np.arange(self.L)*self.kappa
			for a in range(band.n):
				for b in range(band.n):
					
					if m!=0: #In this case we are gauranteed conjugate entry will be added by make_Trm_list
						trm[a, b, :, m] += 0.5*np.real(tq*band.pre_F(a, b)*band.F(a,b, -qx, -qy)*np.exp(-0.5*q2)*np.exp(1j*qy*(ks - qx/2.)))
				
					if m==0: 
						trm[a, b, :, 0] += np.real(tq*band.pre_F(a, b)*band.F(a,b, 0, -qy)*np.exp(-0.5*q2)*np.exp(1j*qy*ks))

	def t_tunneling(self, t, trm, b1, b2):
		"""Delta function tunneling between two bands, 
			t int d^2x psi_mu^D(x) psi_nu(x)
		"""
		
		for a in range(b1.n):
			for b in range(b2.n):
				if b1[a]==b2[b]:
					trm[a, b, :, 0]+=t
				
	def t_pins(self, pins, maxM, bands):	
		"""	pins is list of Gaussian pinning potentials, each specified by
			pin = [x, y, w, h]
				= x-pos, y-pos, width of pin, strength of pin
			WARNING - assumes symmetry x -> -x, so pins must come in pairs!
			
		"""
		if bands.n > 1:
			raise NotImplementedError
		if  bands[0]!=0:
			print "WARNING: pins use LLL form factors"
		Tq = np.zeros((self.L // self.Nlayer, maxM+1), dtype = np.complex)
		
		M, R = np.meshgrid(np.arange(maxM+1,dtype = np.float), np.arange(self.L // self.Nlayer, dtype = np.float))
		R*=self.kappa
		M*=self.kappa
		
		for pin in pins:
			x,y,d,h = pin
			a = 1.+2.*d*d
			Tq += h*np.exp(-1j*M*x - 0.25*a*(M**2) - ((y - R + 0.5*M)**2.)/a)/self.Lx/np.sqrt(np.pi*a)
			
		Tq = Tq.reshape((1, 1,) + Tq.shape)
		
		return Tq.real


	def background_x(self, t, trm, mu):
		"""
			t = { 'tx': tx, 'Vs': {}}
			
			'tx' is a function, t(x), giving background charge at 'x'.
			Background interacts with electrons via interaction 'Vs'
			
		"""
		Vs = t['Vs']
		tx = t['tx']
		self.total_vq(Vs)
		band = self.bands[mu]
		maxM = trm.shape[-1]-1

		#Sample 20x as many points as maxM
		mul=20
		dx = (self.Lx/maxM)/mul
		xs = np.arange(maxM*mul)*dx
		tq = np.fft.rfft(tx(xs))[:maxM+1]/(mul*maxM)
		qs = np.arange(maxM+1)*self.kappa
		if np.linalg.norm(np.real(tq) - tq) > 1e-10:
			print "Warning: only even part of t_x kept"
		tq = np.real(tq)

		tq = 0.5*self.Vq[mu][mu](qs, 0.)*tq

		for a in range(band.n):
			for b in range(band.n):
				for m in range(maxM+1):
					trm[a, b, :, m] += tq[m]*band.pre_F(a, b)*band.F(a,b, -m*self.kappa, 0)*np.exp(-0.5*(m*self.kappa)**2)





	################################################################################
	################################################################################
	##	Code to make coefficients (Vmk) for 2-body potentials.

	def save_Vmk(self, filename, note=''):
		data = {'Lx':self.Lx, \
			'maxM':self.maxM, 'maxK':self.maxK, 'Vmk':self.Vmk, 'V_eps':self.V_eps, 'note':note}
		
		with open(filename, 'wb') as f:
			if self.verbose >= 1: print ("\t\tSaving Vmk file '%s'." % filename)
			cPickle.dump(data, f, protocol=-1)


	def load_Vmk(self, filename):
		with open(filename, 'rb') as f:
			if self.verbose >= 1: print ("\t\tLoading Vmk file '%s'." % filename)
			data = cPickle.load(f)
		if data['Lx'] != self.Lx: print ("Warning! File '%s' Lx = %s different from the current model Lx = %s." % (filename, data['Lx'], self.Lx))
		self.maxM = data['maxM']
		self.maxK = data['maxK']
		self.V_eps = data['V_eps']
		self.Vmk = data['Vmk']		# TODO, merge instead of overwrite
		#TODO: check if the keys in Vmk matches that of the current model.


	class Vmk_table(object):
		"""An object for automatically resizing and conveniently addressesing vmk"""
		#TODO, use this.
		
		def __init__(n_mu, n_nu):
			self.vmk = np.empty((n_mu, n_mu, n_nu, n_nu), dtype=np.object)
			for v in self.vmk.flat:
				v = {}
				
		def add(self, a, b, c, d, m, k, v):
			vm = self.vmk[a, b, c, d].set_default(m, {})
			vk = vm.set_default(k, 0.)
			vm[m][k] = vk + v
		def set(self, a, b, c, d, m, k):
			if m in self.vmk[a, b, c, d] and k in self.vmk[a, b, c, d][m]:
				return True
			else:
				return False
		def array(self):
			maxM = 0
			maxK = 0
			#Figure out largest dimension
			for v in self.vmk.flat:
				maxM = np.max( v.keys() + [maxM] )
				for k in d.iteritems():
					maxK = np.max( k.keys() + [maxK] )
			v = np.zeros(self.vmk.shape + (2*maxM+1, 2*maxK+1), dtype = np.complex)
			for a,b,c,d in itertools.product(range(n_mu),range(n_mu),range(n_nu),range(n_nu)):
				for m, vm in self.vmk[a, b, c, d].iteritems():
					for k, v in vm.iteritems():
						v[a, b, c, d, m+maxM, k+maxK] = v
					

	def make_Vmk(self, Vs, verbose=0):
		"""	Parses the Vs model parameters, and calls function to compute matrix elements Vmk. Writes to self.Vmk and self.V_mask
			
			Types of potentials:
				'coulomb': 'xi', 'v'   = v*e^{-r/xi}/r
				'coulomb_blgated': Coulomb with symmetricly placed metallic top & bottom gates
				'GaussianCoulomb':			'v', 'xi', ['order']
				'rV(r)':			'rV', ['tol', 'coarse', 'd']  (See vq_from_rV documentation for additional details.)
				'TK'   : [t0, t1, . . . ]
				'haldane' : [V0, V1, . . . ]
			Keys:
				'eps' : maximum acceptable error in approximating potential
				'xiK' : hint on real space extent of interations.
				'maxM' : over-ride defaults bounds on M
				'maxK' :						   on K
			Other:
				'loadVmk'	(probably doesn't work)
				'pickleVmk'		(still work, I think. RM)
		"""
		N = len(self.species)		
		magic_kappa_factor = 0.5 * (2*np.pi)**(-1.5) * self.kappa
		species = self.species
		bands = self.bands
		
		if verbose >= 2: print '\tmake_Vmk:'
	##	Load from file, or set defaults
		if 'loadVmk' in Vs:
			if isinstance(Vs['loadVmk'], list):
				[ self.load_Vmk(fn) for fn in Vs['loadVmk'] ]
			else:
				self.load_Vmk(Vs['loadVmk'])
		else:
			eps = self.V_eps = Vs.get('eps', 0.01)
			self.maxM = Vs.get('maxM', int(np.ceil(-np.log(0.1*eps)/self.kappa)))
			xiK = Vs.get('xiK', 1.)
			xiK = max([1., xiK])
			self.maxK = int(np.ceil(-np.log(0.1*eps)*xiK/self.kappa))
			self.Vq0 = { (k1,k2):0. for k1 in species for k2 in species }
			if not hasattr(self, 'Vmk'): self.Vmk = {}

		maxM = self.maxM
		maxK = self.maxK
		if maxM > maxK: raise ValueError, (maxK, maxM)
	##	Create Vmk if it didn't exist already (i.e., loaded.)
		Vmk = self.Vmk
		for mu in species.keys():
			Vmk.setdefault(mu, {})
			for nu in species.keys():
				Vmk[mu].setdefault(nu, None)

		if verbose >= 2:
			print '\t\tmaxM = %s, maxK = %s, eps = %s' % (maxM, maxK, self.V_eps)
			
	##	Helper function for fetching Vmk[mu][nu]'s
	##	Each Vmk[mu][nu] is a 6-array:  ab, cd, m k (or is set to None)
		def get_vmk(mu, nu):
			n_mu = len(species[mu])
			n_nu = len(species[nu])
			if Vmk[mu][nu] is None:
				t = np.zeros((n_mu, n_mu) + (n_nu, n_nu) + (2*maxM+1, 2*maxK+1))
				Vmk[mu][nu] = t
				return t
			else:
				return Vmk[mu][nu]
		###TODO!!! Fix the sizes, the code 'load' is actually incompatible with the following code.
			
	##	Look for Potentials in Vs, and generate the Vmk for each.
		if 'TK' in Vs:
			for (mu, nu), t in Vs['TK'].iteritems():
				if verbose >= 2: print "\t\tMaking TK potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				vmk = get_vmk(mu, nu)
				vmk += self.vmk_TK(t, maxM, maxK, bands[mu], bands[nu]) * magic_kappa_factor
				## Should we add t[0] to Vq0 ???
		
		if 'haldane' in Vs:
			#Haldane Pseudo-pot - array is strength of mth term
			for (mu, nu), v in Vs['haldane'].iteritems():
				if verbose >= 2: print "\t\tMaking Haldane pseudopotentials: (mu,nu) = (%s,%s), V_l = %s" % (mu, nu, v)
				if bands[mu].n > 1 or bands[nu].n > 1:
					raise NotImplementedError, "Haldane-pseudopotentials do not support LL mixing"
				vmk = get_vmk(mu, nu)
				vmk += self.vmk_haldane(v, maxM, maxK, bands[mu], bands[nu]) * magic_kappa_factor

		if 'rV(r)' in Vs:
			"""Real space interactions between mu/nu 
					(mu, nu) : {'rV', 'tol', 'd', 'coarse'}
			
				rV is function in 1 variable, r*V(r)
					or a string describing the function, e.g. rV = "lambda r: np.exp(-(r/6.)**2/2.) * r / np.sqrt(r**2 + 1.**2)"
				
				tol, d, coarse are completely optional and specify accuracy of Fourier transform; default values should be fine.
				See vq_from_rV() documentation for additional details.
			"""
			for (mu, nu), v in Vs['rV(r)'].iteritems():
				if verbose >= 2: print "\t\tMaking rV(r) potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v)
								
				d = v.setdefault('d', 1.)
				coarse = v.setdefault('coarse', 1)
				tol = v.setdefault('tol', 1e-10)
				#Calculate V(q) via Fourier integrals...
				Vq_radial = self.vq_from_rV(v['rV'], tol = tol, d = d, coarse = coarse)
				#Wrap V as function of two variables
				#def vq(qx, qy):
				#	return V(np.sqrt(qx**2+qy**2))
				vq = lambda qx,qy: Vq_radial(np.sqrt(qx**2+qy**2))
				#Treat as usual
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
			
		if 'coulomb' in Vs:
			for (mu, nu), v in Vs['coulomb'].iteritems():
				if verbose >= 2: print "\t\tMaking Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v)
				vq = functools.partial(self.vq_coulomb, v['v'], v['xi'])
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi

		if 'coulomb_blgated' in Vs:
			for (mu, nu), v in Vs['coulomb_blgated'].iteritems():
				if verbose >= 2: print "\t\tMaking Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v)
				vq = functools.partial(self.vq_coulomb_blgated, v['v'], v['d'])
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
				
				
		if 'GaussianCoulomb' in Vs:			
			"""
					(mu, nu) : { 'v', 'xi', 'order' }
				"""
			for (mu, nu), v in Vs['GaussianCoulomb'].iteritems():
				if verbose >= 2: print "\t\tMaking (Gaussian) Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v)
				xi_order = v.get('order', 0)
				if xi_order == 0:
					vq = functools.partial(self.vq_coulomb_gaussian, v['v'], v['xi'])
				elif xi_order == 1:
					vq = functools.partial(self.vq_coulomb_gaussian1, v['v'], v['xi'])
				elif xi_order == 2:
					vq = functools.partial(self.vq_coulomb_gaussian2, v['v'], v['xi'])
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
		
		if 'InterpolateLL' in Vs:
			"""Provides a potential which is \sum alpha[i]V[i], where V[i] is a Gaussian potential in LL i, and alpha is a constant
			allows for tuning between different LL
			This is deprecated since I have subclassed bands
			But if you want to do LL mixing maybe this could still be useful"""
			for (mu, nu), v in Vs['InterpolateLL'].iteritems():
				vmk=get_vmk(mu,nu)
				xi=v['xi']
				xi_order=set_var(v,'xi_order',0)
				vq=functools.partial(self.vq_coulomb_gaussian, 1, xi)
				print "alphas: ",v['alphas']
				bands=[LL([i]) for i in range(len(v['alphas']) ) ]
				for i, alpha in enumerate(v['alphas']):
					vmk+=alpha*self.Vmk_from_vq(vq,bands[i],bands[i])*magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
			
		#These are redundant; use TK they are faster. Just to check method
		if 'TK_vq' in Vs:
			for (mu, nu), t in Vs['TK_vq'].iteritems():
				if verbose >= 2: print "\t\tMaking TK_vq potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				vq = functools.partial(self.vq_TK, t)
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
		
		#These are redundant; use haldane they are faster. Just to check method
		if 'haldane_vq' in Vs:
			for (mu, nu), t in Vs['haldane_vq'].iteritems():
				if verbose >= 2: print "\t\tMaking haldane_vq potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t)
				vq = functools.partial(self.vq_haldanePP, t)
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor

				#self.Vq0[(mu, nu)] = self.Vq0[(nu, mu)] = np.nan
				

	##	(Anti)symmetrize Vmk
		if self.p_type == 'F': eta = -1
		if self.p_type == 'B': eta = 1

		for mu in species.keys():
			for nu in species.keys():
				if Vmk[mu][nu] is not None and mu == nu:
					if Vmk[mu][nu].shape[4] < 2*maxK+1:
						m = (Vmk[mu][nu].shape[4] - 1) // 2
						#Vmk[mu][mu] = np.pad(Vmk[mu][nu], ( (0,0),(0,0),(0,0),(0,0),(maxK-maxM,maxK-maxM),(0,0) ), mode='constant', constant_values=(0.,))
						Vmk[mu][mu] = pad(Vmk[mu][nu], maxK-maxM, 0, maxK-maxM, 0, axis = 4)

					Vmk[mu][mu] = 0.5 * (Vmk[mu][mu] + eta*np.transpose(Vmk[mu][mu], [2, 1, 0, 3, 5, 4]))

		self.make_V_mask(self.V_eps)
		#self.total_vq(Vs)
		if 'pickleVmk' in Vs:
			self.save_Vmk(Vs['pickleVmk'])
		if verbose >= 3: print '\t\tV(q=0):', self.Vq0
		#self.plot_Vmk()


#TODO, merge this with make_vmk
	def total_vq(self, Vs):
		""" Returns Vq[mu][nu](qx, qy), a list of list of functions
		"""
		species=self.species
		verbose=self.verbose
		#First, let each Vq[mu][nu] be list of functions
		self.Vq_list = Vq_list = {}
		for mu in species.keys():
			Vq_list.setdefault(mu, {})
			for nu in species.keys():
				Vq_list[mu].setdefault(nu, [])

		if 'rV(r)' in Vs:
			for (mu, nu), v in Vs['rV(r)'].iteritems():
				d = v.setdefault('d', 1.)
				coarse = v.setdefault('coarse', 1)
				tol = v.setdefault('tol', 1e-10)
				#An intepolating function; depends only on 'q'
				V = self.vq_from_rV(v['rV'], tol = tol, d = d, coarse = coarse)
				def vq(qx, qy):
					return V(np.sqrt(qx**2+qy**2))
				Vq_list[mu][nu].append(vq)
			
		if 'coulomb' in Vs:
			for (mu, nu), v in Vs['coulomb'].iteritems():
				vq = functools.partial(self.vq_coulomb, v['v'], v['xi'])
				Vq_list[mu][nu].append(vq)

		if 'coulomb_blgated' in Vs:
			for (mu, nu), v in Vs['coulomb_blgated'].iteritems():
				vq = functools.partial(self.vq_coulomb_blgated, v['v'], v['d'])
				Vq_list[mu][nu].append(vq)
		
		if 'GaussianCoulomb' in Vs:			
			for (mu, nu), v in Vs['GaussianCoulomb'].iteritems():
				#print('======================mu,nu,v',mu,nu,v)
				xi_order = v.get('order', 0)
				#print('======================mu,nu,v,xi_order',mu,nu,v,xi_order)
				if xi_order == 0:
					vq = functools.partial(self.vq_coulomb_gaussian, v['v'], v['xi'])
				elif xi_order == 1:
					vq = functools.partial(self.vq_coulomb_gaussian1, v['v'], v['xi'])
				elif xi_order == 2:
					vq = functools.partial(self.vq_coulomb_gaussian2, v['v'], v['xi'])
				Vq_list[mu][nu].append(vq)
		
		if 'TK_vq' in Vs:
			for (mu, nu), t in Vs['TK_vq'].iteritems():
				vq = functools.partial(self.vq_TK, t)
				Vq_list[mu][nu].append(vq)
		#These are redundant; use haldane they are faster. Just to check method
		if 'haldane_vq' in Vs:
			for (mu, nu), t in Vs['haldane_vq'].iteritems():
				vq = functools.partial(self.vq_haldanePP, t)
				Vq_list[mu][nu].append(vq)
		
		self.Vq = Vq = {}
		for mu in species.keys():
			Vq.setdefault(mu, {})
			for nu in species.keys():
				Vq[mu].setdefault(nu, [])
				
		for mu in Vq.keys():
			for nu in Vq[mu].keys():
				
				if len(Vq_list[mu][nu])==0:
					Vq[mu][nu] = None
				elif len(Vq_list[mu][nu])==1:
					Vq[mu][nu] = Vq_list[mu][nu][0]
				else:
					def summed(qx, qy):
						r = Vq_list[mu][nu][0](qy, qy)
						for f in Vq_list[mu][nu][1:]:
							r+=f(qx, qy)
						return r
					Vq[mu][nu] = summed
					#Vq[mu][nu] = lambda qx, qy: np.sum([ f(qx, qy) for f in self.Vq_list[mu][nu] ], axis = 0)
		
		return Vq
		

	def plot_Vmk(self):
		from matplotlib import pyplot as plt
		N = self.Nlayer
		species = self.species
		
		for mu in species.keys():
			n_mu = len(species[mu])
			for nu in species.keys():
				n_nu = len(species[nu])
				print('species',mu,nu)
				if self.Vmk[mu][nu] is not None:
					for a, b in itertools.product( xrange(n_mu), xrange(n_mu)):
						#plt.subplot(n_mu, n_mu, 1 + n_mu*a+b,label=str(nu)+str(mu))
						plt.imshow(np.abs(self.Vmk[mu][nu][a, b, 0, 0, :, :] ), cmap=plt.cm.jet, interpolation='nearest')
						plt.contour(self.V_mask[mu][nu][a, b, 0, 0, :, :], [0.5])
						#plt.colorbar()
						plt.title(str(np.count_nonzero(self.V_mask[mu][nu]))+'non-zero')
						#plt.savefig('/home/v/vasiliou/plotV'+str(nu)+str(mu)+'.png')
				
		plt.show()
		print('PLOTTING VMKS AND SAVING IN /home/v/vasiliou/plotV! REMOVE IF HASSLE')
		plt.savefig('/home/v/vasiliou/plotV.png')
		
		
	def make_V_mask(self, eps = 0.):
		"""Calculates elements of Vmk to keep in order to retain largest elements.
		
			For 0 < eps < 1, drop at most eps% of total weight (as measured by sum |Vmr|), but keeping at most Dmax values.
			
			If Dmax = None, D will be taken large enough to satisfy eps.
		"""
		print "V_mask was called"
		if eps >= 1. or eps < 0.:
			raise ValueError, "eps is wacky: " + str(eps)
					
		species = self.species
		mask = {}
		#Each Vmk[mu][nu] is array   ab, cd m k
		def get_mask(mu, nu):
		
			n_mu = len(species[mu])
			n_nu = len(species[nu])
			
			if self.Vmk[mu][nu] is None:
				return None
			else:
				return np.zeros( self.Vmk[mu][nu].shape, dtype = np.uint8)
				
		def keep_which(v):
			nrm = np.sum(np.abs(v))
			if nrm == 0.:
				return []
			v = (np.abs(v)/nrm).flatten()
			perm = np.argsort(v)
			perminv = np.arange(len(perm))[perm]
			running = np.cumsum(v[perm]) # Running sum of error due to dropping below ith element.
			keep = np.nonzero(running > eps)[0]
			if len(keep)>0:
				min = (v[perm])[keep[0]]
				keep = np.nonzero(v[perm] > min*0.999)
				return perminv[keep]
			else:
				return []
		self.V_mask = {}
		for mu in species.keys():
			self.V_mask[mu] = {}
			for nu in species.keys():
				self.V_mask[mu][nu] = V_mask = get_mask(mu, nu)
				if V_mask is not None:
					n_mu = len(species[mu])
					n_nu = len(species[nu])
					Vmk = self.Vmk[mu][nu]
					maxM, maxK = Vmk.shape[-2:]
					
					maxM = (maxM-1)/2
					maxK = (maxK-1)/2
					
					for a, b, c, d in itertools.product(xrange(n_mu), xrange(n_mu), xrange(n_nu), xrange(n_nu) ):
						vmk = Vmk[a, b, c, d].copy()
								
						keep_m0 = keep_which( vmk[maxM, :])
						keep_k0 = keep_which( vmk[:, maxK])
						vmk[maxM, :] = 0.
						vmk[:, maxK] = 0.
						keep = keep_which( vmk)
						for i in keep:
							V_mask[a, b, c, d, :, :].flat[i] = True
						V_mask[a, b, c, d, maxM, keep_m0] = True
						V_mask[a, b, c, d, keep_k0, maxK] = True
	
	"""
		The smoothness of a potential is determined by two length scales, which bound the short distance cutoff, 'd', and the long distance cutoff, 'xi'; specifically
	
			V(r) < a exp(-r/xi)
			V(q) < b exp(-q d)
			
		d = 0 is ok, but xi = infinity is trouble.
			
			
	"""

	def Vq_discretization_parameters(self, maxK):
		"""
			Determines discretization parameters for taking
				1) V(q) -> V_mk (Vmk_from_vq)
			and
				2) V(r) -> V_mk (from_vr)
		
			In both cases we discretize an integral
				int dqy   exp(-i qy k) f(qy)
			 
			using qy = np.linspace(0, self.Lx/2.*mul, n*mul+1); so compute here 'n' and 'mul.'
			
			In case 2), we discretize an integral
				\int_0^\inf dr 2 pi r  J_0(q r) V(r)
			using r = ???????????????
		"""
		#Since maxK determines the falloff after FT (the bandwidth), maxK also determines how fine the discretization must be. We set the Nyquist Frequency to be 1.5 x  bandwidth.
		q_max = 9.
		mul = 1
		while mul*self.Lx/2 < q_max:
			mul+=1
		n = int(1.5*maxK)
				
		return q_max, mul, n
		

	def Vmk_from_vq(self, vq, mu, nu):
		""" Builds Vmk for a particular species-species interaction given the Fourier transformed interaction vq(qx, qy) (see notes).
		
			self.maxM, self.maxK - extent in m, k
			mu, mu - either a species key, or a 'bands' object
			The bands have properties
				F - `form factors' F(a, b, qx, qy) is form factor for band a--b
				eta - the parities (odd even) of the bands
				
			Vabcd_filter[(mu, nu)][a, b, c, d] = bool is set, an optional filter specifying which matrix elements types should be computed.
				When false, the value of Vmk is illdefined (not necessarily 0)
		
		"""

		if self.p_type!='F': raise NotImplementedError,"not yet set up to do Vmk_from_vq for bosons, only pseudopotentials can be used"

		kappa = self.kappa
		
		if isinstance(mu, bands):
			b1  = mu
			mu = None
		else:
			b1 = self.bands[mu]
		if isinstance(nu, bands):
			b2  = nu
			nu = None
		else:
			b2 = self.bands[nu]
		
		if mu is not None and nu is not None and (mu, nu) in self.Vabcd_filter:
			filter = self.Vabcd_filter[(mu, nu) ]
		else:
			filter = None

		P1=b1.P
		P2=b2.P
		F1=b1.F
		F2=b2.F
		pre_F1=b1.pre_F
		pre_F2=b2.pre_F
		
		n_mu = b1.n
		n_nu = b2.n
		maxM, maxK = self.maxM, self.maxK
		
		vmk = np.zeros( (n_mu, n_mu, n_nu, n_nu, 2*maxM+1, 2*maxK+1) )

		M = self.kappa*np.arange(-maxM, maxM+1)
		K = self.kappa*np.arange(-maxK, maxK+1)
		
		#Keep track of which a,b,c,d have been computed (many are related by symmetries)
		did = np.zeros( (n_mu, n_mu, n_nu, n_nu), dtype = np.bool )
		
		#Do the two species have the exact same bands?
		same_bands = (b1==b2)
		if self.verbose >= 2:
			print "\t\tComputing Vmk [a b c d]: ",		# no new line
		
		#Determine integration bounds in qy - must be commensurate with Lx. We integrate out to q_max
				
		q_max, mul, n  = self.Vq_discretization_parameters(maxK)
		qy = np.linspace(0, self.Lx/2.*mul, n*mul+1)
		dqy = self.Lx/2./n

		#implements Eq. 2 from arxiv 1307.2909, which gives a phenomenological formula for the LL mixing in graphene
		if self.graphene_mixing <0.25 and self.graphene_mixing is not None:
			print "Warning! I'm not sure that LL mixing in graphene with kappa<0.25 is supported, maybe don't bother"
		
		def vq_mixed(m,q):
			if self.graphene_mixing is not None: 
#				print m
#				print q
#				print vq(m,q)*np.sqrt(m**2)/np.sqrt(m**2)
				if abs(m)>1e-15: return vq(m,q)*np.sqrt(m**2+q**2)/(np.sqrt(m**2+q**2)+4*np.log(4*self.graphene_mixing)*np.tanh(0.5*(m**2+q**2)))
				else: 
					outarray=np.zeros(len(q))
					outarray[0]=vq(m,q[0])
					outarray[1:]=vq(m,q[1:])*np.sqrt(m**2+q[1:]**2)/(np.sqrt(m**2+q[1:]**2)+4*np.log(4*self.graphene_mixing)*np.tanh(0.5*(m**2+q[1:]**2)))
					return outarray
			else: raise ValueError,"something went wrong with vq_mixed"

		#The integrand obeys f(qy) = f(-qy)^\ast
		def f(qy, M, a, b, c, d):
			if self.graphene_mixing is not None: return F1(a, b, M, qy)*F2(c, d, -M, -qy)*vq_mixed(M, qy)
			else: return F1(a, b, M, qy)*F2(c, d, -M, -qy)*vq(M, qy)
		
		#Only evaluate function out to keep, then pad with 0s
		vr = np.zeros(len(qy), dtype=np.float)
		vz = np.zeros(len(qy), dtype=np.complex)
		keep = int(q_max/dqy) #No need to evaluate function beyond here.
		qy = qy[:keep]

		for a, b, c, d in itertools.product(xrange(n_mu), xrange(n_mu), xrange(n_nu), xrange(n_nu) ):
			
			if filter is not None and not filter[a, b, c, d]:
				continue
			
			if did[a, b, c, d]:
				if self.verbose >= 3: print '(%s, %s, %s, %s)*, ' % (a, b, c, d),
				continue
			else:
				if self.verbose >= 3: print '(%s, %s, %s, %s), ' % (a, b, c, d),
			
			#The following bands are related by symmetries
			did[a, b, c, d] = True
			did[b, a, d, c] = True
			if same_bands:
				did[c, d, a, b] = True
				did[d, c, b, a] = True
			
			#An m/k independent prefactor - we seperate it from the numerical integration.
			pre = pre_F1(a, b)*pre_F2(c, d)*dqy/np.sqrt(2*np.pi)
			P = P1(a, b)*P2(c, d) #Parity of sector
			
			#Only need upper quadrant in mk space - rest related by symmetry

			for m in range(maxM+1): #Only upper corner
				y = f(qy, M[m], a, b, c, d)
				if y.dtype == np.float:
					vr[:keep] = y
					y = np.fft.hfft(vr)[::mul]
				else:
					vz[:keep] = y
					y = np.fft.hfft(vz)[::mul]
				
				y = np.concatenate((y[-maxK:], y[:maxK+1]))
				y *= pre
				if np.abs(y[-1]) > self.V_eps*0.01:
					print "\n\tfrom_vq(): Tail end of the Vmk's aren't small enough!  m, M[m], v[-5:] =", m, M[m], y[-5:]
					if np.abs(y[-1]) > 1e-6:
						print "It means that the tail end of the matrix elements is  not negligible, and more matrix elements should be calculated.\n\n        This will occur if either a) xiK is too small  b) you use an interaction which is\nactually long range (for instance, when you combine gaussian with layer width\n(remember?)\n\nM"

				vmk[a, b, c, d, m, :] =  y
				vmk[b, a, d, c, -1 - m, :] = y

				if a!=b or c!=d:
					vmk[b, a, d, c, m, :] = y[::-1]*P
					vmk[a, b, c, d, -1 - m, :] = y[::-1]*P
				
				if same_bands:
					vmk[c, d, a, b, m, :] = y*P
					vmk[d, c, b, a,  -1 - m, :] = y*P
					if a!=b or c!=d:
						vmk[d, c, b, a,  m, :] = y[::-1]
						vmk[c, d, a, b, -1 - m, :] = y[::-1]
		#plt.show()
		if self.verbose >= 2:
			print 		# terminate the line from "Computing Vmk [a b c d]"
		
		return vmk
		

	def vq_from_rV(self, rV, tol=1e-12, d = 1., coarse = 1):
		"""	Given function rV = r*V(r), the scaled real space potential, returns an interpolated Fourier transform in the variable |q| (see interp1d).

			The domain is set by Vq_discretization_parameters, guaranteed to be sufficient for use in Vmk_from_vq.
			
			rV: a 1-parameter callable function, or a string that represents one.
				For example:
				>>>	rV = lambda r: np.exp(-r/4)
				>>>	rV = "lambda r: np.exp(-(r/6.)**2/2.) * r / np.sqrt(r**2 + 1.**2)"
				>>>	rV = "lambda r: mQHfunc.rVr_InfSqWell(2.0)(r) * np.exp(-r)"
				
				Note 1: It's advantageous to use a string for rV, since it can be pickled and is human-readable in the log files.
				Note 2: rV(r) needs to decay sufficiently fast and cannot be a long-ranged potential.

			tol: absolute accuracy of each Fourier integral.
			d: an estimate of the short distance softening which might help in the accurate evaluation og the Fourier integrals;
			coarse > 1: implies that V(q)-values are computed on a coarser grid than that use by Vmk_from_vq, so the result will depend more on the accuracy of the interpolation.
		"""
		from scipy.interpolate import interp1d
		maxM, maxK = self.maxM, self.maxK
		#Determine exactly compute points vq(q) by the points 'qs' which will be requested in Vmk_from_vq
		q_max, mul, n = self.Vq_discretization_parameters(maxK)
		dq = self.Lx/2./n * coarse
		q_max = np.sqrt(q_max**2 + (self.kappa*maxM)**2) #Maximum 1-value
		qs = np.arange(0., q_max+dq, dq)			# sampling
		
		if isinstance(rV, str):		# convert string to a function if needed be
			rV = eval(rV, {"__builtins__":None, "np":np, "mQHfunc":mQHfunc}, {})		# kind of dangerous code
		
		if self.verbose >= 3:		# Talkative mode
			print "\t\tvq_from_rV is calculating " + str(len(qs)) + " Fourier points in [0, " + str(qs[-1]) + "]",
			vq = np.zeros(len(qs), dtype=float)
			for i,q in enumerate(qs):
				vq[i] = mQHfunc.J0_integral(rV, q, tol=tol, d=d)
				if i % 100 == 99: print ".",		# comes with progress indicator!
			print
		else:
			vq = np.array([ mQHfunc.J0_integral(rV, q, tol=tol, d=d) for q in qs ])
		return interp1d(qs, vq, kind = 'cubic')


	def vq_coulomb(self, v, xi, qx, qy):
		"""	Real space potential V(r) = e^{-r/xi}/r  (Yukawa). """
		return (2*np.pi*v)/np.sqrt(1./xi**2 + qx**2 + qy**2)
	
	@classmethod
	def vq_coulomb_gaussian(cls, v, xi, qx, qy):
		"""	Coulomb interaction, with Gaussian damping.
				V(r) = v e^(-r^2/2xi^2) / r
			"""
		q2 = qx**2 + qy**2
		#return (np.sqrt(2 * np.pi**3) * xi) * np.exp(-layersep * np.sqrt(q2)) * sp.special.ive(0, q2 * (xi**2 / 4.))
		return (v*np.sqrt(2 * np.pi**3) * xi) * sp.special.ive(0, q2 * (xi**2 / 4.))
	
	@classmethod
	def vq_coulomb_gaussian1(cls, v, xi, qx, qy):
		"""	V(r) = v F1(r^2/2xi^2) / r,   with F1(z) = (1 + z) e^{-z}
				This can be derived from Vq0 = vq_coulomb_gaussian, with Vq1 = Vq0 + (xi/2) d/d(xi) Vq0.
			"""
		qxi = (qx**2 + qy**2) * xi**2 / 4.
		return (v * np.sqrt(2 * np.pi**3) * xi) * ( (1.5-qxi) * sp.special.ive(0, qxi) + qxi * sp.special.ive(1, qxi) )

	@classmethod
	def vq_coulomb_gaussian2(cls, v, xi, qx, qy):
		"""	V(r) = v F2(r^2/2xi^2) / r,   with F2(z) = (1 + z + z^2/2) e^{-z} """
		qxi = (qx**2 + qy**2) * xi**2 / 4.
		return (v * np.sqrt(2 * np.pi**3) * xi) * ( (1.875 - 2.5*qxi + qxi**2) * sp.special.ive(0, qxi) + (2*qxi - qxi**2) * sp.special.ive(1, qxi) )
	
	@classmethod
	def vq_coulomb_blgated(self, v, d, qx, qy):
		"""	 
			A coulomb potential with metallic gates placed above and below the gas at distance 'd.' The gates screen the interaction,
			
			V(r) = 1/r + 2 sum_{n>0} [ (-1)^n / sqrt{r^2 + (d*n)^2} ]
			
		"""
		q = np.sqrt(qx**2 + qy**2)
		qd = d*q
		qd2 = qd*qd
		mask = qd > 0.1
		v = (2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, (1./2)*(1 - (1./12)*qd2*(1. - (1/10.)*qd2*(1. - (17./168.)*qd2*(1.-(31./306.)*qd2)))))
		"""from matplotlib import pyplot as plt
		plt.plot(q, v)
		plt.plot(q, 2*np.pi/q, '--')
		plt.ylim([0, 8])
		plt.show()"""
		#1/0
		return v
	
	def vq_haldanePP(self, Vm, qx, qy):
		"""	Haldane pseudopotentials.  Vm is a list of coeffcients. """
		q2 = qx**2+qy**2
		v = np.zeros_like(q2)
		for n in range(len(Vm)):
			v+= 2*Vm[n]*sp.special.eval_laguerre(n, qx**2+qy**2)
		return v
			
	def vq_TK(self, t, qx, qy):
		"""	Trugman-Kivelson pseudopotentials.  t is a list of coeffcients, [t0, t1, ... ]
			
			V(q) = \sum_j t_j (-q^2)^j
			
		"""
		nq2 = -1*(qx**2 + qy**2)
		#return np.dot(t, nq2**range(len(t)))
		#t0 + q t1 + q^2 t2 . . . =
		#t0 + q( t1 + q( t2 + t3 q ... ) )
		if len(t) > 1:
			v = t[-1]*nq2
		else:
			return t[0]
			
		for n in np.arange(len(t)-2, 0, -1):
			v+=t[n]
			v*=nq2
		v += t[0]

		return v
	

	def vmk_TK(self, t, maxM, maxK, b1, b2):
		"""
            Calculates matrix elements for Trugman-Kivelson pseudopotentials analytically.
			I never want to think about this again. MZ
		"""

		from numpy.polynomial import Polynomial as P
		from numpy.polynomial import Laguerre as L
		from numpy.polynomial import hermite as H
		
		n_mu = b1.n
		n_nu = b2.n
		
		vmk = np.zeros( (n_mu, n_mu, n_nu, n_nu, 2*maxM+1, 2*maxK+1) )

		def F(a, b, m, conj):
			z = np.poly1d([np.sqrt(2.), m])*conj
			zb = np.poly1d([-np.sqrt(2.), m])*conj
			if a > b:
				l = sp.special.genlaguerre(b, a-b)*1.
				l = float(np.sqrt(1.*sp.misc.factorial(b, exact=True)/sp.misc.factorial(a, exact=True))*np.sqrt(2)**(b-a))*z**(a-b)*l(0.5*zb*z)
				return P(l.coeffs[::-1])
			else:
				l = sp.special.genlaguerre(a, b-a)*1.
				pre = np.sqrt(1.*sp.misc.factorial(a, exact=True)/sp.misc.factorial(b, exact=True))*np.sqrt(2)**(a-b)
				l =  float(pre)*((-zb)**(b-a)*l(0.5*zb*z))
				return P(l.coeffs[::-1])
			
		M = self.kappa*np.arange(-maxM, maxM+1)
		K = self.kappa*np.arange(-maxK, maxK+1)
		for a, b, c, d in itertools.product(xrange(n_mu), xrange(n_mu), xrange(n_nu), xrange(n_nu) ):
			for m in range(2*maxM+1):
				form_factor = F(b1[a], b1[b], M[m], 1)*F(b2[c], b2[d], M[m], -1)
				p = P([0])
				for j in range(len(t)):
					if t[j]!=0.:
						p += t[j]*P([-M[m]**2, 0, 2])**j
						
				p = p*form_factor
				v = H.hermval(K/np.sqrt(2), p.coef*(-0.5)**np.arange(0, p.degree()+1 ))
				v = v*np.exp(-0.5*(M[m]**2 + K**2))
				
				vmk[a, b, c, d, m, :] = v
			
		return vmk
		

	def vmk_haldane(self, Vm, maxM, maxK, b1, b2):
		""" 
            Calculates matrix elements for Haldane pseudopotentials analytically.
            I never want to think about this again. MZ
		"""
		from numpy.polynomial import Polynomial as P
		from numpy.polynomial import Laguerre as L
		from numpy.polynomial import hermite as H
		
		if b1.n > 1 or b2.n > 1:
			raise NotImplemented
			
		Vmk = np.zeros( (2*maxM+1, 2*maxK+1))
		M = self.kappa*np.arange(0, maxM+1)
		K = self.kappa*np.arange(0, maxK+1)
		
		for m in range(maxM+1):
			p = P([0])
			for j in range(len(Vm)):
				if Vm[j]!=0.:
					p += Vm[j]*L.basis(j)(P([M[m]**2, 0, -2]))
					
			v = H.hermval(K/np.sqrt(2), p.coef*(-0.5)**np.arange(0, p.degree()+1 ))
			v = v*np.exp(-0.5*(M[m]**2 + K**2))
			
			Vmk[maxM + m, maxK:] = v
			Vmk[maxM + m, :maxK] = v[-1:0:-1]
			Vmk[maxM - m, maxK:] = v
			Vmk[maxM - m, :maxK] = v[-1:0:-1]

		return 2*np.reshape(Vmk, (1, 1) + Vmk.shape)

	def make_Ht(self):
		"""Makes 2-site Hamiltonian for the onsite and hopping part of problem"""
		H = []
		if self.Nlayer > 1:
			raise NotImplemented

		T = self.Trm[0][0][0, 0]
		mu = T[:, 0]
			
		if T.shape[1] > 1:
			t = T[:, 1]
			hop = True
		else:
			hop = False
			
		nOp = self.nOp[0]
		Id = self.Id[0]
		aOp = self.aOp[0]
		AOp = self.AOp[0]
		for j in range(self.L):
			
			h = 0.5*mu[j]*np.outer( nOp,  Id) + 0.5*mu[(j+1)%self.L]*np.outer( Id,  nOp)
			if hop:
				h+= (t[(j+1)%self.L]*np.outer( AOp,  aOp) + t[(j+1)%self.L]*np.outer( aOp,  AOp))
			
			h = h.reshape((2, 2, 2, 2))
			h = h.transpose( [0, 2, 1, 3]).copy()
			H.append(h)
		
		if self.bc!='periodic':
			H[0]+=0.5*mu[0]*np.kron( self.nOp,  self.Id)
			H[L-2]+=0.5*mu[L-1]*np.kron(self.Id, self.nOp)

		self.H = H

#	@staticmethod
#	def static_translate_Q1_Kvec(Qind, Kvec, Nlayer, r):
#		"""	Does the translateQ given Kvec.
#			This shouldn't get used as translate_Q1, rather one should use:
#			>>> self.translate_Q1 = lambda Q, r: QH_model.static_translate_Q1_Kvec(Q, self.Kvec, self.Nlayer, r)
#			"""
#			
#		Q = Qind.copy()
#		if r % Nlayer: raise ValueError
#		if Q.ndim == 1:	# a single charge vector
#			Q[0] += r // Nlayer * np.dot(Q[1:], Kvec)
#		else:		# a table / q_ind
#			charge_idx = Q.shape[-1] - len(Kvec)
#			Q[:, charge_idx-1] += r // Nlayer * np.dot(Q[:, charge_idx:], Kvec)
#		return Q
	

	@staticmethod
	def translate_Q1_Kvec(Qind, r, Kvec, Nlayer):
		"""	Does the translateQ given Kvec.
			This shouldn't get used as translate_Q1, rather one should use:
			>>>	self.translate_Q1 = functools.partial(QH_model.translate_Q1_Kvec, Kvec = self.Kvec, Nlayer = self.Nlayer)
			"""
		Q = Qind.copy()
		if r % Nlayer: raise ValueError
		if Q.ndim == 1:	# a single charge vector
			Q[0] += r // Nlayer * np.dot(Q[1:], Kvec)
		else:		# a table / q_ind
			charge_idx = Q.shape[-1] - len(Kvec)
			Q[:, charge_idx-1] += r // Nlayer * np.dot(Q[:, charge_idx:], Kvec)
		return Q


	def compatible_translate_Q1_data(self, tQdat, verbose=0):
		"""	Check if self.translate_Q1_data matches tQdat.  Returns True/False.	"""
	#	ds = self.translate_Q1_data
	#	if ds is None and tQdat is None: return True
	#	if ds['cls'] != tQdat['cls'] or ds['module'] != tQdat['module'] or ds['tQ_func'] != tQdat['tQ_func']:
	#		if verbose > 0: print 'translate_Q1_data mismatch:\n%s\n%s\n' % (ds, tQdat)
	#		return False
	#	if np.any(ds['keywords']['Kvec'] != tQdat['keywords']['Kvec']):
	#		if verbose > 0: print 'translate_Q1_data mismatch:\n%s\n%s\n' % (ds['keywords']['Kvec'], tQdat['keywords']['Kvec'])
	#		return False
	#	return True
		return identical_translate_Q1_data(self.translate_Q1_data, tQdat, verbose=verbose)


	def make_Hilbert_space(self):
		"""	Set variables, charges, and operators.
				
			Operators: Id, StrOp, aOp (lowering), AOp (raising), nOp.
				Each one an array (N long).
			"""
		N = self.Nlayer		# number of species
		L = self.L		# unit cell size
		assert L % N == 0
		if self.p_type == 'F':
		##	Hilbert space & Operators
			self.d = 2 * np.ones(L, dtype=np.int)

			self.Id = [ np.eye(2) for s in range(N) ]
			self.StrOp = [ np.diag([1., -1]) for s in range(N) ]
			self.nOp = [ np.diag([0., 1]) for s in range(N) ]
			self.nOp_shift=[self.nOp[s]-0.5*self.Id[s] for s in range(N)]#shifted density so that I can easily calculate particle-hole symmetric things using your correlation functions, SG
			self.AOp = [ np.diag([1.], -1) for s in range(N) ]		# creation
			self.aOp = [ np.diag([1.], 1) for s in range(N) ]		# annihilation
			self.invnOp = [ np.diag([1., 0]) for s in range(N) ]
			self.site_ops = ['Id', 'StrOp', 'nOp', 'AOp', 'aOp', 'invnOp','nOp_shift']

			self.AOp_l = np.outer( self.AOp[0], self.Id[0] ).reshape(2, 2, 2, 2).transpose([0, 2, 1, 3])
			self.AOp_r = np.outer(self.Id[0], self.AOp[0] ).reshape(2, 2, 2, 2).transpose([0, 2, 1, 3])
			self.bond_ops = ['AOp_l', 'AOp_r']
			
			self.Qp_flat = np.zeros([L, 2, self.num_q], np.int)
			print('----'*100)
			print('----'*100)
			print('ISINSTANXE',isinstance(self.Id[0],np.ndarray))
			print('----'*100)
			print('----'*100)
				

		elif self.p_type == 'B':
		##	Hilbert space & Operators
			d = self.N_bosons + 1		# assume all the d's are the same
			self.d = d * np.ones(L, dtype=np.int)
			self.Id = [np.eye(d) for s in range(N)]
			self.StrOp = self.Id
			self.nOp = [np.diag(range(d)) for s in range(N)]
			aOp=np.zeros((d,d))
			AOp=np.zeros((d,d))
			for i in range(d-1):
				aOp[i,i+1]=np.sqrt(i+1)
				AOp[i+1,i]=np.sqrt(i+1)
			self.aOp = [aOp for s in range(N)]
			self.AOp = [AOp for s in range(N)]
			self.AAOp = [np.dot(AOp,AOp) for s in range(N)]	
			self.aaOp = [np.dot(aOp,aOp) for s in range(N)]	
			self.AAaaOp = [ nOp**2-nOp for nOp in self.nOp ]
			self.site_ops = ['Id', 'StrOp', 'nOp', 'AOp', 'aOp', 'AAOp', 'aaOp', 'AAaaOp']
			#make AA, aa, NN, other more complicated versions for multiple levels/lack of momentum conservation
			self.Qp_flat = np.zeros([L, d, self.num_q], dtype=np.int)


	##	Fill in charges
	##	Get total filling (p/q) in unit cell from root_config
	##		Qmat is 2-array shaped (num_cons_Q, N), denoting unscaled charges of each layer
	##		root_config is a 2-array shaped (#, N)  (# need not equal L/N)
	
		ps = np.sum(np.tensordot(self.Qmat, self.root_config, axes = [[1], [1]]), axis = 1) #ith total charge in root_config
		print('================',ps)
		q = self.root_config.shape[0] * self.root_config.shape[1]		# number of sites
		self.filling = [ (ps[i], q) for i in range(len(ps)) ]		#p/q for each charge
		print('FILLINGS=======',self.filling)

		do_cons_K = int(self.cons_K > 0) # a true or false statement. Either you dont or you do. Meaning either 0 or 1
		#print('Conserve K =======================================',self.cons_K,do_cons_K)		# 0 or 1
		for s in range(L):		# runs over sites
			for n in range(self.d[s]):		# run over Hilbert space (empty/full)
				for i in range(len(ps)):	# runs over num_cons_Q
					self.Qp_flat[s, n, i+do_cons_K] = n*q*self.Qmat[i, s%N] - ps[i]
					#C = q N - p
				if self.cons_K == 1:
					self.Qp_flat[s, n, 0] = int(s//N) * np.dot(self.Kvec, self.Qp_flat[s,n,1:])
					#K = j (q N - p)
				elif self.cons_K>1:
					#Don't scale or shift K
					self.Qp_flat[s, n, 0] = int(s//N) * np.dot(self.Kvec, n*self.Qmat[:, s%N])
		#print('----------------------------------',L,'---',self.num_q,'----',self.Qp_flat)
		
		#TEST ME
		#print q, self.mod_q
	##	Rescale mod_q by q; skip K cons since it isn't scaled if cons_K!=1
		for j in range(do_cons_K, self.num_q):
			if self.mod_q[j] > 1:
				self.mod_q[j]*=q
		#print q, self.mod_q
		#quit()


	################################################################################
	################################################################################
	##	Convention for MPOgraph (see model.py for full description)
	#TODO

	##	Helper functions
	def addNodes(self, nodeheader, start_layer, length, start_node, start_Op, StrOp):
		"""	Construct the chain: nodes i -> i+1, and start_node -> node 1.  (nodes labelled 1 to length)
				Return pair: last_layer, last_node
			
				The start node is on start_layer, the end node would then be in (start_layer + length) % len(MPOgraph).
				The nodes are labelled by nodeheader + (i,),
				start_Op (a tuple) connects start_node to node 1, each subsequent pair of nodes are connected by StrOp (a string).
			"""
		MPOgraph = self.MPOgraph
		N = len(MPOgraph)
		after_node = None
		for i in range(length, 0, -1):	# scan backwards: length to 1
			site = (start_layer + i) % N
			node = nodeheader + (i,)
			stoploop = False
			try:
				XNode = MPOgraph[site][node]
				stoploop = True
			except KeyError:
				XNode = MPOgraph[site][node] = {}
			if after_node is not None:
				XNode[after_node] = [(StrOp, 1.)]
			if stoploop: break		# node already exist, so no need to make more
			after_node = node
		if length > 0 and (not stoploop):
			XNode = MPOgraph[start_layer][start_node]
			XNode[nodeheader + (1,)] = [start_Op]		# no need to append to the list.
		if length > 0: return (start_layer+length) % N, (nodeheader + (length,))
		if length == 0: return start_layer, start_node


	def addNodesB(self, nodeheader, end_layer, length, end_node, end_Op, StrOp):
		"""	Construct the chain: nodes i -> i-1, with node 0 -> end_node.  (nodes labelled 0 to length-1)
				Return pair: first_layer, first_node
			
				The end node is on end_layer, the first node would then be in (end_layer - length + 1) % len(MPOgraph).
				The nodes are labelled by nodeheader + (i,),
				Pair of nodes are connected by StrOp (a string), the last node (node 0) connects to end_node via end_Op (a tuple).
			"""
		MPOgraph = self.MPOgraph
		N = len(MPOgraph)
		for i in range(length-1, -1, -1):	# scan backwards: length-1 to 0
			site = (end_layer - i) % N
			node = nodeheader + (i,)
			stoploop = False
			try:
				XNode = MPOgraph[site][node]
				stoploop = True
			except KeyError:
				XNode = MPOgraph[site][node] = {}
			if i > 0:
				XNode[nodeheader + (i-1,)] = [(StrOp, 1.)]
			else:
				XNode[end_node] = [end_Op]
			if stoploop: break		# node already exist, so no need to make more
		if length > 0: return (end_layer-length+1) % N, (nodeheader + (length-1,))
		if length == 0: return end_layer, end_node



	################################################################################
	################################################################################
	##	Convention for Vmk_list:
	##		There are eight columns, [ m, k, alpha, beta, gamma, delta, V_{m,k}, must_keep ] which represents the Hamiltonian:
	##
	##		H = \sum_{j,l,l'} V_{l,l'} \psi^\dag_{j+l;alpha} \psi^\dag_{j-l;beta} \psi_{j-l',gamma} \psi_{j+l',delta}
	##		  = \sum_{j,m,k} V_{m,k} \psi^\dag_{j+k;alpha} \psi^\dag_{j+m;beta} \psi_{j,gamma} \psi_{j+k+m,delta}
	##
	##		where m = l'-l, k = l'+l.  ( Or equivalently, l' = (k+m)/2, l = (k-m)/2 )
	##			* Note that l, l' may be positive or negative.
	##			* Hermitian conjugate terms are explicitly included in Vmk_list.
	def convert_Vmk_tolist(self, drop0 = True):
		"""	Convert array-style Vmk to (num x 7) list of non-zero Vs.
			Takes in self.Vmk and writes to self.Vmk_list
			Vmk_list explicitly store -m's.
			"""
		Vmk = self.Vmk
		V_mask = self.V_mask
		
		Vmk_list = []
		species = self.species
		for mu in Vmk.keys():
			for nu in Vmk.keys():
				vmk = Vmk[mu][nu]
				mask = V_mask[mu][nu]
				if vmk is not None:
					maxM = (vmk.shape[-2]-1)//2
					maxK = (vmk.shape[-1]-1)//2
					
					#t0 = time.time()
					packVmk(Vmk_list, vmk, mask, maxM, maxK, species[mu], species[nu], tol = 1e-12)
					
				####	Below is what packVmk does.
				#	n_mu = len(species[mu])
				#	n_nu = len(species[nu])
				#	spec1 = species[mu]
				#	spec2 = species[nu]
				#	for a, b, c, d, m, k in itertools.product( xrange(n_mu), xrange(n_mu), xrange(n_nu), xrange(n_nu), xrange(0, 2*maxM+1), xrange(0, 2*maxK+1) ):
				#		v = vmk[a, b, c, d, m, k]
				#		if abs(v) > tol:
				#			keep = mask[a, b, c, d, m, k]
				#			Vmk_list.append([ m-maxM, k-maxK, spec1[a], spec2[c], spec2[d], spec1[b], v, keep ])

		self.Vmk_list = Vmk_list


	def make_split_Vmk_list(self, verbose, force_keep):
		"""	Take Vmk_list and make splitV_list, where the layers are now split.
			splitV_list has 9 columns, storing data for creation/annihilation operators.
			"""
		N = self.Nlayer
		L = self.L

	##	Modify the Vmk_list so m,r now represents distance in sites (as opposed to N-sites)
		if self.p_type == 'F': swapph = -1
		if self.p_type == 'B': swapph = 1
		self.splitV_list = splitV_list = []
		for Vmk in self.Vmk_list:
			m,k,ly0,ly1,ly2,ly3,Vstrength,must_keep = Vmk
			Op = ['A0', 'A1', 'a2', 'a3']		# The Vmk_list is written for a3 (lowering) operator applied first and A0 (raising) last.
			loc = np.array([ly0 + N * k, ly1 + N * m, ly2, ly3 + N * (m+k)], int)
		##	Have to sort the locations from smallest to largest
			order = np.argsort(loc)
			locdist = loc[order][1:] - loc[order][:-1]
			ly = np.array([ly0, ly1, ly2, ly3])[order]
			#print loc, 'order', order, 'locdist', locdist, 'layer', ly
		##	Get the sign.
		##	The right most operator is applied first, left most last.
		##	Since argsort is stable, when you operators lie on the same site, A is to the left of a.
			if swapph == -1: Vstrength *= perm_sign(order)
			splitV_list.append([(Op[order[0]], ly[0]), locdist[0], (Op[order[1]], ly[1]), locdist[1], \
				(Op[order[2]], ly[2]), locdist[2], (Op[order[3]], ly[3]), Vstrength, must_keep or force_keep])
			if verbose >= 6: print ' ', Vmk, '->', splitV_list[-1]



	##################################################	
	def make_MPOgraph1_from_Vmk_old(self, verbose=0, force_keep=False):
		"""	Given Vmk_list, make the MPOgraph1.  Make a node for each (m,r) """
		if verbose >= 1: print "Making MPOgraph1 from Vmk..."
		N = self.Nlayer
		L = self.L
		self.make_split_Vmk_list(verbose, force_keep)
	##	The elements of MPOgraph1[s] labels the states to the *left* of site s, so Op's in MPOgraph1[s] always sit in site s.
		try:
			MPOgraph1 = self.MPOgraph1
		except AttributeError:
			MPOgraph1 = self.empty_MPOgraph(N)
		for G in MPOgraph1:		# put keep flags
			G['R']['&&keep'] = True
			G['F']['&&keep'] = True

	##	Helpers to make 'strings nodes', have a 'must_keep' feature
		def addNodes(nodeheader, start_layer, length, start_node, start_Op, StrOp, must_keep):
			"""	Construct the chain: nodes i -> i+1, with start_node -> node 1.  (nodes labelled 1 to length) """
			after_node = None
			for i in range(length, 0, -1):	# scan backwards: length to 1
				site = (start_layer + i) % N
				node = nodeheader + (i,)
				stoploop = False
				try:
					XNode = MPOgraph1[site][node]
					if not must_keep: stoploop = True		# all the previous nodes already exist
					if must_keep and '&&keep' in XNode: stoploop = True		# node already marked with keep
				except KeyError:
					XNode = MPOgraph1[site][node] = {}
				if must_keep: XNode['&&keep'] = True
				if after_node is not None:
					XNode[after_node] = [(StrOp, 1.)]
				if stoploop: break		# node already exist, so no need to make more
				after_node = node
			if not stoploop:
				XNode = MPOgraph1[start_layer][start_node]
				XNode[nodeheader + (1,)] = [start_Op]		# no need to append to the list.
		
		def addNodesB(nodeheader, end_layer, length, end_node, end_Op, StrOp, must_keep):
			"""	Construct the chain: nodes i -> i-1, with node 0 -> end_node.  (nodes labelled 0 to length-1) """
			for i in range(length-1, -1, -1):	# scan backwards: length-1 to 0
				site = (end_layer - i) % N
				node = nodeheader + (i,)
				stoploop = False
				try:
					XNode = MPOgraph1[site][node]
					if not must_keep: stoploop = True		# all the subsequent nodes already exist
					if must_keep and '&&keep' in XNode: stoploop = True	# node already marked with keep
				except KeyError:
					XNode = MPOgraph1[site][node] = {}
				if i > 0:
					XNode[nodeheader + (i-1,)] = [(StrOp, 1.)]
				else:
					XNode[end_node] = [end_Op]
				if must_keep: XNode['&&keep'] = True
				if stoploop: break		# node already exist, so no need to make more
		
	##	Make the graph from splitV_list (12 columns)
	##	Hopping is of form: Vstrength * Op4_{m1+r+m2} Op3_{m1+r} Op2_{m1} Op1_{0}
		for V in self.splitV_list:
			(Op1,ly1),m1,(Op2,ly2),r,(Op3,ly3),m2,(Op4,ly4),Vstrength,keep = V
			print (Op1,ly1),m1,(Op2,ly2),r,(Op3,ly3),m2,(Op4,ly4),Vstrength,keep
			Op1 = Op1[0]; Op2 = Op2[0]; Op3 = Op3[0]; Op4 = Op4[0];
			if np.abs(Vstrength) == 0: continue
			#print ' ', V
			
		##	Determine sign from ordering of raising/lowering operators (when string operators coincide with AOp).
		##	Order that operators are placed is Op4, and then Op3, Op2, and finally Op1.
			if self.p_type == 'F':
				if Op1 == 'a': Vstrength = -Vstrength	# StrOp from Op2,Op3,Op4
				if Op3 == 'a': Vstrength = -Vstrength	# StrOp from Op4
			
		##	Make the nodes
			if r == 0: print "    Skipping case r = 0 in %s" % V
			
			if m1 > 0:
				##	String coming out of Op1
				if Op1 == 'a' or Op1 == 'A':
					addNodes((Op1+'_',ly1), ly1, m1, 'R', (Op1+'Op',1.), 'StrOp', keep)
				else:
					raise NotImplementedError, "Don't know what to do with Op1 '%s' in %s" % (Op1,V)
			
			if m1 > 0 and r > 0:
				##	String coming out of Op2
				Op12 = Op1 + Op2
				if Op12 == 'aA' or Op12 == 'Aa' or Op12 == 'aa' or Op12 == 'AA':
					addNodes((Op12+'_',ly1,m1), ly2, r, (Op1+'_',ly1,m1), (Op2+'Op',1.), 'Id', keep)
					pre_Op3_node = (Op12+'_', ly1, m1, r)	# last node before Op3
				else:
					raise NotImplementedError, "Don't know what to do with Op12 '%s' in %s" % (Op12,V)
			
			if m1 == 0 and r > 0:
				##	String coming out of Op12 (because Op1 and Op2 are on the same site)
				Op12 = Op1 + Op2
				if Op12 == 'aa' or Op12 == 'AA': raise NotImplementedError, "Don't know what to do with Op12 '%s' in %s" % (Op12,V)
				if Op12 == 'aA' or Op12 == 'Aa':		# sign has already been switched earlier, also normal-ordering means not 1-n.
					addNodes(('n_',ly1), ly1, r, 'R', ('nOp',1.), 'Id', keep)
					pre_Op3_node = ('n_', ly1, r)	# last node before Op3
			
			if m2 > 0:
				##	The final string leading to 'F' (backwards)
				if Op4 == 'a':
					addNodesB(('_a',ly4), ly4, m2, 'F', ('aOp',1.), 'StrOp', keep)
				elif Op4 == 'A':
					addNodesB(('_A',ly4), ly4, m2, 'F', ('AOp',1.), 'StrOp', keep)
				else:
					raise NotImplementedError, "Don't know what to do with Op4 '%s' in %s" % (Op4,V)
				post_Op3_node = ('_'+Op4, ly4, m2-1)	# the node right after Op3
			
			if m2 > 0 and r > 0:
				##	The third operator to connect the forward / backward strings
				XNode = MPOgraph1[ly3][pre_Op3_node].setdefault(post_Op3_node, [])
				if Op3 == 'A' or Op3 == 'a':
					XNode.append((Op3+'Op', Vstrength))
				else:
					raise NotImplementedError, "Don't know what to do with Op3 '%s' in %s" % (Op3,V)
			
			if r > 0 and m2 == 0:
				##	Op34 to connect the forward string to 'F'
				Op34 = Op3 + Op4
				if Op34 == 'aa' or Op34 == 'AA': raise NotImplementedError, "Don't know what to do with Op34 '%s' in %s" % (Op34,V)
				if Op34 == 'aA' or Op34 == 'Aa':		# sign has already been switched earlier
					XNode = MPOgraph1[ly3][pre_Op3_node].setdefault('F', [])
					XNode.append(('nOp', Vstrength))
		
			#break

	##	Only keep marked items
		if not force_keep:
		##	Delete upmarked items
			for s,G in enumerate(MPOgraph1):
				NumNodes = len(G)
				for k,v in G.items():
					if '&&keep' in v:
						del v['&&keep']
					else:
						del G[k]
				if verbose >= 1: print "\tOnly keeping %s items out of %s in MPOgraph1[%s]" % (len(G), NumNodes, s)
		##	Remove references to dead items
			for s in range(len(MPOgraph1)):
				G = MPOgraph1[s]
				nextG = MPOgraph1[(s + 1) % N]
				for src,dest_dict in G.iteritems():
					for dest in dest_dict.keys():
						if dest not in nextG: del dest_dict[dest]
		
		self.MPOgraph1 = MPOgraph1



	##################################################	
	def make_MPOgraph1_from_Vmk_exp(self, verbose=0):
		"""	Given Vmk_list, make the MPOgraph1.
				Use the (1-in many-out) exp approximation for each (mu,nu,m).
				"""
		if self.p_type != 'F': raise NotImplementedError
		if verbose >= 1: print "Making MPOgraph1 (with exponential approximation) from Vmk..."
		N = self.Nlayer
		L = self.L
		self.make_split_Vmk_list(verbose, False)
	##	The elements of MPOgraph1[s] labels the states to the *left* of site s, so Op's in MPOgraph1[s] always sit in site s.
		if hasattr(self, 'MPOgraph'): raise RuntimeError
		try:
			MPOgraph1 = self.MPOgraph1
		except AttributeError:
			MPOgraph1 = self.empty_MPOgraph(N)
		self.MPOgraph = MPOgraph1		# Hack to get addNodes and addNodesB to work
		
	##	1) Turn splitV_list in to dictionary
	##		{ (Op0,ly0,m1,Op1,ly1): { (Op2,ly2,m2,Op3,ly3): { 'Vr':Vr_list }, ... }, ... }
	##		Vr_list is a list of Vstrength's at each r (starting with r=0)
	##		This allows us to group Vmr's based on relevant data.
		splitV_dict = {}
		for V in self.splitV_list:
			(Op0,ly0),m1,(Op1,ly1),r,(Op2,ly2),m2,(Op3,ly3),Vstrength,must_keep = V
			Op0 = Op0[0]; Op1 = Op1[0]; Op2 = Op2[0]; Op3 = Op3[0];
			if np.abs(Vstrength) == 0: continue
			if self.p_type == 'F':
				if Op0 == 'a': Vstrength = -Vstrength	# StrOp from Op1,Op2,Op3
				if Op2 == 'a': Vstrength = -Vstrength	# StrOp from Op3
			D01 = splitV_dict.setdefault((Op0,ly0,m1,Op1,ly1), {})
			D0123 = D01.setdefault( (Op2,ly2,m2,Op3,ly3), {'keep': False, 'Vr': np.zeros(1,self.dtype)} )
			Vr_list = D0123['Vr']
			if r == 0:
				##	One day, we'll solve global warming, one day.  Then we'll address the r=0 case.
				print "!   Skipping case r = 0 in %s" % V
			else:
				if len(Vr_list) <= r:
					Vr_list = np.hstack([Vr_list, np.zeros(r+1-len(Vr_list),self.dtype)])
					D0123['Vr'] = Vr_list
				Vr_list[r] += Vstrength
				if must_keep: D0123['keep'] = True
		Opkey_str = lambda (OO1,ll1,mm,OO2,ll2): OO1+str(ll1)+'_'+str(mm)+'_'+OO2+str(ll2)
		Opkey_Herm = lambda (OO1,ll1,mm,OO2,ll2): ('a' if OO1=='A' else 'A', ll1, mm, 'a' if OO2=='A' else 'A', ll2)
		#print('---'*100)
		#print(Opkey_str(('a','b','c','d','e')))
		#print('---'*100)
	#TODO: pair them up by Herm and use the ignore_herm_conj flag
		
	##	2) Compute the total norms
		total_norms = []
		dropped_norms = []
		for Op01key,Op01Dict in splitV_dict.iteritems():
			for Op23key,Op23Dict in Op01Dict.iteritems():
				if not np.any([ v['keep'] for k,v in Op01Dict.iteritems() ]):
					dropped_norms.append(np.linalg.norm( Op23Dict['Vr'] ))
				total_norms.append(np.linalg.norm( Op23Dict['Vr'] ))
		total_norm = np.linalg.norm(total_norms)
		if verbose >= 1: print '\tTotal norm:', total_norm

	##	3) Do the exponential approximation and store the result
		for Op01key,Op01Dict in splitV_dict.iteritems():
			Op0,ly0,m1,Op1,ly1 = Op01key
			Op23keys = Op01Dict.keys()	# define order
			print(Op23keys)
			if not np.any([ v['keep'] for k,v in Op01Dict.iteritems() ]): continue		# nothing worth keeping, move along
		##	First gather the sequences of the numbers
			seqs = []
			#print('---'*30)
			#print(Op23keys)
			#print('---'*30)
			#quit()
			for Op23key in Op23keys:
				Op2,ly2,m2,Op3,ly3 = Op23key
				Op23Dict = Op01Dict[Op23key]
				seqs.append( Op23Dict['Vr'][(ly2-ly1-1)%N+1 : : N] )
				#print '\t ', Op23key, Op23Dict['keep'], seqs[-1]
		##	The approximation
			#print('---'*30)
			#print(Op23Dict)
			#print('---'*30)
			#quit()
			Op01Dict['keys'] = Op23keys	# fix the order of the keys (to match with 'Markov')
			Op01Dict['Markov'] = Markov.seq_approx(seqs, target_err = self.V_eps * total_norm, target_frac_err = 1., max_nodes = None)
		
		
	##	4) Now build the MPOgraph, scan every (Op0,ly0,m1,Op1,ly1) quintuple.
		for Op01key,Op01Dict in splitV_dict.iteritems():
			Op0,ly0,m1,Op1,ly1 = Op01key
			if 'keys' not in Op01Dict:
				if verbose >= 4:
					print '\t%s\t#nodes = 0/__,      (dropped),       ' % ( Op01key, ),
					print 'targets = ' + ', '.join([ '%s (%.2e)' % (Opkey_str(Ok), np.linalg.norm(OD['Vr'])) for Ok,OD in Op01Dict.iteritems() if isinstance(Ok, tuple) ])
				continue		# nothing worth keeping, move along
			Op23keys = Op01Dict['keys']
			#print('Op23keys',Op23keys)
			#print('Op01dict',Op01Dict[('A', 0, 0, 'a', 0)])
			#print('-'*100)
			#print(Op01Dict)
			#quit()
			M,entry,exits,MInfo = Op01Dict['Markov'].MeeInfo()
			if verbose >= 3:
				print '\t%s\t#nodes = %s/%s, norm err = %.7f, ' % ( Op01key, len(M), MInfo['l'],  MInfo['norm_err'] ),
				print 'targets = ' + ', '.join([ '%s (%.2e)' % (Opkey_str(Ok), np.linalg.norm(OD['Vr'])) for Ok,OD in Op01Dict.iteritems() if isinstance(Ok, tuple) ])
		
		##	Prepare the nodes leading up to the loop
			if m1 > 0:		# String coming out of Op0
				if Op0 == 'a' or Op0 == 'A':
					pre_Op1_ly, pre_Op1_node = self.addNodes((Op0+'_',ly0), ly0, m1, 'R', (Op0+'Op',1.), 'StrOp')
				else:
					raise NotImplementedError, "Don't know what to do with Op0 '%s' in %s" % (Op0,Op01key)
				entry_Op = Op1 + 'Op'
			else:
				Op01 = Op0 + Op1
				pre_Op1_node = 'R'
				if Op01 == 'aA' or Op01 == 'Aa':
					entry_Op = 'nOp'
				else:
					raise NotImplementedError, "Don't know what to do with Op01 '%s' in %s" % (Op01,Op01key)
			
		##	Make nodes in the loop
		##	M is shaped (L,L), where L is the number of nodes (exponentials) required.
		##	entry is shaped (L,), exits is shaped (# Op23keys, L)
			num_nodes = len(M)
			for i in range(num_nodes):
			##	Make nodes for each layers
				for s in range(N - 1):
					MPOgraph1[(ly1 + s + 1) % N][Op01key + (i,)] = { Op01key+(i,): [('Id',1.)] }
				mNode = MPOgraph1[ly1].setdefault(Op01key + (i,), {})
				for j in range(num_nodes):
					if M[j, i] != 0.:
						mNode[Op01key + (j,)] = [('Id', M[j, i])]
			##	Make entries
				if entry[i] != 0.:
					MPOgraph1[ly1][pre_Op1_node][Op01key + (i,)] = [(entry_Op, entry[i])]

		##	Prepare the nodes after exiting the loop and connect them
			#print('-'*100)
			#print(Op23keys)
			#quit()
			for j,Op23key in enumerate(Op23keys):
				#print('=======================================')
				#print(Opkey)
				#print('=======================================')
				Op2,ly2,m2,Op3,ly3 = Op23key
				if m2 > 0:		# The final string leading to 'F' (backwards)
					if Op3 == 'a' or Op3 == 'A':
						post_Op2_ly, post_Op2_node = self.addNodesB(('_' + Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), 'StrOp')
					else:
						raise NotImplementedError, "Don't know what to do with Op3 '%s' in %s" % (Op3,Op23key)
					exit_Op = Op2 + 'Op'
				else:
					Op23 = Op2 + Op3
					if Op23 == 'aA' or Op23 == 'Aa':
						exit_Op = 'nOp'
					else:
						raise NotImplementedError, "Don't know what to do with Op23 '%s' in %s" % (Op23,Op23key)
					post_Op2_node = 'F'
			##	Make exits
				for i in range(num_nodes):
					if exits[j, i] != 0.:
						MPOgraph1[ly2][Op01key + (i,)][post_Op2_node] = [(exit_Op, exits[j,i])]

		##	The loop ends

	##	5) Wrap-up
		self.MPOgraph1 = MPOgraph1
		del self.MPOgraph		# Undo the hack earlier

		if verbose >= 1:
			print '\ttarget fractional error:', self.V_eps
			norm_errs = np.array([ OD['Markov'].info['norm_err'] for Ok,OD in splitV_dict.iteritems() if 'Markov' in OD ])
			print '\tMarkov approx error, dropped / total norm: %s, %s / %s' % (np.linalg.norm(norm_errs), np.linalg.norm(dropped_norms), total_norm)
			str_nodes = np.zeros(N, dtype=int)
			M_nodes = np.zeros(N, dtype=int)
			for s in range(N):
				for k in MPOgraph1[s]:
					if '_' in k[0]: str_nodes[s] += 1
					elif len(k) == 6: M_nodes[s] += 1
			print '\t# of string-nodes:', str_nodes
			print '\t# of Markov-nodes:', M_nodes
		#self.print_MPOgraph(MPOgraph1)



	##################################################	
	def make_MPOgraph1_from_Vmk_slycot(self, verbose=0):
		"""	Given Vmk_list, make the MPOgraph1.
				Use the slycot (q-in p-out) exp approximation.
				"""
		#if self.p_type != 'F': raise NotImplementedError
		t0 = time.time()
		if verbose >= 1: print "Making MPOgraph1 (with slycot) from Vmk..."
		N = self.Nlayer
		L = self.L
		self.make_split_Vmk_list(verbose, False)
		#MAKE SPLIT VMK LIST WORKS

	##	The elements of MPOgraph1[s] labels the states to the *left* of site s, so Op's in MPOgraph1[s] always sit in site s.
		if hasattr(self, 'MPOgraph'): raise RuntimeError
		try:
			MPOgraph1 = self.MPOgraph1
		except AttributeError:
			MPOgraph1 = self.empty_MPOgraph(N)
		self.MPOgraph = MPOgraph1		# Hack to get addNodes and addNodesB to work
		
	##	1) Turn split_Vlist in to MPO_system
	##		Group all the terms to the form (input, output, r),
	##		where input = (Op0,ly0,m1,Op1,ly1), output = (Op2,ly2,m2,Op3,ly3)
	## elements of split_Vlist are either (operator, layer) tuples or spacings between the layers
	## m1 is spacing between Op0 and Op1, r is betwen Op1 and Op2, m2 is between Op2 and Op3
	## the positions are the positions of the actual tensors, not the orbitals
	## therefore orbitals at the same physical site but in different layers will have different positions
		MS = Markov.MPO_system(dtype = self.dtype)
		r0_list = {}		# store all the terms with r = 0 separately
		r0m0_list = {}
		r0_onem0_list={}
		for V in self.splitV_list:
			(Op0,ly0),m1,(Op1,ly1),r,(Op2,ly2),m2,(Op3,ly3),Vstrength,must_keep = V
			Op0 = Op0[0]; Op1 = Op1[0]; Op2 = Op2[0]; Op3 = Op3[0];
			if np.abs(Vstrength) == 0: continue
			if self.p_type == 'F':
				if Op0 == 'a': Vstrength = -Vstrength	# StrOp from Op1,Op2,Op3
				if Op2 == 'a': Vstrength = -Vstrength	# StrOp from Op3
			if r == 0:
				##	These are cases where Op1 and Op2 overlap on the same site; these are handled separately from the MarkovSys
				Op12 = Op1 + Op2
				if m1 > 0 and m2 > 0:
					if Op12 == 'Aa':
						key12 = 'n'
					elif Op12 == 'AA' or Op12 == 'aa':
						key12 = Op12
					else:
						raise ValueError, "unidentfied operator %s" % Op12
					r0_list.setdefault( (Op0,ly0,m1,key12,ly2,m2,Op3,ly3), 0. )
					r0_list[(Op0,ly0,m1,key12,ly2,m2,Op3,ly3)] += Vstrength
				elif m1 == 0 and m2 == 0:
					# Assumes that the operators are AAaa in some order.
					assert self.p_type == 'B' and ly0==ly1 and ly1==ly2 and ly2==ly3 and Op0=='A' and Op1=='A'
					r0m0_list.setdefault(('AAaa',ly0), 0.)
					r0m0_list[('AAaa',ly0)] += Vstrength
				else:
					raise RuntimeError ("Fuck.", V, Vstrength)
			else:
				if (r - ly2 + ly1) % N: raise RuntimeError
				MS.add_io_single_r( (Op0,ly0,m1,Op1,ly1), (Op2,ly2,m2,Op3,ly3), (r-ly2+ly1) // N, Vstrength )
		if verbose >= 1:
			print '\tProcessed %s Vmk terms.' % len(self.splitV_list)
			if self.ignore_herm_conj: print '\tAvoiding Hermitian conjugate terms.'
		Opkey_str = lambda (OO1,ll1,mm,OO2,ll2): OO1+str(ll1)+'_'+str(mm)+'_'+OO2+str(ll2)
		Opkey_Herm = lambda (OO1,ll1,mm,OO2,ll2): ('a' if OO1=='A' else 'A', ll1, mm, 'a' if OO2=='A' else 'A', ll2)
		MS.check_sanity()
		if verbose >= 4: print "\tr0m0:", r0m0_list
		
	##	Add in the necessary information for exponential approximation
	##	2a) Make the ignore_r0 table
		for io in MS.io:
			seq = io['seq']
			ignore_r0 = np.zeros(seq.shape[0:2], int)
			for a,o in enumerate(io['out']):
				for b,i in enumerate(io['in']):
					if i[4] >= o[1]:		# compare ly1 to ly2
						ignore_r0[a, b] = 1
						if seq[a, b, 0] != 0: raise RuntimeError, (a, b, seq[a, b, 0])
			#io['ignore_r0'] = ignore_r0
			#print joinstr([ ignore_r0, seq[:, :, 0] ]), '\n'
	##	2b) Identify the complex conjugates
		HermConj = { 'a':'A', 'A':'a' }
		for ikey in MS.Din:
			Op0,ly0,m1,Op1,ly1 = ikey
			if m1 > 0:
				MS.Din[ikey]['conj'] = (HermConj[Op0],ly0,m1,HermConj[Op1],ly1)
			else:
				MS.Din[ikey]['conj'] = (HermConj[Op1],ly1,m1,HermConj[Op0],ly0)	# assume that Op01 is one of Aa, aa, or AA
		for okey in MS.Dout:
			Op2,ly2,m2,Op3,ly3 = okey
			if m2 > 0:
				MS.Dout[okey]['conj'] = (HermConj[Op2],ly2,m2,HermConj[Op3],ly3)
			else:
				MS.Dout[okey]['conj'] = (HermConj[Op3],ly3,m2,HermConj[Op2],ly2)	# assume that Op23 is one of Aa, aa, or AA
		MS.pair_conj_blocks()
		
	##	3) Make the exponential approximations
	##	Each ioblock will then have keys 'A', 'B', 'C' and 'N', 'norm_err'
		MS.compute_norm()		# create key '2norm'
		target_err = self.V_eps * MS.info['2norm']
		MS.compute_ABC(tol = target_err, verbose = max(verbose-3,0), verbose_prepend = '\t\t')

	##	4-8) Let someone else finish the job
		self.make_MPOgraph1_from_MS(MS, r0_list, t0=t0, r0m0_list=r0m0_list, verbose=verbose)
		

	def make_MPOgraph1_from_MS(self, MS, r0_list, t0=None, r0m0_list={}, verbose=0):
		"""Given a MS (Markov system) and a r0_list, make the MPO graph.
		This function is a continuation of make_MPOgraph1_from_Vmk_slycot().
			"""
		if t0 is None: t0 = time.time()
		N = self.Nlayer
		MPOgraph1 = self.MPOgraph		# Hack to get addNodes and addNodesB to work
		Opkey_str = lambda (OO1,ll1,mm,OO2,ll2): OO1+str(ll1)+'_'+str(mm)+'_'+OO2+str(ll2)

	##	4) Add MPO_graph information to MPO_system, and also make the string nodes.
	##	Sources
		create_nodes_in_list = []
		for Op01key,D in MS.Din.iteritems():
			if MS.io[D['index'][0]]['N'] == 0: continue		# ignore string belonging in blocks with 0 nodes
			(Op0,ly0,m1,Op1,ly1) = Op01key
			if m1 > 0:		# String coming out of 'R' with operator Op0
				if Op0 == 'a' or Op0 == 'A':
					pre_Op1_ly, D['src'] = self.addNodes((Op0+'_',ly0), ly0, m1, 'R', (Op0+'Op',1.), 'StrOp')
				else:
					raise NotImplementedError, "Don't know what to do with Op0 '%s' in %s" % (Op0,Op01key)
				D['Op'] = Op1 + 'Op'
			else:
				Op01 = Op0 + Op1
				if Op01 != 'Aa': raise NotImplementedError, "Don't know what to do with Op01 '%s' in %s" % (Op01,Op01key)
				D['src'] = 'R'
				D['Op'] = 'nOp'
			D['orb'] = ly1
			if self.ignore_herm_conj and D['Op'] != 'AOp': create_nodes_in_list.append(Op01key)
	##	Targets
		for Op23key,D in MS.Dout.iteritems():
			if MS.io[D['index'][0]]['N'] == 0: continue		# ignore string belonging in blocks with 0 nodes
			Op2,ly2,m2,Op3,ly3 = Op23key
			if m2 > 0:		# The final string leading to 'F' (backwards) with Op3
				if Op3 == 'a' or Op3 == 'A':
					post_Op2_ly, D['dest'] = self.addNodesB(('_'+Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), 'StrOp')
				else:
					raise NotImplementedError, "Don't know what to do with Op3 '%s' in %s" % (Op3,Op23key)
				D['Op'] = Op2 + 'Op'
			else:
				Op23 = Op2 + Op3
				if Op23 != 'Aa': raise NotImplementedError, "Don't know what to do with Op23 '%s' in %s" % (Op23,Op23key)
				D['Op'] = 'nOp'
				D['dest'] = 'F'
			D['orb'] = ly2
	##	Prepare the io blocks, and give them names (unique identifiers for the MPOgraph)
		for bid,io in enumerate(MS.io):
			io['StrOp'] = 'Id'		# operator between loop nodes
			Op01key = io['in'][0]		# a sample input key, used to generate a name for the ioblock
			io['name'] = Op01key[0] + str(self.layers[Op01key[1]][0]) + '-' + str((Op01key[2] - Op01key[4] + Op01key[1]) // N) + '-' + Op01key[3] + str(self.layers[Op01key[4]][0]) + '.' + str(bid)
	
	##	5)	Create the nodes for the r = 0 cases
		for r0Op,V in r0_list.iteritems():
			Op0,ly0,m1,Op12,ly12,m2,Op3,ly3 = r0Op		# guarentee that m1, m2 > 0
			if self.ignore_herm_conj:
				if Op0 == 'A': continue
				V = 2. * V
			pre_Op1_ly, pre_Op1_node = self.addNodes((Op0+'_', ly0), ly0, m1, 'R', (Op0+'Op',1.), 'StrOp')
			post_Op2_ly, post_Op2_node = self.addNodesB(('_'+Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), 'StrOp')
			if (post_Op2_ly - pre_Op1_ly - 1) % N: raise RuntimeError, (r0Op, pre_Op1_ly, post_Op2_ly, N)
			XNode = MPOgraph1[pre_Op1_ly][pre_Op1_node].setdefault(post_Op2_node, [])
			XNode.append((Op12+'Op',V))
			if verbose >= 5: print "\t\tr = 0: (%s, %s, %s) -> (_%s, %s, %s-1), %sOp, ly=%s, V=%s" % (Op0+'_',ly0,m1,Op3,ly3,m2,Op12,ly12,V)
		for r0Op,V in r0m0_list.iteritems():
			if r0Op[0] == 'AAaa':
				XNode = MPOgraph1[r0Op[1]]['R'].setdefault('F', [])
				XNode.append(('AAaaOp',V))
				if verbose >= 4: print "\tAdding operator 'AAaa' R->F with V = {}".format(V)
			else:
				raise ValueError, "don't know what to do with operators %s" % r0Op[0]
		
	##	6) ........
		if self.ignore_herm_conj:
			io_create_nodes = np.zeros(len(MS.io), dtype=int)
			for inKey in create_nodes_in_list:
				io_create_nodes[MS.Din[inKey]['index'][0]] = 1
			MS.create_partial_MPO_nodes(MPOgraph1, np.nonzero(io_create_nodes)[0], double_unconj_ioblocks=True)
		else:
			io_create_nodes = np.ones(len(MS.io), dtype=int)
			MS.create_MPO_nodes(MPOgraph1)
		
	##	7) Print stuff for educational purposes
		if verbose >= 3:	
			for j,block in enumerate(MS.io):
				if not io_create_nodes[j]: continue
				#if block['conj'][0] < j: continue		# Only print one in a Hermitian conjugates pair
				if len(block['A']) > 0 or verbose >= 4:
					print '\t\t%2d (%2d*): %2d nodes / %s, norm err: %.3e / %.5e,' % ( j, block['conj'][0], block['N'], block['num_above_tol'], block['norm_err'], block['2norm'] ),
					print ' [%s]  %s  ->  %s' % ( block['name'], ', '.join(map(Opkey_str, block['in'])), ', '.join(map(Opkey_str, block['out'])) )
					#graph_err = np.array([ MS.calc_MPOgraph_err(MPOgraph1, i) for i in block['in'] ])
					#if len(block['A']) > 0: print joinstr(['\t\t\t', np.linalg.norm(graph_err.reshape(-1)), graph_err])
					#print '\t\t\t', block['conj']
		if verbose >= 1:
			print '\ttarget fractional error: %s  (tol = %s)' % (self.V_eps, MS.info['ABC_tol'])
			print '\tnorm err / total norm: %s / %s' % (MS.info['total_norm_err'], np.linalg.norm([MS.info['2norm']] + [V for OOOO,V in r0_list.iteritems()]))
			str_nodes = np.zeros(N, dtype=int)
			M_nodes = np.zeros(N, dtype=int)
			for s in range(N):
				for k in MPOgraph1[s]:
					if isinstance(k, tuple):
						if '_' in k[0]: str_nodes[s] += 1
						elif k[0] == 'Mk': M_nodes[s] += 1
			print '\t# of string-nodes:', str_nodes
			print '\t# of Markov-nodes:', M_nodes
			print '\tConstructing MPOgraph took {} seconds'.format(time.time() - t0)
		
	##	8) Wrap-up
		self.MPOgraph1 = MPOgraph1
		del self.MPOgraph		# Undo the hack earlier
		self.Vmk_MS = {'version':1.2, 'Nlayer':self.Nlayer, 'MS':MS, 'r0_list':r0_list, 'r0m0_list':r0m0_list,
			'Lx':self.Lx, 'Vq0':self.Vq0, 'p_type':self.p_type, 'species':self.species, 'bands':self.bands}
		#self.print_MPOgraph(MPOgraph1)



	def save_Markov_system_to_file(self, filename):
		"""Once a slycot exp. approximation has been used, this function will save its work to disk.
		Use load_Markov_system_from_file() to recover the MPOgraph.
			"""
		with open(filename, 'wb') as f:
			print ("Saving Markov system to file '{}'...".format(filename))
			cPickle.dump(self.Vmk_MS, f, protocol=-1)

	def load_Markov_system_from_file(self, filename, verbose=0):
		"""Load a Markov system from file.
		This function is paired with save_Markov_system_to_file(), and to be used in place of make_MPOgraph1_from_Vmk_slycot()
			"""
		t0 = time.time()
		with open(filename, 'rb') as f:
			if verbose >= 1: print ("\tLoading Markov system from file '{}'...".format(filename))
			Vmk_MS = cPickle.load(f)
		if verbose >=1: print "\tkeys: ", Vmk_MS.keys()	#TODO, check compatibility of species and bands
		if 'version' not in Vmk_MS: raise RuntimeError
		if Vmk_MS['version'] < 1.: raise RuntimeError
		if Vmk_MS['Nlayer'] != self.Nlayer: raise RuntimeError, "Error!  Nlayer mismatch: {} != {}".format(Vmk_MS['Nlayer'], self.Nlayer)
		if Vmk_MS['Lx'] != self.Lx: print "Warning!  Lx mismatch {} != {}".format(Vmk_MS['Lx'], self.Lx)
		if Vmk_MS['p_type'] != self.p_type: print "Warning!  p_type mismatch {} != {}".format(Vmk_MS['p_type'], self.p_type)
		if 'Vq0' in Vmk_MS: self.Vq0 = Vmk_MS['Vq0']
		
		if hasattr(self, 'MPOgraph'): raise RuntimeError
		try:
			self.MPOgraph = self.MPOgraph1
		except AttributeError:
			self.MPOgraph = self.empty_MPOgraph(self.Nlayer)
		self.make_MPOgraph1_from_MS(Vmk_MS['MS'], Vmk_MS['r0_list'], r0m0_list=Vmk_MS.get('r0m0_list',{}), t0=t0, verbose=verbose)
		

	##################################################	
	def make_empty_MPOgraph1(self, verbose=0):
		"""	Make an empty MPOgraph1 if it doesn't exist already.  """
		if verbose >= 1: print "Making empty MPOgraph1."
		if hasattr(self, 'MPOgraph'): raise RuntimeError
		try:
			MPOgraph1 = self.MPOgraph1
		except AttributeError:
			self.MPOgraph1 = self.empty_MPOgraph(self.Nlayer)


	def MPOgraph1_to_MPOgraph(self, verbose=0):
		"""	Make MPOgraph from MPOgraph1.
				MPOgraph1 (should) have length Nlayers, whereas MPOgraph would have length L.
			"""
		MPOgraph1 = self.MPOgraph1
		if not isinstance(MPOgraph1, list): raise ValueError, type(MPOgraph1)
		N = len(MPOgraph1)
		self.MPOgraph = [ deepcopy(MPOgraph1[s%N]) for s in range(self.L) ]

	def add_chemical_pot(self, mu):
		"""	Given mu, add a chemical potential for each LL. """
		try:
			iter(mu)
		except TypeError:
			mu = [mu]
		#print('chemical potential',mu)
		#quit()
		for s,G in enumerate(self.MPOgraph):
			XNode = G['R'].setdefault('F', [])
			XNode.append(( 'nOp', -mu[s%len(mu)] ))

	def add_three_body(self, V):
		"""	puts in a simple-to-implement but non-local term which breaks ph-symmetry-SG"""
		try:
			iter(V)
		except TypeError:
			V = [V]
		for s,G in enumerate(self.MPOgraph):
			Node1 = G['R'].setdefault('N1', [])
			Node1.append(( 'nOp_shift', -V[s%len(V)] ))
			Node2 = G.setdefault('N1',{}).setdefault('N2', [])
			Node2.append(( 'nOp_shift', 1. ))
			Node3 = G.setdefault('N2',{}).setdefault('F', [])
			Node3.append(( 'nOp_shift', 1. ))

	################################################################################
	################################################################################
	##	Convention for Trm_list:
	##		There are five columns, [ r, m, mu, nu, T_{r,m} ] which represents the Hamiltonian:
	##
	##		H = \sum_{r,m} T_{r,m} c^\dag_{r,\mu} c_{r-m,\nu}
	##			* Note that r, c may be positive or negative.
	##			* Hermitian conjugate terms are explicitly included in Trm_list.
	def convert_Trm_tolist(self):
		"""	Convert array-style Trm to (num x 5) list of non-zero Ts.
			Takes in self.Trm and writes to self.Trm_list.
			While Trm doesn't store negative m-values, Trm_list *does* explicitly store -m's.
			"""
		Trm = self.Trm
		N = self.Nlayer
		Trm_list = []
		
		species = self.species
		for mu in Trm.keys():
			for nu in Trm.keys():
				trm = Trm[mu][nu]
				if trm  is not None:
				
					n_mu = len(species[mu])
					n_nu = len(species[nu])
					for a, b, r, m in itertools.product( xrange(n_mu), xrange(n_mu), xrange(trm.shape[2]), xrange(trm.shape[3])  ):
					
						if m==0:
							Trm_list.append([ r, m, species[mu][a], species[nu][b], trm[a, b, r, m] ])
						else:
							# T[r][m] psi^D_r psi_{r - m} + conj(T[r+m][m]) psi^D_{r} psi_{r + m}  #
							if np.abs(trm[a, b, r, m]) > 1e-10:
								Trm_list.append([ r - m, -m, species[nu][b], species[mu][a], np.conj(trm[a, b, r, m]) ])
								Trm_list.append([ r, m, species[mu][a], species[nu][b], trm[a, b, r, m] ])
					
		self.Trm_list = Trm_list

	def add_Trm(self, verbose=0):
		"""	Given self.Trm (one-body terms), add the terms to MPOgraph. """
		L = self.L
		N = self.Nlayer
		try:
			MPOgraph = self.MPOgraph
		except AttributeError:
			MPOgraph = self.MPOgraph = self.empty_MPOgraph()
		for V in self.Trm_list:
			r,m,ly0,ly1,Tstrength = V
			loc0 = ly0 + N * r		# 'A'
			loc1 = ly1 + N * (r - m)		# 'a'
			locMid = (loc0 + loc1) // 2	# locMid is (maybe) closer to the smaller of the two [loc0, loc1]
			if self.bc == 'finite':
				if loc0 < 0 or loc0 >= L or loc1 < 0 or loc1 >= L: continue
			if loc0 == loc1:		# apply nOp, no strings needed.
				XNode = MPOgraph[loc0 % L]['R'].setdefault('F', [])
				XNode.append(('nOp', Tstrength))
				if verbose >= 3: print '  ', V, '->', [(loc0%N, 'n')]

			elif loc0 > loc1:		# aOp to the left of AOp
				if self.ignore_herm_conj: Tstrength = 2. * Tstrength		# double
				##	Add a string 'a' from loc1 to locMid, and string '_A' from locMid to loc0.
				pre_Op2_ly,pre_Op2_node = self.addNodes(('a_', loc1%N), loc1%L, locMid-loc1, 'R', ('aOp',1.), 'StrOp')
				ly,post_Op2_node = self.addNodesB(('_A', loc0%N), loc0%L, loc0-locMid, 'F', ('AOp',1.), 'StrOp')
				XNode = MPOgraph[pre_Op2_ly][pre_Op2_node].setdefault(post_Op2_node, [])
				if locMid > loc1: XNode.append(('StrOp', Tstrength))
				if locMid == loc1: XNode.append(('aOp', Tstrength))
				if verbose >= 3: print '  ', V, '->', [('a', loc1%N), loc0-loc1, ('A', loc0%N)]

			elif loc0 < loc1:		# AOp to the left of aOp
				if self.ignore_herm_conj: continue		# ignore this piece
				##	Add a string 'A' from loc0 to locMid, and string '_a' from locMid to loc1.
				pre_Op2_ly,pre_Op2_node = self.addNodes(('A_', loc0%N), loc0%L, locMid-loc0, 'R', ('AOp',1.), 'StrOp')
				ly,post_Op2_node = self.addNodesB(('_a', loc1%N), loc1%L, loc1-locMid, 'F', ('aOp',1.), 'StrOp')
				XNode = MPOgraph[pre_Op2_ly][pre_Op2_node].setdefault(post_Op2_node, [])
				if locMid > loc0: XNode.append(('StrOp', Tstrength))
				if locMid == loc0: XNode.append(('AOp', Tstrength))
				if verbose >= 3: print '  ', V, '->', [('A', loc0%N), loc1-loc0, ('a', loc1%N)]




################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

##	Convenience functions for manipulating root configurations.
def interlace(*args):
	""" Given (variable #) of args l1, l2, . . . 	
		interleaves them to form  
			[ l1[0], l2[0], ..., l1[1], l2[1], ..., l1[max_len], l2[max_len] ... ]
	
		where max_len is the longest length li. Any len(li) < max_len are periodized.
	If any li is a number (rather than list), it is treated as [li].
	
	Example:
	>>>	interlace(0, [1,2,3,4], [8,9])
		array([0, 1, 8, 0, 2, 9, 0, 3, 8, 0, 4, 9])
	"""
	Nlayer = len(args)
	ls = []
	for i in range(Nlayer):
		ls.append(toiterable(args[i]))
	max_len = max( len(l) for l in ls)
	
	stacked = np.zeros((max_len, len(ls) ), dtype=np.int)
	for i in range(Nlayer):
		stacked[:, i] = np.tile(ls[i], max_len/len(ls[i]))

	stacked = np.reshape(stacked, (-1,))
	return stacked
	#return list(stacked)

def interlace_tile(a, N):
	"""	Interlace a sequence (a) with copies of itself.
			Example:
			>>>	interlace_zero(np.array([0, 1, 0, 2, 3]), 2)
				array([0, 0, 1, 1, 0, 0, 2, 2, 3, 3])
		"""
	if a.ndim != 1: raise ValueError
	if N <= 0: return np.zeros([0], a.dtype)
	return np.tile(a, N).reshape([N,-1]).transpose().reshape([-1])

def interlace_zero(a, N):
	"""	Interlace a sequence (a) with zeros.
			Example:
			>>>	interlace_zero(np.array([0, 1, 0, 2, 3]), 2)
				array([0, 0, 1, 0, 0, 0, 2, 0, 3, 0])
		"""
	if a.ndim != 1: raise ValueError
	if N <= 0: return np.zeros([0], a.dtype)
	t = np.zeros([len(a), N], a.dtype)
	t[:, 0] = a
	return t.reshape(-1)



####################################################################################################
####################################################################################################
class bands(object):
	""" Encapsulates information about a set of bands. Meant to be subclassed.
	
			__init__( [b0, b1, b2])
		
		Initialize a set of bands of type b0, b1 . . .
		The properties of the bands are then accessed by the numbers 0, 1, 2. These properties are:
	
	
			phi(a, y) = wavefunction
			eta(a) = parity of bands under mirror reflection
			F(a, b, qx, qy) = form factors
			pre_F(a, b) = optional prefactor for the form factors. (see below)
			anistropy = 1. + ? . Scales qx and qy by 1/ani, 1*ani, before any use in form-factors (for valley nematics).
					
		For example: 
	
			B = bands_subclass(['x', 'y', 'z'])
		
		initializes a set of bands of type x, y, z. To obtain parity of 'x', we call
		
			B.eta(0)
			
		If we had initialized as
			B = bands_subclass(['y', 'x', 'z'])
			
		then B.eta(0) would be parity of 'y'.
		
		Prefactors. Often times F has a prefactor independent of q, 
			
				Full F(a, b, q) = pre_F(a,b)*F(a, b, q)
				
		It can save a lot of time in the numerical integration if f_ab is kept seperate and multiplied at end! So optionally you can set pre_F and it will multiply by this factor.
			
	"""

	def __init__(self, bands, eps = 1.):
		self.bands = bands
		self.anisotropy = 1.
		self.n = len(bands)
		
	def __getitem__(self, i):
		return self.bands[i]
	
	def __eq__(self, other):
		"""Determine if two bands represent same set of bands"""
		if  self.n!=other.n:
			return False
		if  self.__class__.__name__ != other.__class__.__name__:
			return False
			
		for i in xrange(self.n):
			if self[i]!=other[i]:
				return False
				
		return True
	
	def __repr__(self):
		return self.__class__.__name__ + str(self.bands)
	
	def phi(self, a, y):
		raise NotImplementedError	# need to overload
	def eta(self, a):
		raise NotImplementedError	# need to overload
	def P(self, a, b):
		raise NotImplementedError	# need to overload
	def pre_F(self, a, b):
		return 1.
	def F(self, a, b, qx, qy):
		raise NotImplementedError	# need to overload
	def orbital_spin(self, a):
		return 0.



def F_NM(a, b, qx, qy):
		q2 = qy**2 + qx**2
		if a==0 and b==0:
			return  np.exp(-q2/4.)
		elif a > b:
			return   (  np.sqrt(1.*sp.misc.factorial(b, exact=True)/sp.misc.factorial(a, exact=True))*np.sqrt(2.)**(b-a)  )*np.exp(-q2/4.) * (qx-1j*qy)**(a-b) * sp.special.eval_genlaguerre(b, a-b, q2/2.)
		if a < b:
			return   (np.sqrt(1.*sp.misc.factorial(a, exact=True)/sp.misc.factorial(b, exact=True))*np.sqrt(2.)**(a-b))*np.exp(-q2/4.) * (-qx-1j*qy)**(b-a) * sp.special.eval_genlaguerre(a, b-a, q2/2.)
		else:
			return np.exp(-q2/4.) * sp.special.eval_genlaguerre(a, 0, q2/2.)



class LL(bands):
	""" Encapsulates information about LLs	
		Band types are numbers - the LL.
	"""
		
	def eta(self, a):
		#parity of orbital
		return (-1)**self[a]
	
	def P(self, a, b):
		#parity of form factor
		return (-1)**(self[a] + self[b])
	
	def pre_F(self, a, b):
		a = self[a]
		b = self[b]
		if a==b:
			return 1.
		elif a > b:
			return np.sqrt(1.*sp.misc.factorial(b, exact=True)/sp.misc.factorial(a, exact=True))*np.sqrt(2.)**(b-a)
		else:
			return np.sqrt(1.*sp.misc.factorial(a, exact=True)/sp.misc.factorial(b, exact=True))*np.sqrt(2.)**(a-b)

	def F(self, a, b, qx, qy):
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		a = self[a]
		b = self[b]

		#return F_NM(a, b, qx, qy)
		
		
		q2 = qy**2 + qx**2
		if a==0 and b==0:
			return np.exp(-q2/4.)
		if a > b:
			return np.exp(-q2/4.) * (qx-1j*qy)**(a-b) * sp.special.eval_genlaguerre(b, a-b, q2/2.)
		if a < b:
			return np.exp(-q2/4.) * (-qx-1j*qy)**(b-a) * sp.special.eval_genlaguerre(a, b-a, q2/2.)
		else:
			return np.exp(-q2/4.) * sp.special.eval_genlaguerre(a, 0, q2/2.)

	
	def phi(self, a, y):
		a = self[a]
		return np.exp(-0.5*y**2) * sp.special.eval_hermite(a, y) / np.sqrt(np.sqrt(np.pi) * sp.misc.factorial(a, exact=True) * 2**a)

	def orbital_spin(self, a):
		a = self[a]
		return a + 0.5

	@staticmethod
	def latex_F(a, b):
		from fractions import Fraction as Fr
		max_ab = max(a, b)
		min_ab = min(a, b)
		diff_ab = max_ab - min_ab
		sqrt_coef = Fr(sp.misc.factorial(min_ab, exact=True), sp.misc.factorial(max_ab, exact=True) * 2**diff_ab)
		S = ""
		if sqrt_coef.denominator == 1:
			if sqrt_coef.numerator != 1: S += sqrt_coef.numerator
		else:
			S += r"\sqrt{\frac{%s}{%s}} " % (sqrt_coef.numerator, sqrt_coef.denominator)
		LagPoly = sp.special.genlaguerre(min_ab, diff_ab)
		if diff_ab > 0:
			if a > b: S += '(q_x - i q_y)'
			if b > a: S += '(-q_x - i q_y)'
			if diff_ab > 1: S += '^{' + str(diff_ab) + '}'
			S += ' '
		if len(LagPoly) > 0:
			S += '('
			deg = len(LagPoly)
			S += ' + '.join([ "{0:f} q^{1:d}".format(c, 2*deg-2*i) for i,c in enumerate(LagPoly) ][:-1])
			S += ' + ' + str(LagPoly[0]) + ') '
		S += r"\exp{-\frac{q^2}{4}}"
		return S
		

class dLL(bands):
	""" Encapsulates information about LLs of a Dirac cone.
		Does not allow for LL mixing!
		Band types are numbers - the LL.
		This is now deprecated! If you want the behavior of this function call LL_general_combination wth alphas[a]=alphas[a-1]=0.5
	"""
	
	def __init__(self, bands, eps = 1.):
		self.bands = bands
		self.anisotropy = 1.
		self.n = len(bands)
		
		if self.n>1:
			raise NotImplemented
	
	def eta(self, a):
		return 1
		
	def pre_F(self, a, b):
		return 1.

	def F(self, a, b, qx, qy):
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		a = self[a]
		b = self[b]

		q2 = qy**2 + qx**2
		if a==0:
			return np.exp(-q2/4.)
		else:
			return np.exp(-q2/4.) * (sp.special.eval_genlaguerre(a, 0, q2/2.) + sp.special.eval_genlaguerre(a-1, 0, q2/2.))/2.

class general_LL(bands):
	""" Allows you set a linear combination of form factors
		Does not allow for LL mixing!
		Band types are numbers - the LL.
	"""
	
	def __init__(self, u, eps = 1.):
	
		self.anisotropy = eps
		self.bands = self.u = u
		self.n = len(u)
		
		#infer parities of orbitals
		Pa = []
		for ua in u:
			PaA = []
			norm2 = 0.
			for uaA in ua:
				norm2 += np.sum(np.array(uaA.values())**2)
				p = list(set((-1)**np.array(uaA.keys())))
				if len(p)==0:
					PaA.append(np.nan)
				elif len(p)==1:
					PaA.append(p[0])
				else:
					raise ValueError
			assert(np.abs(norm2 - 1.) < 1e-14), "u_{} has norm {}".format(a, norm2)
			Pa.append(np.array(PaA))
	
		Pab = np.ones( (len(u), len(u)) )
		for a in xrange(len(u)):
			for b in xrange(len(u)):
				p = list(set( Pa[a]*Pa[b]))
				if len(p)==1:
					if np.isnan(p[0]):
						Pab[a, b] = 1 #won't be used since 0, so safe to default.
					else:
						Pab[a, b] = p[0]
				
				elif len(p)==2:
					print p[0], np.isnan(p[0])
					if np.isnan(p[0]):
						Pab[a, b] = p[1]
					elif np.isnan(p[1]):
						Pab[a, b] = p[0]
					else: #contains conflicting parities
						raise ValueError
				elif len(p)==3:
					raise ValueError
		#if self.verbose > 2:
		#	print "F_ab parities", Pab
	
		self.Pab = Pab

	def eta(self, a):
		raise NotImplemented
	
	def P(self, a, b):
		return self.Pab[a, b]
	
	def pre_F(self, a, b):
		return 1.

	def F(self, a, b, qx, qy):
		#this function completely ignores a and b and sets the form factors based on the alphas
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		try:
			len(qx)
			F = np.zeros_like( qx )
		except:
			F = 0.
		
		for ua, ub in izip(self.u[a], self.u[b]):
			for k1, v1 in ua.iteritems():
				for k2, v2 in ub.iteritems():
					F+= v1*v2*F_NM(k1, k2, qx, qy)
				
		return F


class LL_general_combination(bands):
	""" Allows you set a linear combination of form factors
		Does not allow for LL mixing!
		Band types are numbers - the LL.
	"""
	
	def __init__(self, alphas):
		self.alphas=alphas
		if(sum(alphas) != 1):
			print "Warning! You should initialize all the alphas to 1!"
		self.bands = alphas
		self.anisotropy = 1.
		self.n = 1
		
	
	def eta(self, a):
		return 1
		
	def pre_F(self, a, b):
		return 1.

	def F(self, a, b, qx, qy):
		#this function completely ignores a and b and sets the form factors based on the alphas
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		a = self[a]
		b = self[b]

		q2 = qy**2 + qx**2
		out=0
		for i,alpha in enumerate(self.alphas): 
			out+=alpha*sp.special.eval_genlaguerre(i,0,q2/2.)
		return out*np.exp(-q2/4.)

####################################################################################################
####################################################################################################
class mQHfunc(object):
	"""	Module for functions pertaining to quantum Hall.
			
			Currently is has:
				rVr_InfSqWell(wellwidth):
					Effective 2D potential for the infinite square well.
				Gaussian_env(cls, r, xi_order=0):
					?
				FermiDirac_tan(cls, L, f, r):
					?
				J0_integral(cls, rV, q, tol=1e-15, d = 1, full_output=False):
					Computes integral of rV(r) against the Bessel function J_0(q r).
		"""
#TODO: Fang-Howard potential
	Madelung_square_torus = -1.950132460001		# The Madelung energy is this/sqrt{2 pi (#orbitals)}


	@classmethod
	def compute_InfSqWell_rVr(cls, step=0.05):
		"""	Helper function to compute 'dat_InfSqWell_rVr' for rVr_InfSqWell(). """
		r_samples = np.exp(np.arange(-9, 6, step))
		y = np.zeros(len(r_samples), dtype=float)
		abserr = y.copy()
		for i in range(len(r_samples)):
			r = r_samples[i]
			##	ifunc(k) = 128 * r * np.pi**3 * sp.special.k0(k*r) * np.sin(k/2)**2 / (k**3 - 4*k*np.pi**2)**2
			ifuncH = lambda k: 32 * r * np.pi**3 * sp.special.k0(k*r) * np.sinc(k/2/np.pi)**2 / (k**2 - 4*np.pi**2)**2
			ifuncM = lambda k: 32 * r * np.pi**3 * sp.special.k0(k*r) * np.sinc(k/2/np.pi - 1)**2 / (k**2 + 2*k*np.pi)**2
			ifuncTa = lambda k: 64 * r * np.pi**3 * sp.special.k0(k*r) / (k**3 - 4*k*np.pi**2)**2
			##	ifunc = ifuncTa [ 1 - cos(k) ]
			cut1 = min(4., 1/r)
			ya1 = sp.integrate.quad(ifuncH, 0, cut1, epsabs=1e-16, full_output=True)
			ya2 = sp.integrate.quad(ifuncH, cut1, 10, epsabs=1e-16, full_output=True)
			yb1 = sp.integrate.quad(ifuncTa, 10, np.inf, epsabs=1e-16, full_output=True)
			yb2 = sp.integrate.quad(ifuncTa, 10, np.inf, weight='cos', wvar=1., epsabs=1e-16, full_output=True)
			y[i] = ya1[0]+ya2[0]+yb1[0]-yb2[0]
			abserr[i] = ya1[1] + ya2[1] + yb1[1] + yb2[1]
			#yW = sp.integrate.quad(ifuncH, 0, np.inf, epsabs=1e-16, full_output=True)
			#print r, y[i], ya1[1], ya2[1], yb1[1], yb2[1], '\t', abserr[i], '\t', yW[0]-y[i], yW[1]
		x = np.hstack([[0], np.sqrt(np.arctan(r_samples)*2/np.pi), [1]])
		y = np.hstack([[0], y, [1]])
		from scipy.interpolate import interp1d
		mQHfunc.dat_InfSqWell_rVr = interp1d(x, y, kind=3)
		return mQHfunc.dat_InfSqWell_rVr

	@classmethod
	def compute_SepInfSqWell_rVr(cls, w, d, step=0.01):
		"""	Helper function to compute 'dat_SepInfSqWell_rVr' for rVr_Separate_InfSqWell(). """
		#ifuncH = lambda k: 32 * r * np.pi**3 * sp.special.k0(k*r) * np.sinc(k/2/np.pi)**2 / (k**2 - 4*np.pi**2)**2
		# Integrate the above weights by cos(k d/w)
		
		#Since I'm not as smart as roger, I'll do this in a stupid way
		def func_z1(z1,z2,r):
			return np.sin(np.pi*z1/w)**2/np.sqrt(r**2+(z1-z2+d)**2)
		def func_z2(z2,r):
			y=sp.integrate.quad(func_z1,0,w,args=(z2,r))
			return np.sin(np.pi*z2/w)**2*y[0]
		def func_r(r):
			A=sp.integrate.quad(func_z2,0,w,args=(r))
			return A[0]*4/w**2*r

		#don't want to compute this for all possible r, so set some cutoff above which we can basically return 1 safely
		max_r=8*max(d,w) 
		r_samples=np.arange(step,max_r,step)
		out=np.zeros(len(r_samples))
		for i,r in enumerate(r_samples): 
			out[i]=func_r(r)
		
		x=np.hstack([[0],r_samples])
		y=np.hstack([[0],out])
		from scipy.interpolate import interp1d
		mQHfunc.dat_SepInfSqWell_rVr=interp1d(x,y,kind=4)
		return mQHfunc.dat_SepInfSqWell_rVr

	@classmethod
	def rVr_InfSqWell(cls, wellwidth):
		"""	Returns rV(r) for the effective 2D interaction when the z-axis is an infinite potential well.
				rV is a 1-parameter vectorized-function.
			"""
		if not hasattr(mQHfunc, 'dat_InfSqWell_rVr'): cls.compute_InfSqWell_rVr()
		wellwidth = float(wellwidth)
		return lambda r: cls.dat_InfSqWell_rVr(np.sqrt(np.arctan(r / wellwidth) * 2 / np.pi))


	@classmethod
	def rVr_Separate_InfSqWell(cls, wellwidth, separation):
		if separation==0:
			print "WARNING: using Seperate_InfSqWell with 0 separation will likely cause vq_from_rVr to crash"
		wellwidth = float(wellwidth)
		separation = float(separation)
		if not hasattr(mQHfunc, 'dat_SepInfSqWell_rVr'): cls.compute_SepInfSqWell_rVr(wellwidth,separation)
		return lambda r: cls.dat_SepInfSqWell_rVr(r) if r<8*max(separation,wellwidth)-0.1 else 1.


	@classmethod
	def Gaussian_env(cls, r, xi_order=0):
		"""	Return a Gaussian envelope with unit width, or a higher-order version of one.
				
				G_n(r) = \\exp(-r^2/2) \\sum_{i=0}^{n} (r^2/2)^i / i!
					where n is the xi_order
				
				Properties of G_n(r):
				(1)	G_n(0) = 1 ,  G_n(\\inf) = 0 .
				(2)	For r near 0: G_n(r) = 1 - O(r^{2n+2}) .
				(3)	d/dr G_n(r) = -r (r^2/2)^n exp(-r^2/2) / n! .
				Define H_n(r) = -d/dr G_n(r) ,
				(4)	Maximum of H_n occurs at r = \\sqrt{2n+1} .
				(5)	The first moment of H_n is \\sqrt{2} \\Gamma(n+3/2) / n!  \\sim  \\sqrt{2n + 3/2} + O(n^{-3/2}) .
				(6)	The second moment of H_n is simply 2n+2 .
				(7)	From (5),(6), the variance of H_n is approximately to 1/2.
						Hence the derivative H_n is roughly a normalized Gaussian of width 1/\\sqrt{2} centered at \\sqrt{2n+1} .
			"""
		r22 = r**2 / 2.
		G = np.exp(-r22)
		if xi_order <= 0: return G
		p = r22
		c = 1 + r22
		for i in range(2, xi_order+1):
			p = p * r22 / i		# r22^i / i!
			c += p
		return G * c


	@classmethod
	def FermiDirac_tan(cls, L, f, r):
		"""	Returns FDT(r) = [ 1 + Exp(-L cot(pi r/L) / f) ]^(-1)  if 0 <= r < L, 0 otherwise.
				
				Function FDT(r) has the properties:
					FDT(0) = 1
					FDT(r) = 0 when r >= L
					FDT(r) + FDT(L-r) = 1 (for 0 <= r <= L)
					the slope at FDT'(L/2) = -pi/4f

				Vectorized in r.
				L is the circumference, f is the characteristic length scale for decay for FDT(r).
				- This code will fail if r < 0
				- The shape of FDT(r) becomes funky if f > L/2
				- Code may break if type(L) != float
			"""
		# Make sures 0 <= rp <= 0.5, so argument to the np.exp() is always negative to avoid overflows
		# (we take advantage of the feature/bug: np.tan(0.5 * np.pi) is a positive, finite number)
		# ()*1 converts bool to int
		rp = r / L
		rp = np.choose((rp > 0.5)*1 + (rp >= 1)*1, [0.5-rp, rp-0.5, 0])
		H = 1 / ( 1 + np.exp( -L * np.tan(rp * np.pi) / f ))
		return np.choose((r > L/2)*1 + (r >= L)*1, [H, 1-H, 0])


	@classmethod
	def eval_Madelung_energy(cls, rV, Lx, Ly, tol=1e-11, init_rad=None, verbose=1):
		"""	Compute the Madelung energy/particle on a torus.
			
			Parameters:
				rV:  a radial function r*V(r), with V(r) being the potential.
				Lx, Ly:  the circumferences of the torus.
				tol:  target error.
				verbose:  use verbose=0 to suppress non-convergence messages.
			
			This function evaluates [ \\sum_{R \\neq 0} V(|R|) - 2\\pi \\int_0^\\infty r dr V(r) ] / 2,
				with R taken over lattices points (spanned by Lx\\hat{x}, Ly\\hat{y}) except the origin.
			"""
		if Lx <= 0 or Ly <= 0: raise ValueError
		if init_rad is None:
			init_rad = max(Lx, Ly) * 4.6
		if init_rad < 2*max(Lx, Ly): raise ValueError
		if verbose >= 2: print "eval_Madelung_energy():  (Lx, Ly) = ({}, {}), init_rad = {}, tol = {}".format(Lx, Ly, init_rad, tol)

		grid_r = np.array([[0.]])
		grid_rV = np.array([[np.nan]])
		ME = [0.]
		j = 1
		for iterations in range(7):
			newdim = ( int(2*j*init_rad/Lx) + 1, int(2*j*init_rad/Ly) + 1 )
			olddim = grid_r.shape
			#rint j, olddim, newdim
		##	Compute r and rV
			mgrid = np.mgrid[0:newdim[0], 0:newdim[1]].transpose(1,2,0) * np.array([Lx,Ly], dtype=float)
			grid_r = np.sqrt(np.sum(mgrid**2, axis=-1))
			grid_rV = np.pad(grid_rV, [(0, newdim[0]-olddim[0]), (0, newdim[1]-olddim[1])], mode='constant')
			grid_rV[olddim[0]:newdim[0]] = rV(grid_r[olddim[0]:newdim[0]])
			grid_rV[0:olddim[0], olddim[1]:newdim[1]] = rV(grid_r[0:olddim[0], olddim[1]:newdim[1]])
		##	Do the sum
			weight_func = lambda r: 0.5 - 0.5*np.tanh( (r/init_rad-j)+(r/init_rad-j)**3/6 )
			grid_Vwf = grid_rV * weight_func(grid_r) / grid_r		# V(r) * weight_func(r)
			Qsum = np.sum(grid_Vwf[0,1:]) + np.sum(grid_Vwf[1:,0]) + 2*np.sum(grid_Vwf[1:,1:])
		##	Do the integral
			Qint1 = sp.integrate.quad(lambda r: weight_func(r) * rV(r), 0, 2*j*init_rad)
			Qint2 = sp.integrate.quad(lambda r: weight_func(r) * rV(r), 2*j*init_rad, np.inf)
			ME.append( Qsum - np.pi * (Qint1[0]+Qint2[0]) / (Lx*Ly) )
			if verbose >= 2: print '\t', (Qsum,), Qint1, Qint2, ':', (ME[-1],)
			if np.abs(ME[-1] - ME[-2]) < tol: break
			j += 1
		if verbose >= 1 and len(ME) == j:
			print "eval_madelung_energy() failed to converge!"
			print ME[:-1] - ME[-1]
		return ME[-1]


	@classmethod
	def J0_integral(cls, rV, q, tol=1e-15, d = 1, full_output=False):
		"""Integrates  Vq = 2 \\pi \\int_0^\\inf  J_0(q r) rV(r) dr.
				rV is a callable function, with domain [0, +inf) and range in float.
				q is a float
				tol = Desired absolute error
				d = Length scale at which any r-->0 singularities may occur
				full_output is a Boolean: then return number of periods integrated and #func evaluations

			Returns
				if full_output is False:
					Vq
				if full_output is True:
					Vq, number of periods integrated, number of function evaluation

			Example:
				>>>	q = 2.
				>>>	print mod.mQHfunc.J0_integral(lambda r: np.exp(-r), q)
				>>>	print 2 * np.pi / np.sqrt(q**2 + 1)
					2.80992589242
					2.80992589242
	
			Strategy:
			1)	Integrate in subintervals [r_j, r_{j+1}], defined by r_j = pi (j + 3/4) / q .
					The r_j's are the approximate zeros of J_0(q r),
			2)	Tabulate the partial sums, and Euler sum them to accelerate convergence.
				MZ
		"""
		two_pi = 2. * np.pi
		
		def euler_sum(S):
			#Euler summation given partial sums S
			N = len(S) - 1
			binom = [ sp.misc.comb(N, n, exact = True) for n in range(N+1) ]		# binomial coef
			return np.inner(S, binom) / 2**N
			#return np.inner(S, sp.stats.binom.pmf(range(N+1),N, 0.5))
		def I(r):
			#The integrand
			return sp.special.j0(q*r)*rV(r)
	
		n_Eval = 0		# number of rV() evaluations
		
		if q==0.:
			#Break out near singular behaviour near 0
			a0, abserr, infodict = sp.integrate.quad(rV, 0., d, epsabs=1e-16, full_output = True)
			n_Eval+=infodict['neval']
			a1, abserr, infodict = sp.integrate.quad(rV, d, np.inf, epsabs=1e-16, full_output = True)
			n_Eval+=infodict['neval']
			if full_output:
				return (a0+a1) * two_pi, 2, n_Eval
			else:
				return (a0+a1) * two_pi
			
		#Seperate out first interval, [0, r_0], and garauntee subdivision for accuracy
		break_pt = min(0.375*np.pi/q, d)
		a0, abserr, infodict =  sp.integrate.quad(I, 0., np.pi*0.75/q, epsabs=1e-16, full_output = True, points = [break_pt])
		a0 *= two_pi
		n_Eval+=infodict['neval']
		
		#Initialize S, ES
		r, abserr, infodict = sp.integrate.quad(I, np.pi*0.75/q, np.pi*1.75/q, full_output = True)
		r *= two_pi
		n_Eval+=infodict['neval']
		S = [r] #partial sums
		ES = [r] #euler sums
		
		j = 1.75
		while j < 50:
			r, abserr, infodict = sp.integrate.quad(I, np.pi*j/q, np.pi*(j+1.)/q, full_output = True)
			r *= two_pi
			n_Eval+=infodict['neval']
			
			S.append( S[-1] + r )
			
			if abs(r) < max(abserr, tol): #Forget Euler summation - absolute convergence!
				if full_output:
					return S[-1] + a0, int(j+1.25), n_Eval
				return S[-1] + a0
		
			ES.append(euler_sum(S))
		
			if	abs(ES[-1] - ES[-2]) < max(abserr, tol): #Good enough - our work is done.
				if full_output:
					return ES[-1] + a0, int(j+1.25), n_Eval
				return ES[-1] + a0
			
			j+=1.
	
			
		print "\tJ0_integral: Poorly behaved integrand!  q = {}, d = {}, tol = {},  n_Eval = {}".format(q, d, tol, n_Eval)
		print "\t\tS - S[-1]:", np.array(S) - S[-1]
		print "\t\tES - ES[-1]:", np.array(ES) - ES[-1]
		return ES[-1] + a0


	@staticmethod
	def rVr_Haldane_pseudoP_LLL(rV, max_l):
		"""Given an arbitrary function, compute the Haldane pseudopotentials in the lowest LL.
	Return an 1-array (length max_l+1) with the Haldane pseudopotentials.

	The code evaluates the integral
		0.5 \\int_0^\\inf dr rV(r) (r/2)^{2l} e^{-r^2/4} / l!
	"""
		if type(max_l) is not int: raise ValueError
		Haldane_p = np.zeros(max_l + 1, np.float)
		for l in range(max_l + 1):
			if l == 0: log_l_fact = 0.
			if l > 0: log_l_fact += np.log(l)
			integrand = lambda r: rV(r) * np.exp(-r**2/4 + 2*l*np.log(r/2.) - log_l_fact)
			cut = np.sqrt(4 * l + 2)
			#intgl = sp.integrate.quad(integrand, 0., np.inf, epsabs=1e-16)[0]
			intgl1 = sp.integrate.quad(integrand, 0., cut, epsabs=1e-16)
			intgl2 = sp.integrate.quad(integrand, cut, np.inf, epsabs=1e-16)
			intgl = intgl1[0] + intgl2[0]
			Haldane_p[l] = intgl / 2
		return Haldane_p


	@staticmethod
	def Haldane_pseudoP_0_to_1LL(Haldane_pseudoP_0LL):
		"""Given the Haldane Pseudopotentials in the LLL, compute them in the 1st LL."""
		l = np.arange(len(Haldane_pseudoP_0LL) - 2)
		HP0 = np.hstack([ [0.,0.], Haldane_pseudoP_0LL ])
		HP1 = l*(l-1)/4. * HP0[:-4] - l*(l-1) * HP0[1:-3] + (3*l**2-l+1)/2. * HP0[2:-2] - l*(l+1) * HP0[3:-1] + (l+1)*(l+2)/4. * HP0[4:]
		return HP1

	@staticmethod
	def Haldane_pseudoP_0_to_2LL(Haldane_pseudoP_0LL):
		"""Given the Haldane Pseudopotentials in the LLL, compute them in the 2nd LL."""
		ls = np.arange(len(Haldane_pseudoP_0LL) - 4)
		HP0 = np.pad(Haldane_pseudoP_0LL, (4,0), 'constant')
		SecondLLData = np.array([
			[-4,  0,  -6,  11,  -6,  1, 64],
			[-3,  0,   6, -11,   6, -1,  8],
			[-2,  0, -36,  67, -38,  7, 16],
			[-1,  0,  22, -45,  30, -7,  8],
			[ 0, 12, -14,  85, -90, 35, 32],
			[ 1,  0,  -6,   3,   2, -7,  8],
			[ 2,  4  , 0,   7,  18,  7, 16],
			[ 3,  0 , -6, -11,  -6, -1,  8],
			[ 4, 24,  50,  35,  10,  1, 64],
			])
		Coef = SecondLLData[:, 1:6].transpose().astype(float) / SecondLLData[:,6]		# indexed by (power, l offset)
		assert np.max(np.abs( np.sum(Coef, axis=1) - np.array([1,0,0,0,0]) )) == 0
		l_powers = np.vstack([ np.ones(len(ls)), ls, ls**2, ls**3, ls**4 ]).transpose()
		transform_table = np.dot(l_powers, Coef)		# indexed by (m, m offset)
		HP2 = np.zeros(len(ls), np.float)
		for l in ls:
			HP2[l] = np.dot(transform_table[l], HP0[l: l+9])		# perform convolution
		return HP2


####################################################################################################
####################################################################################################

"""

	#@profile
	def from_vqold(self, vq, maxM, maxK, b1, b2):
		 Builds Vmk for a particular species-species interaction given the Fourier transformed interaction vq(qx, qy) (see notes).
		
			maxM, maxK - extent in m, k
			F1, F2 - `form factors' F(a, b, qx, qy) is form factor for band a-b
			eta1/2 - the parities (odd even) of the bands
			b1/2 - the bands
		
		
		kappa = self.kappa
		eta1=b1.eta
		eta2=b2.eta
		F1=b1.F
		F2=b2.F
		pre_F1=b1.pre_F
		pre_F2=b2.pre_F
		n_mu = b1.n
		n_nu = b2.n
		
		vmk = np.zeros( (n_mu, n_mu, n_nu, n_nu, 2*maxM+1, 2*maxK+1) )

		M = self.kappa*np.arange(-maxM, maxM+1)
		K = self.kappa*np.arange(-maxK, maxK+1)
		
		#The integrand obeys f(qy) = f(-qy)^\ast
		#So we restrict to qy>0 and take real part
		
		def fR(qy, M, a, b, c, d):
			return np.real(F1(a, b, M, qy)*F2(c, d, -M, -qy))*vq(M, qy)*np.exp(-0.5*qy**2)
		def fI(qy, M, a, b, c, d):
			return np.imag(F1(a, b, M, qy)*F2(c, d, -M, -qy))*vq(M, qy)*np.exp(-0.5*qy**2)

		#Keep track of which a,b,c,d have been computed (many are related by symmetries)
		did = np.zeros( (n_mu, n_mu, n_nu, n_nu), dtype = np.bool)
		
		#Do the two species have the exact same bands?
		same_bands = b1==b2
		if self.verbose >= 2:
			print "\t\tComputing Vmk [a b c d]: ",		# no new line
		for a, b, c, d in itertools.product(xrange(n_mu), xrange(n_mu), xrange(n_nu), xrange(n_nu) ):
			
			if did[a, b, c, d]:
				if self.verbose >= 3: print '(%s, %s, %s, %s)*, ' % (a, b, c, d),
				continue
			else:
				if self.verbose >= 3: print '(%s, %s, %s, %s), ' % (a, b, c, d),
			
			#The following bands are related by symmetries
			did[a, b, c, d] = True
			did[b, a, d, c] = True
			if same_bands:
				did[c, d, a, b] = True
				did[d, c, b, a] = True
			
			#An m/k indpenedent prefactor - we seperate it from the numerical integration.			
			pre = pre_F1(a, b)*pre_F2(c, d)
			P = eta1(a)*eta1(b)*eta2(c)*eta2(d) #Parity of sector
			
			#Only need upper quadrant in mk space - rest related by symmetry
			for k in range(maxK+1):
				r_mom = None #Cache chebychev momements - they depend only on K
				i_mom = None
				print ".",
				for m in range(maxM+1): #Only upper corner
					
					if m==maxM and P==-1:
						r = 0
					else:
						if r_mom==None:
							out = sp.integrate.quad(fR, 0, 8., weight = 'cos', wvar = K[k],  args = ( M[m], a, b, c, d), full_output = True)
							try:
								r,  abserr, infodict = out
							except:
								print out
								raise
								
							r_mom = (infodict['momcom'], infodict['chebmo'] )
						else:
							out = sp.integrate.quad(fR, 0, 8., weight = 'cos', wvar = K[k],  args = ( M[m], a, b, c, d), wopts = r_mom)
							try:
								r,  abserr = out
							except:
								print out
								raise
					
					#r = sp.integrate.romberg(fC, 0, 8., args = ( M[m],K[k],  a, b, c, d), vec_func=True)
					#The following combaniation have no imag part
					if a==b and c==d or (b1[a]==b2[d] and b1[b]==b2[c]) or k==maxK or (m==maxM and P==1):
						i=0
					else:
						if i_mom==None:
							i,  abserr, infodict = sp.integrate.quad(fI, 0, 8., weight = 'sin', wvar = K[k],  args = ( M[m], a, b, c, d), full_output = True )
							i_mom = (infodict['momcom'], infodict['chebmo'] )
						else:
							i,  abserr = sp.integrate.quad(fI, 0, 8., weight = 'sin', wvar = K[k],  args = ( M[m], a, b, c, d), wopts=i_mom )
							
					r*=np.exp(-0.5*M[m]**2)*pre
					i*=np.exp(-0.5*M[m]**2)*pre
					vmk[a, b, c, d, m, k] = (r+i)
					vmk[a, b, c, d, m, -1 - k] = (r-i)
					vmk[b, a, d, c, -1 - m, k] = (r+i)
					vmk[b, a, d, c, -1 - m, -1 - k] = (r-i)
					
					if a!=b or c!=d:
						vmk[b, a, d, c, m, k] = (r-i)*P
						vmk[b, a, d, c, m, -1 - k] = (r+i)*P
						vmk[a, b, c, d, -1 - m, k] = (r-i)*P
						vmk[a, b, c, d, -1 - m, -1 - k] = (r+i)*P
					
					if same_bands:
						vmk[c, d, a, b, m, k] = (r+i)*P
						vmk[c, d, a, b, m, -1 - k] = (r-i)*P
						vmk[d, c, b, a,  -1 - m, k] = (r+i)*P
						vmk[d, c, b, a, -1 - m, -1 - k] = (r-i)*P
						if a!=b or c!=d:
							vmk[d, c, b, a,  m, k] = (r-i)
							vmk[d, c, b, a, m, -1 - k] = (r+i)
							vmk[c, d, a, b, -1 - m, k] = (r-i)
							vmk[c, d, a, b, -1 - m, -1 - k] = (r+i)
				
		if self.verbose >= 2:
			print 		# terminate the line from "Computing Vmk [a b c d]"

		vmk*=2/np.sqrt(2*np.pi) #factor of two for region qy < 0

		return vmk
"""
