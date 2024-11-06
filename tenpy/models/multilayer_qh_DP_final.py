"""
This is minimal possible code to define calculate Vmk and get MPOs - modified by DP

For other properties look at multilayer_qh including charge conservation etc

"""


""" Multicomponent quantum Hall model.
	Maintained by Mike and Roger.
	How to specify a multicomponent Hilbert space.

	A typical QH interaction in real space looks like
	
		rho_mu(r1)  V_{mu nu}(r1 - r2) rho_nu(r2)
	
	where mu label 'species' (like spin). Each species can have multiple bands - like the multiple Landau levels of each spin. The properties of a band (its form factor) is encoded in a subclass of a 'band' object. A 'layer' is specified by a (species, band) tuple; there is one layer for each type of orbital in the problem. Throughout, Greek indices denote species and latin the bands.
 haldane
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

	How to specify 2-body interactions.
	
	
	Set the Vs model par describe more in make_Vmk
	
	'Vs' = {'eps': 0.0001 (total acceptable truncation of potentials)
			'xiK': 6, (an optional hint giving the rough real-space extent of the potential. 
						Helps scale the overlap integrals properly)
			'screening': Pi(\ell_B q),  (phenomenological screening by filled LLs, see below)
			
			'potential type' : { options, . . . ,  { (mu, nu) : opts, } } #dict specifying parameters for species mu, nu interaction.
			}
			
			
	Screening:
	
		Implements  RPA screening,
	
		Veff(k) = V(k)/(1 - Pi(k) V(k) )
	
		See Papic, 1307.2909 Eq 2. for BLG
		
"""

from scipy.linalg import block_diag




import numpy as np
import scipy as sp
import scipy.integrate
import sys, time
import itertools, functools
from copy import deepcopy
import pickle
#from itertools import izip
#import tenpy.cluster.omp as omp

from tenpy.linalg import np_conserved as npc

from tenpy.additional_import.Zalatel_import import packVmk

#INSTEAD OF LA_TOOLS, TO USE SVD_THETA
#from tenpy.linalg import truncation as LA_tools


from tenpy.models.model_old_tenpy2_final_DP import model, set_var, identical_translate_Q1_data
from tenpy.tools.string import joinstr
from tenpy.tools.math import perm_sign, toiterable, pad
from tenpy.algorithms import Markov

try:
	from matplotlib.pyplot import figure
	import matplotlib.cm as cm
except:
	pass

class QH_model(model): #CHANGED model to Model
	#IN ORDER TO MOVE THIS TO THE MODEL IN TENPY3 WE NEED TO ALSO DEFINE THE LATTICE - SEE WHAT IS THIS LATTICE TYPE EXACTLY ETC
	#

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
			print( "An unambitious Multilayered quantum Hall model")
			print( "\tverbose =", verbose)
		self.dtype = set_var(pars, 'dtype', float)
		
		self.Lx = set_var(pars, 'Lx', 10.)	# circumference
		self.kappa = 2*np.pi/self.Lx
		
	##	Specifying the layers
		if 'layers' in pars:
			if 'Nlayer' in pars: print( "!\tWarning, model parameter 'Nlayer' not used.  (Using 'layers' instead.)")
			layers = set_var(pars, 'layers', [[0,0]]) 	# list of pairs: [ (species, band), ... ]
			N = len(layers)
		else:
			N = set_var(pars, 'Nlayer', 1)	# number of layers
			if N < 1 or (not isinstance(N, int)): raise TypeError((N, type(N)))
			layers = [ [l, 0] for l in range(N) ]		# Default - N distinct species in LLL

		self.Nlayer = N
	##	Compute useful layer data (i.e., species, bands, layers)
		self.species = species = {}
		bands = {}
		for j, (mu, a) in enumerate(layers):
			if type(mu) is tuple:
				print( "Species lables cannot have type 'tuple' ")
				raise ValueError
			species.setdefault(mu, []).append(j)
			bands.setdefault(mu, []).append(a)
		
	##set the type of bands depending the second part of the layer tuple
	#if its a list of integers, use the usual LL subclass corresponding to GaAs, 
	#if a list of lists, use the general_LL subclass, written to handle  graphene & bilayer graphene
		self.bands={}	
		for mu, b in bands.items():
			if hasattr(b[0],'__iter__'):
				if b[0][0]=="tilt":
					self.bands[mu]=tilted_Dirac_cone(b[0][1])
				else:
					self.bands[mu]=general_LL(b)
			else:		
				self.bands[mu]=LL(b)
			
	##  Anisotropy = {mu1: anisotropy1, mu2: anisotropy2, ... } sets anisotropy of form factors
		for mu, anisotropy in set_var(pars, 'anisotropy', {}).items():
			self.bands[mu].anisotropy = anisotropy
					
		self.layers = layers
		if verbose >= 1: print( "\tSpecies {name: orb_inds, ...}:", species)
		if verbose >= 1: print( "\tBands:", self.bands)


	##  Boundary conditions an unit cell:
		# ('periodic'):			with 1-flux unit cell
		# ('periodic', N_phi):	periodic with N_phi flux unit cell
		# ('finite', N_phi):	finite
		if 'L' in pars: 
			print( "!\tWarning! 'L' parameter not used in quantum Hall.  Use 'boundary_conditions': ('periodic', NumPhi) instead.")
		
		pars_bc = set_var(pars, 'boundary_conditions', set_var(pars, 'bc', 'periodic'))
		if isinstance(pars_bc, tuple) or isinstance(pars_bc, list):
			self.bc = pars_bc[0]
			self.L = pars_bc[1] * N
		else:
			self.bc = pars_bc
			self.L = N
		if verbose >= 1: print( "\tL (number of sites):", self.L)
		self.Nflux = self.L // self.Nlayer
		
		self.p_type = p_type = set_var(pars, 'particle_type', 'F')
		

		self.LL_mixing = False
		for mu,b in bands.items():
			if self.bands[mu].n >1: self.LL_mixing = True
		
		
		#DOES SOME CHARGE CONSERVATION STUFF I DO NOT UNDERSTAND YET

		#THIS BIT OF CODE PROBABLY SHOULD NOT AFFECT THE SHAPE OF MPO BUT I LEAVE IT HERE BECAUSE 
		# I GUESS THIS PRODUCES CHARGE MATRICES ETC

		cons_Q = set_var(pars, 'cons_C', 'total')	# either, 'total', 'each', False, 0, or a matrix
		# Q-mat is a [num_cons_C, N] matrix - each row gives the charges of the layers under the conserved quantity
		if cons_Q ==True or cons_Q == 1:
			if N != 1: raise ValueError("=== Mike, update your code.  This is unacceptable. ===")
			if N == 1: Qmat = np.ones([1, 1], dtype=np.int64)	# Fine, Mike, you get a pass here.
		elif cons_Q =='total' or cons_Q == 't':
			Qmat = np.ones([1, N], dtype=np.int64)
		elif cons_Q == 'each' or cons_Q == 'e':
			Qmat = np.eye(N, dtype=np.int64)
		elif cons_Q == 'species' or cons_Q == 's':
			Qmat = np.zeros([len(self.species), N], dtype=np.int64)
			for spi,sp in enumerate(self.species.items()):
				for layeri in sp[1]:
					Qmat[spi,layeri] = 1
		elif cons_Q == False or cons_Q == 0:
			Qmat = np.empty([0, N], dtype=np.int64)
		elif isinstance(cons_Q, np.ndarray) or isinstance(cons_Q, list):
			Qmat = np.atleast_2d(cons_Q)
		else:
			raise ValueError("invalid cons_Q")
		self.num_cons_Q = Qmat.shape[0]
		mod_q = np.ones(self.num_cons_Q, int)
		self.Qmat = Qmat		# Qmat shaped (num_cons_Q, Nlayer)
		self.num_q = self.num_cons_Q

		self.cons_K = set_var(pars, 'cons_K', 1)		# is K conserved (mod q)?  0 means no, 1 means yes (integer), 2 and higher means mod.
		if self.cons_K > 0:
			self.Kvec = np.array(set_var(pars, 'Kvec', np.ones(self.num_cons_Q, int)))	# K = position * (Q . Kvec)
			self.num_q += 1
			mod_q = np.hstack([[self.cons_K], mod_q])
			if self.cons_K > 0:
				self.translate_Q1_data = { 'module':self.__module__, 'cls':self.__class__.__name__,
					'tQ_func':'translate_Q1_Kvec', 'keywords':{ 'Kvec':self.Kvec, 'Nlayer':self.Nlayer } }
			else:
				self.translate_Q1_data = None
			self.set_translate_Q1()
		

		#GET root configuration here
		self.mod_q = mod_q
		self.root_config = np.array(set_var(pars, 'root_config', np.zeros(self.L))).reshape((-1, N))
		self.mu = set_var(pars, 'mu', None) #mu = [mu_{layer1}, mu_{layer2}, ...];  if len(mu) < Nlayer, goes through cyclically
		
		self.Vs = deepcopy(pars.get('Vs', {}))
		exp_approx = self.exp_approx = set_var(pars, 'exp_approx', True)
		self.add_conj = self.ignore_herm_conj = set_var(pars, 'ignore_herm_conj', False)
		self.Vabcd_filter = set_var(pars, 'Vabcd_filter', {})

		#PRODUCE OPERATORS AS NPC ARRAYS- CHARGES NEED TO BE FIXED HERE SOMEWHERE
		##	Process stuff from the parameters collected above
		self.make_Hilbert_space()
		if verbose >= 4:
			print( joinstr(['\tQmat: ', self.Qmat, ',  Qp_flat: ', joinstr(self.Qp_flat)]))
		
		
		#PRODUCE V_MK TERMS
		self.make_Vmk(self.Vs, verbose=verbose)	# 2-body terms
		self.convert_Vmk_tolist()
	

		if 'Ts' in pars:
			self.make_Trm(pars['Ts'])	# 1-body terms
			self.convert_Trm_tolist()
		

		#PRODUCES MPOGRAPH WHICH THEN PRODUCES OUR MPO
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
			if verbose >= 3: print( "\tSlycot loaded: %s, more_slicot loaded: %s" % (Markov.Loaded_Sylcot, Markov.Loaded_more_slicot))
			self.make_MPOgraph1_from_Vmk_slycot(verbose=verbose)
		else:
			print( "Unrecognized options: 'exp_approx': {}".format(exp_approx))
		self.MPOgraph1_to_MPOgraph(verbose=verbose)

		if self.mu is not None:
			self.add_chemical_pot(self.mu)
			
		self.V3=set_var(pars,'nnn',None)		
		if self.V3 is not None:
			self.add_three_body(self.V3)
			
		if hasattr(self, 'Trm_list'):
			self.add_Trm()

		if verbose >= 7: 
			self.print_MPOgraph()

		print("MPO GRAPH IS CONSTRUCTED, SO NEED TO CONSTRUCT HMPO FROM MPOGRAPH....")
		
		#BUILD H_MPO USING NPC ARRAYS - THIS IS THE ONLY BIT I MODIFIED IN model_old_tenpy2 - rest is untouched
		self.build_H_mpo_from_MPOgraph(self.MPOgraph, verbose=verbose)
		
		if verbose >= 2: 
			print( "\tBuilding QH_model took {} seconds.".format(time.time() - self.model_init_t0))


		
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
		for spi,sp in enumerate(self.species.items()):
			Sp_list.append(sp[0])
			Sp_dict[sp[0]] = spi
			Q = self.Qmat[:, np.array(sp[1])]		# columns are the charges
			if Q.shape[1] > 1 and np.any(Q[:, 1:] - Q[:, :-1] > 0): raise RuntimeError(str(Q))
			Q_Sp[spi] = Q[:, 0]
		return Sp_list, Sp_dict, Q_Sp


	def momentum_polarization(self, psi, verbose=0):
		if not self.compatible_translate_Q1_data(psi.translate_Q1_data, verbose=1): raise ValueError
		OrbSpin = np.zeros(self.Nlayer, dtype=float)
		for Sp,band_index in self.species.items():
			for i,ly in enumerate(band_index):
				OrbSpin[ly] = self.bands[Sp].orbital_spin(i)
		q = self.root_config.size		# This should be the qDenom
		if self.p_type == 'F':
			return self.wf_mompol(psi, bonds=None, Lx=self.Lx, OrbSpin=OrbSpin, PhiX=0.5, verbose=verbose)
		else:
			#print( 'momentum_polarization:  This code probably doesn\'t work with p_type != F.  But I\'m going to try anyway.'
			#WARNING: it seems that with bosons the result is too small by a constant factor which (I think) is 0.25
			#we know this from comparisons to RSES
			#so be careful when measuring central charge/topo spin for bosons
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
		if verbose >= 1: print( 'Momentum Polarization:')
		spec = psi.entanglement_spectrum()
		L = psi.L
		Nlayer = psi.translate_Q1_data['keywords']['Nlayer']
		if L % Nlayer:
			print( "L incommesurate with Nlayer")
			raise ValueError
		M = L // Nlayer		# number of orbitals/flux in psi
		Kvec = psi.translate_Q1_data['keywords']['Kvec'] 
		try:
			iter(OrbSpin)
		except TypeError:
			OrbSpin = [OrbSpin] * Nlayer
		if L % len(OrbSpin): raise ValueError
		num_q = psi.num_q	#num_cons_Q = num_q - 1
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
				if (b+1) % Nlayer: raise ValueError((b, Nlayer))	# make sures it's on orbital boundary

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
		if verbose >= 1: print( '\t<n> =', site_avgN, unit_site_avgN)
		
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
					#print( s, site_avgN[s % L] - unit_site_avgN[s % Nlayer], sloc, sloc*(sloc-M)/2./M
					cdw_K += (site_avgN[s % L] - unit_site_avgN[s % Nlayer]) * (sloc * (sloc-M) / 2. / (M-1))
			
			if verbose >= 1: print( '\tbond', b, avgQ, K, KCw, (K-avgQ[0])/qDenom, (K-avgQ[0])/qDenom + np.sum(background_K), ', site loc', (b + 1) // Nlayer + PhiX, cdw_K)
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
				raise ValueError("We don't know what Nlayer is.")
		spectrum = psi.entanglement_spectrum()
		bonds = np.arange(-1, psi.L, Nlayer)		# -1 is included first (and last)
		if isinstance(weight_func, str):
			weight_func = {
						'prob': lambda x: np.sum(np.exp(-2*x)),
						'min': lambda x: np.min(x),
					}[weight_func]	# if weight_func is a string, convert it to a function
		
		Qweights_list = []
		for b in bonds:
			spec = spectrum[b]
			Qspec = {}
			for s in spec:
				Q = s[0][Qi:]
				Qspec.setdefault(Q, []).append(s[1])
			
			if output_form == 'dict':
				Qweights = {}
				for Q,ws in Qspec.items():
					Qweights[Q] = weight_func(np.hstack(ws))
			elif output_form == 'list':
				Qweights = []
				for Q in sorted(Qspec.keys()):
					ws = Qspec[Q]
					Qweights.append( list(Q) + [weight_func(np.hstack(ws))] )
			Qweights_list.append(Qweights)
		
		return Qweights_list


	def charge_polarization_old(self, psi):
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
		cons_K = self.cons_K!=0
		for p in P:
			diffs=p[cons_K:-1,1:]-p[cons_K:-1,:-1]
			for i in range(Nlayer): 
				try:
					n=min(abs(x) for x in diffs[i,:] if abs(x)>0)
					if(n>charges[i]): charges[i]=n
				except ValueError:
					continue

		out=np.empty((len(P),Nlayer))
		for j,p in enumerate(P):			
			#need to subtract off the piece with the highest weight
			k=np.argmax(p[-1,:])
			p[cons_K:-1,:]=p[cons_K:-1,:]-np.multiply(p[cons_K:-1,k,np.newaxis],np.ones([Nlayer,p.shape[1]]))

			#now take the inner product which gives the charge polarization
			out[j,:]=[np.inner(p[cons_K+i,:],p[-1,:])/charges[i] for i in range(Nlayer)]

		
		return out
	
	
	def charge_polarization(self, psi):
		""" Returns the charge polarization on every bond on the chain.
			in the form of a [num_charges,num_bonds] 2-array. Only does physical charges, not k's
			
			
			Pol = mod( sum_a p_a Q_a , 1)
			
			The charge is properly scaled such that the electron has charge as specified in Qspec (as opposed to in Q_flat, where it is scaled by some additional factor "q" )
			
		"""

		Nlayer=self.Nlayer

		P = psi.probability_per_charge(bonds=np.arange(-1, psi.L-1, Nlayer)) #get the (unnormalized) probabilities

		if self.cons_K:
			P = [p[1:, :] for p in P]

		q = np.array([ n[1] for n in self.filling ]) #charge was scaled by this amount
		Q = [ np.mod(np.dot(p[:-1, :] - p[:-1, :1], p[-1, :])/q, 1.) for p in P]

		return Q
	
		

	@classmethod
	def split_site(self, psi, site, x, trunc_par, charge_dist = 'L', verbose = 0, anisotropy=1.):
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
				raise NotImplementedError("ds must be the same for all layers")
		except:
			pass
				
		if site >= L or site < 0: raise ValueError
		if not np.array_equiv(psi.form, np.array([0., 1.])): raise NotImplementedError

	##	The probabilities on each side is given by the error function.
	##		erf(z) is defined as 2/sqrt(pi)*integral(exp(-t**2), t=0..z), with limits -1 and 1 as z -> -inf and inf.
	##		erfc(z) = 1 - erf(z).  (limits 2 and 0 respectively)
#		probL = scipy.special.erfc(x/anisotropy) / 2
#		probR = scipy.special.erfc(-x/anisotropy) / 2

		#this works right now for a single band, in any landau level
		#probably not to hard to generalize it to multiple bands
		if type(self.bands[0]).__name__=='LL' and self.bands[0].n==1:
			anisotropy=self.bands[0].anisotropy
			n=self.bands[0].bands[0]
			probL=self.general_erfc(x/anisotropy,n)
			probR=self.general_erfc(-x/anisotropy,n)
			print( "probs:",x,probL,probR,self.bands[0].bands[0])
		else:
			print( type(self.bands[0]).__name__,self.bands[0].n,self.bands[0].bands[0])
			raise ValueError( "can't to RSES for that kind of system!")
			
		if verbose > 0: print( "\tsplit %2s, x = % 2.3f, amp = (%.9f, %.9f)," % (str(site), x, np.sqrt(probL), np.sqrt(probR)))
		sLeft = psi.s[(site - 1) % sLen]		# s on the left
		sB = psi.B[site].scale_axis(sLeft, axis = 1)		# TODO assumes B form
	
	##	Form the 3-tensor that converts two-particle filling to single particle filling
		partition = np.zeros((d,d,d), float)	# legs L, R, together
		for fL in range(d):
			for fR in range(d):
				if fL + fR < d:
					partition[fL, fR, fL+fR] = np.sqrt(sp.misc.comb(fL+fR,fR))*probL**(fL/2.) * probR**(fR/2.)	# gives the amplitude

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
		
		if verbose > 0: print( "chi = %s, s_trunc = %s (%s mb)" % (len(Y), s_trunc, round(omp.memory_usage())))
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
		maxM = int(np.ceil( 10./self.kappa))
		verbose = self.verbose
		if verbose >= 2: print( '\tmake_Trm:')
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
			for mu, t in Ts['pins'].items():
				if verbose >= 2: print( "\t\tMaking pins: mu = %s, t = %s" % (mu, t))
				trm = get_trm(mu, mu)
				trm += self.t_pins(t, maxM, bands[mu])
		
		if 'Tq' in Ts:
			for mu, t in Ts['Tq'].items():
				if verbose >= 2: print( "\t\tMaking egg carton: mu = %s, t = %s" % (mu, t))
				trm = get_trm(mu, mu)
				self.t_q(t, trm, bands[mu])	
		if 'tunneling' in Ts:
			"""For each (mu, nu), gives
			
				[  t_{mu nu} \int d^2r  psi^D_mu(r) \psi_nu(r)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
			"""
			for (mu, nu), t in Ts['tunneling'].items():
				if mu==nu:
					print( "Warning: mu = nu tunneling")
				if verbose >= 2: print( "\t\tMaking tunneling terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
				trm = get_trm(mu, nu)				
				self.t_tunneling(t, trm, bands[mu], bands[nu])
				trm = get_trm(nu, mu)
				self.t_tunneling(t, trm, bands[nu], bands[mu])
		
		if 'background_x' in Ts:
			""" For each (mu, nu), gives
			
				[   \int d^2r  n_mu(x, y) V_{mu mu}(x, y) t(x)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for mu, t in Ts['background_x'].items():
				if verbose >= 2: print( "\t\tMaking background terms: (mu,) = (%s,), t = %s" % (mu, t))
				trm = get_trm(mu, mu)
				self.background_x(t, trm, mu)

				
		if 'Tx' in Ts:
			""" For each (mu, nu), gives
			
				[   \int d^2r  t_{mu nu}(x) psi^D_mu(x, y) \psi_nu(x, y)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for (mu, nu), t in Ts['Tx'].items():
				if verbose >= 2: print( "\t\tMaking Tx terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
				
				trm = get_trm(mu, nu)
				self.t_x(t, trm, bands[mu], bands[nu])
				trm = get_trm(nu, mu)
				self.t_x(t, trm, bands[nu], bands[mu], conj=True)
				
		if 'Ty' in Ts:
			""" For each (mu, nu), gives
			
				[   0.5 \int d^2r  t_{mu nu}(y) psi^D_mu(x, y) \psi_nu(x, y)  + h.c. ]
				
				Note  + h.c. is included! Even when mu = nu
				
			"""
			for (mu, nu), t in Ts['Ty'].items():
				if verbose >= 2: print( "\t\tMaking tunneling terms: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
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
			print( "Warning: only even part of t_x kept")
		tq = np.real(tq)
		for a in range(band1.n):
			for b in range(band1.n):
				for m in range(maxM+1):
					trm[a, b, :, m] += tq[m]*band1.pre_F(a, b)*band1.F(a,b, -m*self.kappa, 0)*np.exp(-0.5*(m*self.kappa)**2)
				

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
				trm[a, b, :, 0] += s*dy*0.5   #*0.5 coming from convention with + h.c.



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
				print( "nx:", m)
				raise ValueError
			m = int(round(m))
				
			n = qy*self.Nflux/self.Lx
			if abs( n - round(n)) > 10**(-10):
				print( "ny:", n, "Nflux:", self.Nflux,"qy:",qy)
				raise ValueError	
			n = int(round(n))
			
			q2 = qx*qx+qy*qy
			ks = np.arange(self.Nflux)*self.kappa
			for a in range(band.n):
				for b in range(band.n):
					
					if m!=0: #In this case we are gauranteed conjugate entry will be added by make_Trm_list
						trm[a, b, :, m] += 0.5*np.real(tq*band.pre_F(a, b)*band.F(a,b, -qx, -qy)*np.exp(1j*qy*(ks - qx/2.)))
				
					if m==0: 
						trm[a, b, :, 0] += np.real(tq*band.pre_F(a, b)*band.F(a,b, 0, -qy)*np.exp(1j*qy*ks))

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
			print( "WARNING: pins use LLL form factors")
		Tq = np.zeros((self.L // self.Nlayer, maxM+1), dtype = np.complex128)
		
		M, R = np.meshgrid(np.arange(maxM+1,dtype = float), np.arange(self.L // self.Nlayer, dtype = float))
		R*=self.kappa
		M*=self.kappa
		
		for pin in pins:
			x,y,d,h = pin
			a = 1.+2.*d*d
			Tq += h*np.exp(-1j*M*x - 0.25*a*(M**2) - ((y - R + 0.5*M)**2.)/a)/self.Lx/np.sqrt(np.pi*a)
			
		Tq = Tq.reshape((1, 1,) + Tq.shape)
		
		return Tq.real

	################################################################################
	################################################################################
	##	Code to make coefficients (Vmk) for 2-body potentials.

	def save_Vmk(self, filename, note=''):
		data = {'Lx':self.Lx, \
			'maxM':self.maxM, 'maxK':self.maxK, 'Vmk':self.Vmk, 'V_eps':self.V_eps, 'note':note}
		
		with open(filename, 'wb') as f:
			if self.verbose >= 1: print( ("\t\tSaving Vmk file '%s'." % filename))
			pickle.dump(data, f, protocol=-1)


	def load_Vmk(self, filename):
		with open(filename, 'rb') as f:
			if self.verbose >= 1: print( ("\t\tLoading Vmk file '%s'." % filename))
			data = pickle.load(f)
		if data['Lx'] != self.Lx: print( ("Warning! File '%s' Lx = %s different from the current model Lx = %s." % (filename, data['Lx'], self.Lx)))
		self.maxM = data['maxM']
		self.maxK = data['maxK']
		self.V_eps = data['V_eps']
		self.Vmk = data['Vmk']		# TODO, merge instead of overwrite
		#TODO: check if the keys in Vmk matches that of the current model.


	def make_Vmk(self, Vs, verbose=0):
		"""	Parses the Vs model parameters, and calls function to compute matrix elements Vmk. Writes to self.Vmk and self.V_mask
			
			Types of potentials: (see their functions for appropriate dict format)
				'coulomb': 'xi', 'v'   = v*e^{-r/xi}/r
				'coulomb_blgated': Coulomb with symmetricly placed metallic top & bottom gates
				'GaussianCoulomb':			'v', 'xi', ['order']
				'rV(r)':			'rV', ['tol', 'coarse', 'd']  (See vq_from_rV documentation for additional details.)
				'TK'   : [t0, t1, . . . ]
				'haldane' : [V0, V1, . . . ]
				'coulomb_anisotropic': 'v', 'xi', 'lam' (added by MI)
			Keys:
				'eps' : maximum acceptable error in approximating potential
				'xiK' : hint on real space extent of interations.
				'maxM' : over-ride defaults bounds on M
				'maxK' :			   on K
				'screening':  Veff = V / (1 -  V(q) Pi(q ell)),   where screening is the function 'Pi'
								for BLG, use Pi(q ell) = - a tanh(b ell^2 q^2)  (Eq. 2 from arxiv 1307.2909)
			Other:
				'loadVmk'	(probably doesn't work)
				'pickleVmk'	(still works, I think. RM)
		"""
		N = len(self.species)		
		magic_kappa_factor = 0.5 * (2*np.pi)**(-1.5) * self.kappa
		species = self.species
		bands = self.bands
		#self.screening = Vs.setdefault('screening', {'a':0, 'b':0.5} )
		#implements Eq. 2 from arxiv 1307.2909, which gives a phenomenological formula for the LL mixing in graphene
		self.screening = Vs.setdefault('screening', None )
		
		
		
		if verbose >= 2: print( '\tmake_Vmk:')
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
		if maxM > maxK: raise ValueError((maxK, maxM))
	##	Create Vmk if it didn't exist already (i.e., loaded.)
		Vmk = self.Vmk

		if verbose >= 2:
			print( '\t\tmaxM = %s, maxK = %s, eps = %s' % (maxM, maxK, self.V_eps))
			
	##	Helper function for fetching Vmk[mu][nu]'s
	##	Each Vmk[mu][nu] is a 6-array:  ab, cd, m k (or is set to None)
		def get_vmk(mu, nu):
			""" mu, nu can be tuple; if  mu = (a, b) and nu = (c, d), interaction takes the form
			
				: psi*_a(r) psi_b(r) V(r - r')  psi*_c(r') psi_d(r') : / 2
			"""
			if type(mu) is not tuple:
				mu = (mu, mu)
			if type(nu) is not tuple:
				nu = (nu, nu)
			if mu not in Vmk:
				Vmk[mu] = {}
			if nu not in Vmk[mu]:
				n_mu1 = len(species[mu[0]])
				n_mu2 = len(species[mu[1]])
				n_nu1 = len(species[nu[0]])
				n_nu2 = len(species[nu[1]])

				t = np.zeros((n_mu1, n_mu2) + (n_nu1, n_nu2) + (2*maxM+1, 2*maxK+1))

				Vmk[mu][nu] = t

			return Vmk[mu][nu]

		###TODO!!! Fix the sizes, the code 'load' is actually incompatible with the following code.			
		## Look for Potentials in Vs, and generate the Vmk for each.
		if 'TK' in Vs:
			for (mu, nu), t in Vs['TK'].items():
				if verbose >= 2: print( "\t\tMaking TK potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vmk = get_vmk(mu, nu)
				vmk += self.vmk_TK(t, maxM, maxK, bands[mu], bands[nu]) * magic_kappa_factor
				## Should we add t[0] to Vq0 ??? Yes.
		
		if 'haldane' in Vs:
			#Haldane Pseudo-pot - array is strength of mth term
			for (mu, nu), v in Vs['haldane'].items():
				if verbose >= 2: print( "\t\tMaking Haldane pseudopotentials: (mu,nu) = (%s,%s), V_l = %s" % (mu, nu, v))
				vmk = get_vmk(mu, nu)
				if np.prod(vmk.shape[:4]) > 1:
					raise NotImplementedError("Haldane-pseudopotentials do not support LL mixing")
				
				vmk += self.vmk_haldane(v, maxM, maxK) * magic_kappa_factor

		if 'rV(r)' in Vs:
			"""Real space interactions between mu/nu 
					(mu, nu) : {'rV', 'tol', 'd', 'coarse'}
			
				rV is function in 1 variable, r*V(r)
					or a string describing the function, e.g. rV = "lambda r: np.exp(-(r/6.)**2/2.) * r / np.sqrt(r**2 + 1.**2)"
				
				tol, d, coarse are completely optional and specify accuracy of Fourier transform; default values should be fine.
				See vq_from_rV() documentation for additional details.
			"""
			for (mu, nu), v in Vs['rV(r)'].items():
				if verbose >= 2: print( "\t\tMaking rV(r) potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError								
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
			for (mu, nu), v in Vs['coulomb'].items():
				if verbose >= 2: print( "\t\tMaking Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vq = functools.partial(self.vq_coulomb, v['v'], v['xi'])
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi

		if 'coulomb_blgated' in Vs:
			for (mu, nu), v in Vs['coulomb_blgated'].items():
				if verbose >= 2: print( "\t\tMaking Gated Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
# 				vq = functools.partial(self.vq_coulomb_blgated, v['v'], v['d'], v.get('z', 0.) )
# 				vq = functools.partial(self.vq_coulomb_blgated, v['v'], v['d'], v.get('z', 0.), v.get('epsA', 1.), v.get('betaA', 1.) )
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
                    
		if 'coulomb_bilayer' in Vs:
			for (mu, nu), v in Vs['coulomb_bilayer'].items():
				if verbose >= 2: print( "\t\tMaking Gated Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vq = functools.partial(self.vq_coulomb_bilayer, v['v'], v['d'], v.get('z0', 0.), v.get('z', 0.) )
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi
				
		if 'coulomb_anisotropic' in Vs:
			for (mu, nu), v in Vs['coulomb_anisotropic'].items():
				if verbose >= 2: print( "\t\tMaking ad-hoc anisotropic Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vq = functools.partial(self.vq_coulomb_anisotropic, v['v'], v['xi'], v['lam'])
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				self.Vq0[(mu, nu)] += vq(0, 0) / 8. / np.pi
				self.Vq0[(nu, mu)] += vq(0, 0) / 8. / np.pi	
								
		if 'GaussianCoulomb' in Vs:			
			"""
					(mu, nu) : { 'v', 'xi', 'order' }
				"""
			for (mu, nu), v in Vs['GaussianCoulomb'].items():
				if verbose >= 2: print( "\t\tMaking (Gaussian) Coulomb potentials: (mu,nu) = (%s,%s), v = %s" % (mu, nu, v))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
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
			
		#These are redundant; use TK they are faster. Just to check method
		if 'TK_vq' in Vs:
			for (mu, nu), t in Vs['TK_vq'].items():
				if verbose >= 2: print( "\t\tMaking TK_vq potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vq = functools.partial(self.vq_TK, t)
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
		
		#These are redundant; use haldane they are faster. Just to check method
		if 'haldane_vq' in Vs:
			for (mu, nu), t in Vs['haldane_vq'].items():
				if verbose >= 2: print( "\t\tMaking haldane_vq potentials: (mu,nu) = (%s,%s), t = %s" % (mu, nu, t))
				if type(mu) is tuple or type(nu) is tuple:
					raise ValueError
				vq = functools.partial(self.vq_haldanePP, t)
				vmk = get_vmk(mu, nu)
				vmk += self.Vmk_from_vq(vq, mu, nu) * magic_kappa_factor
				

	##	(Anti)symmetrize Vmk
	
		if self.p_type == 'F': eta = -1
		if self.p_type == 'B': eta = 1

		for mu in Vmk.keys():
			for nu in Vmk[mu].keys():
				if mu[0] == nu[0]:
					if Vmk[mu][nu].shape[4] < 2*maxK+1:
						Vmk[mu][nu] = pad(Vmk[mu][nu], maxK-maxM, 0, maxK-maxM, 0, axis = 4)
					Vmk[mu][nu] = 0.5 * (Vmk[mu][nu] + eta*np.transpose(Vmk[mu][nu], [2, 1, 0, 3, 5, 4]))
				if mu[1] == nu[1]:
					if Vmk[mu][nu].shape[4] < 2*maxK+1:
						Vmk[mu][nu] = pad(Vmk[mu][nu], maxK-maxM, 0, maxK-maxM, 0, axis = 4)
					Vmk[mu][nu] = 0.5 * (Vmk[mu][nu] + eta*np.transpose(Vmk[mu][nu], [0, 3, 2, 1, 5, 4])[:, :, :, :, ::-1, ::-1])
				

		self.make_V_mask(self.V_eps)

		if 'pickleVmk' in Vs:
			self.save_Vmk(Vs['pickleVmk'])
		if verbose >= 3: print( '\t\tV(q=0):', self.Vq0)


	def plot_Vmk(self):
		###I broke this function because Vmk is now of the form Vmk[(mu, mu')][(nu, nu')]
		raise NotImplemented
		from matplotlib import pyplot as plt
		N = self.Nlayer
		species = self.species
		
		for mu in species.keys():
			n_mu = len(species[mu])
			for nu in species.keys():
				n_nu = len(species[nu])
				if self.Vmk[mu][nu] is not None:
					for a, b in itertools.product( range(n_mu), range(n_mu)):
						plt.subplot(n_mu, n_mu, 1 + n_mu*a+b)
						plt.imshow(np.abs(self.Vmk[mu][nu][a, b, 0, 0, :, :] ), cmap=plt.cm.gray, interpolation='nearest')
						plt.contour(self.V_mask[mu][nu][a, b, 0, 0, :, :], [0.5])
						plt.title(str(np.count_nonzero(self.V_mask[mu][nu]))+'non-zero')
				
		plt.show()
		
		
	def make_V_mask(self, eps = 0.):
		"""Calculates elements of Vmk to keep in order to retain largest elements.
		
			For 0 < eps < 1, drop at most eps% of total weight (as measured by sum |Vmr|), but keeping at most Dmax values.
			
			If Dmax = None, D will be taken large enough to satisfy eps.
		"""
		if eps >= 1. or eps < 0.:
			raise ValueError( "eps is wacky: " + str(eps)	)	
		mask = {}
		#Each Vmk[mu][nu] is array   ab, cd m k
		def get_mask(mu, nu):
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
		V_mask = {}
		for mu in self.Vmk.keys():
			V_mask[mu] = {}
			for nu in self.Vmk[mu].keys():
				vmask = get_mask(mu, nu)
				if vmask is not None:
					Vmk = self.Vmk[mu][nu]
					Na, Nb, Nc, Nd, maxM, maxK = Vmk.shape
					
					maxM = (maxM-1)//2
					maxK = (maxK-1)//2
					
					for a, b, c, d in itertools.product(range(Na), range(Nb), range(Nc), range(Nd)):
						vmk = Vmk[a, b, c, d].copy()
								
						keep_m0 = keep_which( vmk[maxM, :])
						keep_k0 = keep_which( vmk[:, maxK])
						vmk[maxM, :] = 0.
						vmk[:, maxK] = 0.
						keep = keep_which( vmk)
						for i in keep:
							vmask[a, b, c, d, :, :].flat[i] = True
						vmask[a, b, c, d, maxM, keep_m0] = True
						vmask[a, b, c, d, keep_k0, maxK] = True
				V_mask[mu][nu] = vmask
		self.V_mask = V_mask
	
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
			filter = self.Vabcd_filter[(mu, nu)]
		else:
			filter = None

		P1=b1.P
		P2=b2.P
		F1=b1.F
		F2=b2.F
		pre_F1=b1.pre_F
		pre_F2=b2.pre_F
		
		
		
		def vq_screened(m,q):
			""" V_eff(q) = V(q) / (1 - V(q) Pi(q) )"""
			"""E.g., for BLG  V_eff = V(q) / (1 + V(q) a tanh(b q^2) )"""
			
			V0 = vq(m, qy)
			if self.screening is not None:
				return V0/(1 - V0*self.screening(m, q))
			else:
				return V0
			#return V0/(1 + V0*a_screening*np.tanh(b_screening*q2))
		
		#a_screening, b_screening = self.screening['a'], self.screening['b']
		screened = True
		
		n_mu = b1.n
		n_nu = b2.n
		maxM, maxK = self.maxM, self.maxK
		
		vmk = np.zeros( (n_mu, n_mu, n_nu, n_nu, 2*maxM+1, 2*maxK+1) )

		M = self.kappa*np.arange(-maxM, maxM+1)
		K = self.kappa*np.arange(-maxK, maxK+1)
		
		#Keep track of which a,b,c,d have been computed (many are related by symmetries)
		did = np.zeros( (n_mu, n_mu, n_nu, n_nu), dtype = np.bool_ )
		
		#Do the two species have the exact same bands?
		same_bands = (b1==b2)
		if self.verbose >= 2:
			print( "\t\tComputing Vmk [a b c d]: ")		# no new line
		
		#Determine integration bounds in qy - must be commensurate with Lx. We integrate out to q_max
				
		q_max, mul, n  = self.Vq_discretization_parameters(maxK)
		qy = np.linspace(0, self.Lx/2.*mul, n*mul+1)
		dqy = self.Lx/2./n



		#The integrand obeys f(qy) = f(-qy)^\ast
		def f(qy, M, a, b, c, d):
			
			if screened: return F1(a, b, M, qy)*F2(c, d, -M, -qy)*vq_screened(M, qy)
			else: return F1(a, b, M, qy)*F2(c, d, -M, -qy)*vq(M, qy)
		
		#Only evaluate function out to keep, then pad with 0s
		vr = np.zeros(len(qy), dtype=float)
		vz = np.zeros(len(qy), dtype=np.complex128)
		keep = int(q_max/dqy) #No need to evaluate function beyond here.
		qy = qy[:keep]

		for a, b, c, d in itertools.product(range(n_mu), range(n_mu), range(n_nu), range(n_nu) ):
			
			if filter is not None and not filter[a, b, c, d]:
				continue
			
			if did[a, b, c, d]:
				if self.verbose >= 3: print( '(%s, %s, %s, %s)*, ' % (a, b, c, d))
				continue
			else:
				if self.verbose >= 3: print( '(%s, %s, %s, %s), ' % (a, b, c, d))
			
			#The following bands are related by symmetries
			did[a, b, c, d] = True
			did[b, a, d, c] = True #hermitian conjugation
			if same_bands: #Echange species
				did[c, d, a, b] = True
				did[d, c, b, a] = True #Exchange species + h.c.
			
			#An m/k independent prefactor - we seperate it from the numerical integration.
			pre = pre_F1(a, b)*pre_F2(c, d)*dqy/np.sqrt(2*np.pi)
			P = P1(a, b)*P2(c, d) #Parity of sector
			
			#Only need upper quadrant in mk space - rest related by symmetry

			for m in range(maxM+1): #Only upper corner
				y = f(qy, M[m], a, b, c, d)
				if y.dtype == float:
					vr[:keep] = y
					y = np.fft.hfft(vr)[::mul]
				else:
					vz[:keep] = y
					y = np.fft.hfft(vz)[::mul]
				
				y = np.concatenate((y[-maxK:], y[:maxK+1]))
				y *= pre
				if np.abs(y[-1]) > self.V_eps*0.01:
					print( "\n\tfrom_vq(): Tail end of the Vmk's aren't small enough!  m, M[m], v[-5:] =", m, M[m], y[-5:])
					if np.abs(y[-1]) > 1e-6:
						print( "It means that the tail end of the matrix elements is  not negligible, and more matrix elements should be calculated.\n\n        This will occur if either a) xiK is too small  b) you use an interaction which is\nactually long range (for instance, when you combine gaussian with layer width\n(remember?)\n\nM")

				vmk[a, b, c, d, m, :] =  y
				vmk[b, a, d, c, -1 - m, :] = y #hermitian conjugate

				if a!=b or c!=d:
					vmk[b, a, d, c, m, :] = y[::-1]*P
					vmk[a, b, c, d, -1 - m, :] = y[::-1]*P
				
				if same_bands:
					vmk[c, d, a, b, m, :] = y*P
					vmk[d, c, b, a,  -1 - m, :] = y*P
					if a!=b or c!=d:
						vmk[d, c, b, a,  m, :] = y[::-1] #h.c. + species
						vmk[c, d, a, b, -1 - m, :] = y[::-1] #species
		
		if self.verbose >= 2:
			pass
		
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
			print( "\t\tvq_from_rV is calculating " + str(len(qs)) + " Fourier points in [0, " + str(qs[-1]) + "]")
			vq = np.zeros(len(qs), dtype=float)
			for i,q in enumerate(qs):
				vq[i] = mQHfunc.J0_integral(rV, q, tol=tol, d=d)
				if i % 100 == 99: print( ".")		# comes with progress indicator!
			
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
	def vq_coulomb_blgated_old(self, v, d, z, qx, qy):
		"""	 
			A coulomb potential with metallic gates placed above and below the gas. The top gate is at z =  d/2, the bottom gate is at z = - d/2, and the 2DEG is at z = 0, so for z0 = 0 the gates are d/2 from the gas. The gates screen the interaction. For z = 0, the real-space method of images shows
			
			V(r) = 1/r + 2 sum_{n>0} [ (-1)^n / sqrt{r^2 + (d*n)^2} ]
			V(q) = 2 pi tanh(q d / 2) / q
			
			More generally, we find
			
			V(q) = 2 pi / q    2  sinh( (d/2 - z) q) sinh( (d/2 + z) q) / sinh(d q)
		"""
		q = np.sqrt(qx**2 + qy**2)
		qd = d*q
		zq = z*q
		#mask = qd > 0.05
		#qd2 = qd*qd
		#V = v*(2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, (1./2)*(1 - (1./12)*qd2*(1. - (1/10.)*qd2*(1. - (17./168.)*qd2*(1.-(31./306.)*qd2)))))
		mask = (qd != 0.)
		if z==0.:
			V = v*(2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, 0.5)
		else:
			z_d = z/d
			V = v*(2*np.pi*d)*np.where(mask, 2*np.sinh(qd/2. - zq)*np.sinh(qd/2. + zq)/(np.sinh(qd)*qd), 0.5*(1 - 2*z_d)*(1 + 2*z_d) )
		return V
    
	@classmethod
	def vq_coulomb_blgated(self, v, d, z, epsA, betaA, qx, qy):
		"""	 
			A coulomb potential with metallic gates placed above and below the gas.
			The top gate is at z =  d/2, the bottom gate is at z = - d/2, and the 2DEG is at z.
			The bottom substrate dielectric was taken into account in Ec and rescaling of length.
			We allow the top substrate to be different from the bottom substrate (epsA = eps_t/eps_b, betaA = beta_t/beta_b).
			
			V(q) = 4 pi / q / (epsA coth(betaA (d/2 - z) q) + coth( (d/2 + z) q) )
		"""

		q = np.sqrt(qx**2 + qy**2)
		qd = d*q
		zq = z*q
		#mask = qd > 0.05
		#qd2 = qd*qd
		#V = v*(2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, (1./2)*(1 - (1./12)*qd2*(1. - (1/10.)*qd2*(1. - (17./168.)*qd2*(1.-(31./306.)*qd2)))))
		mask = (qd != 0.)

		z_d = z/d
		eps_beta = epsA/betaA

		V = v*(4*np.pi*d)*np.where(mask, 1./qd/(epsA/np.tanh(betaA*(qd/2. - zq)) + 1./np.tanh(qd/2. + zq)), 0.5*(1 - 2*z_d)*(1 + 2*z_d)/(1 + eps_beta - 2*z_d*(1 - eps_beta)) )

		return V
    
	@classmethod
	def vq_coulomb_bilayer(self, v, d, z0, z, qx, qy):
		"""	 
			A coulomb potential with metallic gates placed above and below the gas. The top gate is at z =  d/2, the bottom gate is at z = - d/2, and the 2DEG is at z = 0, so for z0 = 0 the gates are d/2 from the gas. The gates screen the interaction. For z = 0, the real-space method of images shows
			
			V(r) = 1/r + 2 sum_{n>0} [ (-1)^n / sqrt{r^2 + (d*n)^2} ]
			V(q) = 2 pi tanh(q d / 2) / q
			
			More generally, we find
			
			V(q) = 2 pi / q    2  sinh( (d/2 - z) q) sinh( (d/2 + z) q) / sinh(d q)
		"""
		q = np.sqrt(qx**2 + qy**2)
		qd = d*q
		zq = z*q
		z0q = z0*q
		#mask = qd > 0.05
		#qd2 = qd*qd
		#V = v*(2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, (1./2)*(1 - (1./12)*qd2*(1. - (1/10.)*qd2*(1. - (17./168.)*qd2*(1.-(31./306.)*qd2)))))
		mask = (qd != 0.)
		if z==0.:
			V = v*(2*np.pi*d)*np.where(mask, np.tanh(qd/2)/qd, 0.5)
		else:
			z_d = z/d
			z0_d = z0/d
			V = v*(2*np.pi*d)*np.where(mask, 2*np.sinh(qd/2. - zq)*np.sinh(qd/2. + z0q)/(np.sinh(qd)*qd), 0.5*(1 - 2*z_d)*(1 + 2*z0_d) )
		return V

	def vq_coulomb_anisotropic(self, v, xi, lam, qx, qy):
		"""
			ADDED BY MI
			Coulomb potential in momentum space with anisotropy added ad-hoc
		"""
		q2plus = qx**2 + qy**2 + (1./xi)**2 # regularization: long-distance cutoff in real space
		coulomb = 2*np.pi*v/np.sqrt(q2plus)
		anisotropic = 1. + lam * (qx*qy/q2plus)**2
		out = coulomb * anisotropic**0.25
		return out
			
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
				
				l = float(np.sqrt(1.*sp.special.factorial(b, exact=True)/sp.special.factorial(a, exact=True))*np.sqrt(2)**(b-a))*z**(a-b)*l(0.5*zb*z)
				return P(l.coeffs[::-1])
			else:
				l = sp.special.genlaguerre(a, b-a)*1.
				pre = np.sqrt(1.*sp.special.factorial(a, exact=True)/sp.special.factorial(b, exact=True))*np.sqrt(2)**(a-b)
				l =  float(pre)*((-zb)**(b-a)*l(0.5*zb*z))
				return P(l.coeffs[::-1])
			
		M = self.kappa*np.arange(-maxM, maxM+1)
		K = self.kappa*np.arange(-maxK, maxK+1)
		for a, b, c, d in itertools.product(range(n_mu), range(n_mu), range(n_nu), range(n_nu) ):
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
		

	def vmk_haldane(self, Vm, maxM, maxK):
		""" 
            Calculates matrix elements for Haldane pseudopotentials analytically.
            I never want to think about this again. MZ
		"""
		from numpy.polynomial import Polynomial as P
		from numpy.polynomial import Laguerre as L
		from numpy.polynomial import hermite as H
			
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


	

	@staticmethod
	def translate_Q1_Kvec(Qind, r, Kvec, Nlayer):
		"""	Does the translateQ given Kvec.
			This shouldn't get used as translate_Q1, rather one should use:
			>>>	self.translate_Q1 = functools.partial(QH_model.translate_Q1_Kvec, Kvec = self.Kvec, Nf = self.Nlayer)
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
	#		if verbose > 0: print( 'translate_Q1_data mismatch:\n%s\n%s\n' % (ds, tQdat)
	#		return False
	#	if np.any(ds['keywords']['Kvec'] != tQdat['keywords']['Kvec']):
	#		if verbose > 0: print( 'translate_Q1_data mismatch:\n%s\n%s\n' % (ds['keywords']['Kvec'], tQdat['keywords']['Kvec'])
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
			self.d = 2 * np.ones(L, dtype=np.int64)


			"""
			change slightly by making it diagonal?
			
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
			
			self.Qp_flat = np.zeros([L, 2, self.num_q], np.int64)
			"""

			self.Id = [ np.eye(2) for s in range(N) ]
			self.StrOp = [ np.diag([1., -1]) for s in range(N) ]
			self.nOp = [ np.diag([0., 1]) for s in range(N) ]
			self.nOp_shift=[self.nOp[s]-0.5*self.Id[s] for s in range(N)]#shifted density so that I can easily calculate particle-hole symmetric things using your correlation functions, SG
			self.AOp = [ np.diag([1.], -1) for s in range(N) ]		# creation
			self.aOp = [ np.diag([1.], 1) for s in range(N) ]		# annihilation
			self.invnOp = [ np.diag([1., 0]) for s in range(N) ]
			#self.StrOp,self.nOp,self.nOp_shift,self.AOp,self.aOp,self.invnOp=block_diag(*self.StrOp), block_diag(*self.nOp),block_diag(*self.nOp_shift),block_diag(*self.AOp),block_diag(*self.aOp),block_diag(*self.invnOp)
			#TRY THIS MAYBE STUPID BUT MAYBE WORTH IT
			
			#OK TOUCHED STUFF
			from tenpy.networks.site import QH_MultilayerFermionSite_2
			
			root_config_ = np.array([0,1,0])
			root_config_ = root_config_.reshape(3,1)
			spin_testara=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
			from tenpy.networks.site import QH_MultilayerFermionSite
			#spin_testara=QH_MultilayerFermionSite(N=1)
			#from tenpy.networks.site import FermionSite
			#spin_testara=FermionSite(conserve='N')
			
			self.site_ops = ['Id', 'StrOp', 'nOp', 'AOp', 'aOp', 'invnOp','nOp_shift']

			self.AOp_l = np.outer( self.AOp[0], self.Id[0] ).reshape(2, 2, 2, 2).transpose([0, 2, 1, 3])
			self.AOp_r = np.outer(self.Id[0], self.AOp[0] ).reshape(2, 2, 2, 2).transpose([0, 2, 1, 3])
			self.bond_ops = ['AOp_l', 'AOp_r']
			
			self.Qp_flat = np.zeros([L, 2, self.num_q], np.int64)

			self.Id, self.StrOp, self.nOp, self.nOp_shift,self.AOp,self.aOp,self.invnOp =spin_testara.Id, spin_testara.StrOp, spin_testara.nOp, spin_testara.nOp_shift,spin_testara.AOp,spin_testara.aOp,spin_testara.invnOp 
			
			#self.Id, self.StrOp, self.nOp, self.nOp_shift,self.AOp,self.aOp,self.invnOp =spin_testara.Id, spin_testara.JW, spin_testara.N, spin_testara.dN,spin_testara.Cd,spin_testara.C,spin_testara.InvN 
			print('INNNBIG')
		elif self.p_type == 'B':
		##	Hilbert space & Operators
			d = self.N_bosons + 1		# assume all the d's are the same
			self.d = d * np.ones(L, dtype=np.int64)
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
			self.Qp_flat = np.zeros([L, d, self.num_q], dtype=np.int64)
				
	##	Fill in charges
	##	Get total filling (p/q) in unit cell from root_config
	##		Qmat is 2-array shaped (num_cons_Q, N), denoting unscaled charges of each layer
	##		root_config is a 2-array shaped (#, N)  (# need not equal L/N)
	
		ps = np.sum(np.tensordot(self.Qmat, self.root_config, axes = [[1], [1]]), axis = 1) #ith total charge in root_config
		q = self.root_config.shape[0] * self.root_config.shape[1]		# number of sites
		self.filling = [ (ps[i], q) for i in range(len(ps)) ]		#p/q for each charge
		
		do_cons_K = int(self.cons_K > 0)		# 0 or 1
		for s in range(L):		# runs over sites
			for n in range(self.d[s]):		# run over Hilbert space (empty/full)
				for i in range(len(ps)):	# runs over num_cons_Q
					self.Qp_flat[s, n, i+do_cons_K] = n*q*self.Qmat[i, s%N] - ps[i]
					#C = q N - p
				if do_cons_K:
					self.Qp_flat[s, n, 0] = (s//N) * np.dot(self.Kvec, self.Qp_flat[s,n,1:])
		
		#TEST ME
		#print( q, self.mod_q
	##	Rescale mod_q by q; skip K cons since it isn't scaled if cons_K!=1
		for j in range(self.num_q):
			if self.mod_q[j] > 1:
				self.mod_q[j] *= q
		#print( q, self.mod_q
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
	def convert_Vmk_tolist(self):
		"""	Convert array-style Vmk to (num x 7) list of non-zero Vs.
			Takes in self.Vmk and writes to self.Vmk_list
			Vmk_list explicitly store -m's.
			"""
		Vmk = self.Vmk
		V_mask = self.V_mask
		
		Vmk_list = []
		species = self.species
		for mu in Vmk.keys():
			for nu in Vmk[mu].keys():
				vmk = Vmk[mu][nu]
				mask = V_mask[mu][nu]
				if vmk is not None:
					maxM = (vmk.shape[-2]-1)//2
					maxK = (vmk.shape[-1]-1)//2
					
					#t0 = time.time()

					#CONVERT vmk into float32 because packVmk requires this in cython, and its easier to change it here
					#potentially change in cython
					vmk = np.asarray(vmk, dtype=np.float32)
					packVmk(Vmk_list, vmk, mask, maxM, maxK, species[mu[0]],species[mu[1]], species[nu[0]],species[nu[1]], tol = 1e-12)
					
				####	Below is what packVmk does.
				#	n_mu = len(species[mu])
				#	n_nu = len(species[nu])
				#	spec1 = species[mu]
				#	spec2 = species[nu]
				#	for a, b, c, d, m, k in itertools.product( range(n_mu), range(n_mu), range(n_nu), range(n_nu), range(0, 2*maxM+1), range(0, 2*maxK+1) ):
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
			#print( loc, 'order', order, 'locdist', locdist, 'layer', ly
		##	Get the sign.
		##	The right most operator is applied first, left most last.
		##	Since argsort is stable, when you operators lie on the same site, A is to the left of a.
			if swapph == -1: Vstrength *= perm_sign(order)
			splitV_list.append([(Op[order[0]], ly[0]), locdist[0], (Op[order[1]], ly[1]), locdist[1], \
				(Op[order[2]], ly[2]), locdist[2], (Op[order[3]], ly[3]), Vstrength, must_keep or force_keep])
			if verbose >= 6: print( ' ', Vmk, '->', splitV_list[-1])



	##################################################	
	def make_MPOgraph1_from_Vmk_old(self, verbose=0, force_keep=False):
		"""	Given Vmk_list, make the MPOgraph1.  Make a node for each (m,r) """
		if verbose >= 1: 
			print( "Making MPOgraph1 from Vmk...")
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
			print( (Op1,ly1),m1,(Op2,ly2),r,(Op3,ly3),m2,(Op4,ly4),Vstrength,keep)
			Op1 = Op1[0]; Op2 = Op2[0]; Op3 = Op3[0]; Op4 = Op4[0];
			if np.abs(Vstrength) == 0: continue
			#print( ' ', V
			
		##	Determine sign from ordering of raising/lowering operators (when string operators coincide with AOp).
		##	Order that operators are placed is Op4, and then Op3, Op2, and finally Op1.
			if self.p_type == 'F':
				if Op1 == 'a': Vstrength = -Vstrength	# StrOp from Op2,Op3,Op4
				if Op3 == 'a': Vstrength = -Vstrength	# StrOp from Op4
			
		##	Make the nodes
			if r == 0: print( "    Skipping case r = 0 in %s" % V)
			
			if m1 > 0:
				##	String coming out of Op1
				if Op1 == 'a' or Op1 == 'A':
					addNodes((Op1+'_',ly1), ly1, m1, 'R', (Op1+'Op',1.), 'StrOp', keep)
				else:
					raise NotImplementedError("Don't know what to do with Op1 '%s' in %s" % (Op1,V))
			
			if m1 > 0 and r > 0:
				##	String coming out of Op2
				Op12 = Op1 + Op2
				if Op12 == 'aA' or Op12 == 'Aa' or Op12 == 'aa' or Op12 == 'AA':
					addNodes((Op12+'_',ly1,m1), ly2, r, (Op1+'_',ly1,m1), (Op2+'Op',1.), 'Id', keep)
					pre_Op3_node = (Op12+'_', ly1, m1, r)	# last node before Op3
				else:
					raise NotImplementedError( "Don't know what to do with Op12 '%s' in %s" % (Op12,V))
			
			if m1 == 0 and r > 0:
				##	String coming out of Op12 (because Op1 and Op2 are on the same site)
				Op12 = Op1 + Op2
				if Op12 == 'aa' or Op12 == 'AA': raise NotImplementedError( "Don't know what to do with Op12 '%s' in %s" % (Op12,V))
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
					raise NotImplementedError("Don't know what to do with Op4 '%s' in %s" % (Op4,V))
				post_Op3_node = ('_'+Op4, ly4, m2-1)	# the node right after Op3
			
			if m2 > 0 and r > 0:
				##	The third operator to connect the forward / backward strings
				XNode = MPOgraph1[ly3][pre_Op3_node].setdefault(post_Op3_node, [])
				if Op3 == 'A' or Op3 == 'a':
					XNode.append((Op3+'Op', Vstrength))
				else:
					raise NotImplementedError("Don't know what to do with Op3 '%s' in %s" % (Op3,V))
			
			if r > 0 and m2 == 0:
				##	Op34 to connect the forward string to 'F'
				Op34 = Op3 + Op4
				if Op34 == 'aa' or Op34 == 'AA': raise NotImplementedError("Don't know what to do with Op34 '%s' in %s" % (Op34,V))
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
				if verbose >= 1: print( "\tOnly keeping %s items out of %s in MPOgraph1[%s]" % (len(G), NumNodes, s))
		##	Remove references to dead items
			for s in range(len(MPOgraph1)):
				G = MPOgraph1[s]
				nextG = MPOgraph1[(s + 1) % N]
				for src,dest_dict in G.items():
					for dest in dest_dict.keys():
						if dest not in nextG: del dest_dict[dest]
		
		self.MPOgraph1 = MPOgraph1



	##################################################	
	def make_MPOgraph1_from_Vmk_exp(self, verbose=0):
		"""	Given Vmk_list, make the MPOgraph1.
				Use the (1-in many-out) exp approximation for each (mu,nu,m).
				"""
		if self.p_type != 'F': raise NotImplementedError
		if verbose >= 1: print( "Making MPOgraph1 (with exponential approximation) from Vmk...")
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
				print( "!   Skipping case r = 0 in %s" % V)
			else:
				if len(Vr_list) <= r:
					Vr_list = np.hstack([Vr_list, np.zeros(r+1-len(Vr_list),self.dtype)])
					D0123['Vr'] = Vr_list
				Vr_list[r] += Vstrength
				if must_keep: D0123['keep'] = True
		#changed from python2
		#Opkey_str = lambda (OO1, ll1, mm, OO2, ll2): OO1 + str(ll1) + '_' + str(mm) + '_' + OO2 + str(ll2)
		Opkey_str = lambda tupla: tupla[0] + str(tupla[1]) + '_' + str(tupla[2]) + '_' + tupla[3] + str(tupla[4])
		
		#Opkey_Herm = lambda (OO1, ll1, mm, OO2, ll2): ('a' if OO1 == 'A' else 'A', ll1, mm, 'a' if OO2 == 'A' else 'A', ll2)
	#TODO: pair them up by Herm and use the ignore_herm_conj flag
		
	##	2) Compute the total norms
		total_norms = []
		dropped_norms = []
		for Op01key,Op01Dict in splitV_dict.items():
			for Op23key,Op23Dict in Op01Dict.items():
				if not np.any([ v['keep'] for k,v in Op01Dict.items() ]):
					dropped_norms.append(np.linalg.norm( Op23Dict['Vr'] ))
				total_norms.append(np.linalg.norm( Op23Dict['Vr'] ))
		total_norm = np.linalg.norm(total_norms)
		if verbose >= 1: print( '\tTotal norm:', total_norm)

	##	3) Do the exponential approximation and store the result
		for Op01key,Op01Dict in splitV_dict.items():
			#print(Op01key)
			Op0,ly0,m1,Op1,ly1 = Op01key
			Op23keys =list( Op01Dict.keys())
			#print(isinstance(Op23keys,np.ndarray))
				# define order
			#print(Op23keys)
			if not np.any([ v['keep'] for k,v in Op01Dict.items() ]): continue		# nothing worth keeping, move along
			#print(Op23keys)
		##	First gather the sequences of the numbers
			seqs = []
			for Op23key in Op23keys:
				Op2,ly2,m2,Op3,ly3 = Op23key
				#print('keyara')
				#print(Op23key)
				Op23Dict = Op01Dict[Op23key]
				#print(Op23Dict)
				#print(Op01Dict[Op23key])
				seqs.append( Op23Dict['Vr'][(ly2-ly1-1)%N+1 : : N] )
				#print( '\t ', Op23key, Op23Dict['keep'], seqs[-1]
		##	The approximation
			
			Op01Dict['keys'] = Op23keys	# fix the order of the keys (to match with 'Markov')
			Op01Dict['Markov'] = Markov.seq_approx(seqs, target_err = self.V_eps * total_norm, target_frac_err = 1., max_nodes = None)
		
		
	##	4) Now build the MPOgraph, scan every (Op0,ly0,m1,Op1,ly1) quintuple.
		for Op01key,Op01Dict in splitV_dict.items():
			Op0,ly0,m1,Op1,ly1 = Op01key
			if 'keys' not in Op01Dict:
				if verbose >= 4:
					print( '\t%s\t#nodes = 0/__,      (dropped),       ' % ( Op01key, ))
					print( 'targets = ' + ', '.join([ '%s (%.2e)' % (Opkey_str(Ok), np.linalg.norm(OD['Vr'])) for Ok,OD in Op01Dict.items() if isinstance(Ok, tuple) ]))
				continue		# nothing worth keeping, move along
			#
			Op23keys = Op01Dict['keys']
			#print(Op23keys[('A', np.int64(0), np.int64(0), 'a', np.int64(0))])
			#quit()
			M,entry,exits,MInfo = Op01Dict['Markov'].MeeInfo()
			if verbose >= 3:
				print( '\t%s\t#nodes = %s/%s, norm err = %.7f, ' % ( Op01key, len(M), MInfo['l'],  MInfo['norm_err'] ))
				#COMMENTED OUT
				print( 'targets = ' + ', '.join([ '%s (%.2e)' % (Opkey_str(Ok), np.linalg.norm(OD['Vr'])) for Ok,OD in Op01Dict.items() if isinstance(Ok, tuple) ]))
		
		##	Prepare the nodes leading up to the loop
			if m1 > 0:		# String coming out of Op0
				if Op0 == 'a' or Op0 == 'A':
					pre_Op1_ly, pre_Op1_node = self.addNodes((Op0+'_',ly0), ly0, m1, 'R', (Op0+'Op',1.), 'StrOp')
				else:
					raise NotImplementedError("Don't know what to do with Op0 '%s' in %s" % (Op0,Op01key))
				entry_Op = Op1 + 'Op'
			else:
				Op01 = Op0 + Op1
				pre_Op1_node = 'R'
				if Op01 == 'aA' or Op01 == 'Aa':
					entry_Op = 'nOp'
				else:
					raise NotImplementedError( "Don't know what to do with Op01 '%s' in %s" % (Op01,Op01key))
			
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
			#print(Op23keys)
			for j,Op23key in enumerate(Op23keys):
				#print("keys")
				#
				#print('_________________')
				#print(Op23key)
				#quit()
				#quit()
				Op2,ly2,m2,Op3,ly3 = Op23key
				if m2 > 0:		# The final string leading to 'F' (backwards)
					if Op3 == 'a' or Op3 == 'A':
						post_Op2_ly, post_Op2_node = self.addNodesB(('_' + Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), 'StrOp')
					else:
						raise NotImplementedError( "Don't know what to do with Op3 '%s' in %s" % (Op3,Op23key))
					exit_Op = Op2 + 'Op'
				else:
					Op23 = Op2 + Op3
					if Op23 == 'aA' or Op23 == 'Aa':
						exit_Op = 'nOp'
					else:
						raise NotImplementedError("Don't know what to do with Op23 '%s' in %s" % (Op23,Op23key))
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
			print( '\ttarget fractional error:', self.V_eps)
			norm_errs = np.array([ OD['Markov'].info['norm_err'] for Ok,OD in splitV_dict.items() if 'Markov' in OD ])
			print( '\tMarkov approx error, dropped / total norm: %s, %s / %s' % (np.linalg.norm(norm_errs), np.linalg.norm(dropped_norms), total_norm))
			str_nodes = np.zeros(N, dtype=int)
			M_nodes = np.zeros(N, dtype=int)
			for s in range(N):
				for k in MPOgraph1[s]:
					if '_' in k[0]: str_nodes[s] += 1
					elif len(k) == 6: M_nodes[s] += 1
			print( '\t# of string-nodes:', str_nodes)
			print( '\t# of Markov-nodes:', M_nodes)
		#self.print(_MPOgraph(MPOgraph1)



	##################################################	
	def make_MPOgraph1_from_Vmk_slycot(self, verbose=0):
		"""	Given Vmk_list, make the MPOgraph1.
				Use the slycot (q-in p-out) exp approximation.
				"""
#		if self.p_type !='F': raise NotImplementedError
		if self.p_type=='B' and self.LL_mixing: raise NotImplementedError
		t0 = time.time()
		if verbose >= 1: print( "Making MPOgraph1 (with slycot) from Vmk...")
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
					elif (Op12 == 'AA' or Op12 == 'aa') and self.p_type=='B':
						key12 = Op12
					else:
						raise ValueError("unidentfied operator %s" % Op12)
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
			print( '\tProcessed %s Vmk terms.', len(self.splitV_list))
			if self.ignore_herm_conj: print( '\tAvoiding Hermitian conjugate terms.')
		Opkey_str = lambda tupla: tupla[0] + str(tupla[1]) + '_' + str(tupla[2]) + '_' + tupla[3] + str(tupla[4])
		
		#Opkey_str = lambda (OO1,ll1,mm,OO2,ll2): OO1+str(ll1)+'_'+str(mm)+'_'+OO2+str(ll2)
		#Opkey_Herm = lambda (OO1,ll1,mm,OO2,ll2): ('a' if OO1=='A' else 'A', ll1, mm, 'a' if OO2=='A' else 'A', ll2)
		MS.check_sanity()
		if verbose >= 4: print( "\tr0m0:", r0m0_list)
		
	##	Add in the necessary information for exponential approximation
	##	2a) Make the ignore_r0 table
		for io in MS.io:
			seq = io['seq']
			ignore_r0 = np.zeros(seq.shape[0:2], int)
			for a,o in enumerate(io['out']):
				for b,i in enumerate(io['in']):
					if i[4] >= o[1]:		# compare ly1 to ly2
						ignore_r0[a, b] = 1
						if seq[a, b, 0] != 0: raise RuntimeError((a, b, seq[a, b, 0]))
			#io['ignore_r0'] = ignore_r0
			#print( joinstr([ ignore_r0, seq[:, :, 0] ]), '\n'
	##	2b) Identify the complex conjugates
		HermConj = { 'a':'A', 'A':'a' }
		for ikey in MS.Din:
			Op0,ly0,m1,Op1,ly1 = ikey
			if m1 > 0:
				MS.Din[ikey]['conj'] = (HermConj[Op0],ly0,m1,HermConj[Op1],ly1)
			else:
				Op01=Op0+Op1
				assert m1==0 and ly0==ly1 and (Op01=='AA' or Op01=='aa' or Op01=='Aa')
				MS.Din[ikey]['conj'] = (HermConj[Op1],ly1,m1,HermConj[Op0],ly0)	# assume that Op01 is one of Aa, aa, or AA
		for okey in MS.Dout:
			Op2,ly2,m2,Op3,ly3 = okey
			if m2 > 0:
				MS.Dout[okey]['conj'] = (HermConj[Op2],ly2,m2,HermConj[Op3],ly3)
			else:
				Op23=Op2+Op3
				assert m2==0 and ly2==ly3 and (Op23=='AA' or Op23=='aa' or Op23=='Aa')
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
		#Opkey_str = lambda (OO1,ll1,mm,OO2,ll2): OO1+str(ll1)+'_'+str(mm)+'_'+OO2+str(ll2)
		if self.p_type=='B': sOp='Id'
		elif self.p_type =='F': sOp='StrOp'
		else: raise NotImplementedError
	##	4) Add MPO_graph information to MPO_system, and also make the string nodes.
	##	Sources
		create_nodes_in_list = []
		for Op01key,D in MS.Din.items():
			if MS.io[D['index'][0]]['N'] == 0: continue		# ignore string belonging in blocks with 0 nodes
			(Op0,ly0,m1,Op1,ly1) = Op01key
			if m1 > 0:		# String coming out of 'R' with operator Op0
				if Op0 == 'a' or Op0 == 'A':
					pre_Op1_ly, D['src'] = self.addNodes((Op0+'_',ly0), ly0, m1, 'R', (Op0+'Op',1.), sOp)
				else:
					raise NotImplementedError( "Don't know what to do with Op0 '%s' in %s" % (Op0,Op01key))
				D['Op'] = Op1 + 'Op'
			else:
				Op01 = Op0 + Op1
				if Op01 != 'Aa': raise NotImplementedError("Don't know what to do with Op01 '%s' in %s" % (Op01,Op01key))
				D['src'] = 'R'
				D['Op'] = 'nOp'
			D['orb'] = ly1
			if self.ignore_herm_conj and D['Op'] != 'AOp': create_nodes_in_list.append(Op01key)
	##	Targets
		for Op23key,D in MS.Dout.items():
			if MS.io[D['index'][0]]['N'] == 0: continue		# ignore string belonging in blocks with 0 nodes
			Op2,ly2,m2,Op3,ly3 = Op23key
			if m2 > 0:		# The final string leading to 'F' (backwards) with Op3
				if Op3 == 'a' or Op3 == 'A':
					post_Op2_ly, D['dest'] = self.addNodesB(('_'+Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), sOp)
				else:
					raise NotImplementedError("Don't know what to do with Op3 '%s' in %s" % (Op3,Op23key))
				D['Op'] = Op2 + 'Op'
			else:
				Op23 = Op2 + Op3
				if Op23 != 'Aa': raise NotImplementedError("Don't know what to do with Op23 '%s' in %s" % (Op23,Op23key))
				D['Op'] = 'nOp'
				D['dest'] = 'F'
			D['orb'] = ly2
	##	Prepare the io blocks, and give them names (unique identifiers for the MPOgraph)
		for bid,io in enumerate(MS.io):
			io['StrOp'] = 'Id'		# operator between loop nodes
			Op01key = io['in'][0]		# a sample input key, used to generate a name for the ioblock
			io['name'] = Op01key[0] + str(self.layers[Op01key[1]][0]) + '-' + str((Op01key[2] - Op01key[4] + Op01key[1]) // N) + '-' + Op01key[3] + str(self.layers[Op01key[4]][0]) + '.' + str(bid)
	
	##	5)	Create the nodes for the r = 0 cases
		for r0Op,V in r0_list.items():
			Op0,ly0,m1,Op12,ly12,m2,Op3,ly3 = r0Op		# guarentee that m1, m2 > 0
			if self.ignore_herm_conj:
				if Op0 == 'A': continue
				V = 2. * V
			pre_Op1_ly, pre_Op1_node = self.addNodes((Op0+'_', ly0), ly0, m1, 'R', (Op0+'Op',1.), sOp)
			post_Op2_ly, post_Op2_node = self.addNodesB(('_'+Op3,ly3), ly3, m2, 'F', (Op3+'Op',1.), sOp)
			if (post_Op2_ly - pre_Op1_ly - 1) % N: raise RuntimeError((r0Op, pre_Op1_ly, post_Op2_ly, N))
			XNode = MPOgraph1[pre_Op1_ly][pre_Op1_node].setdefault(post_Op2_node, [])
			XNode.append((Op12+'Op',V))
			if verbose >= 5: print( "\t\tr = 0: (%s, %s, %s) -> (_%s, %s, %s-1), %sOp, ly=%s, V=%s" % (Op0+'_',ly0,m1,Op3,ly3,m2,Op12,ly12,V))
		for r0Op,V in r0m0_list.items():
			if r0Op[0] == 'AAaa':
				XNode = MPOgraph1[r0Op[1]]['R'].setdefault('F', [])
				XNode.append(('AAaaOp',V))
				if verbose >= 4: print( "\tAdding operator 'AAaa' R->F with V = {}".format(V))
			else:
				raise ValueError("don't know what to do with operators %s" % r0Op[0])
		
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
				#if block['conj'][0] < j: continue		# Only print( one in a Hermitian conjugates pair
				if len(block['A']) > 0 or verbose >= 4:
					print( '\t\t%2d (%2d*): %2d nodes / %s, norm err: %.3e / %.5e,' % ( j, block['conj'][0], block['N'], block['num_above_tol'], block['norm_err'], block['2norm'] ))
					print( ' [%s]  %s  ->  %s' % ( block['name'], ', '.join(map(Opkey_str, block['in'])), ', '.join(map(Opkey_str, block['out'])) ))
					#graph_err = np.array([ MS.calc_MPOgraph_err(MPOgraph1, i) for i in block['in'] ])
					#if len(block['A']) > 0: print( joinstr(['\t\t\t', np.linalg.norm(graph_err.reshape(-1)), graph_err])
					#print( '\t\t\t', block['conj']
		if verbose >= 1:
			print( '\ttarget fractional error: %s  (tol = %s)' % (self.V_eps, MS.info['ABC_tol']))
			print( '\tnorm err / total norm: %s / %s' % (MS.info['total_norm_err'], np.linalg.norm([MS.info['2norm']] + [V for OOOO,V in r0_list.items()])))
			str_nodes = np.zeros(N, dtype=int)
			M_nodes = np.zeros(N, dtype=int)
			for s in range(N):
				for k in MPOgraph1[s]:
					if isinstance(k, tuple):
						if '_' in k[0]: str_nodes[s] += 1
						elif k[0] == 'Mk': M_nodes[s] += 1
			print( '\t# of string-nodes:', str_nodes)
			print( '\t# of Markov-nodes:', M_nodes)
			print( '\tConstructing MPOgraph took {} seconds'.format(time.time() - t0))
		
	##	8) Wrap-up
		self.MPOgraph1 = MPOgraph1
		del self.MPOgraph		# Undo the hack earlier
		#print(MS)
		self.Vmk_MS = {'version':1.2, 'Nlayer':self.Nlayer, 'MS':MS, 'r0_list':r0_list, 'r0m0_list':r0m0_list,
			'Lx':self.Lx, 'Vq0':self.Vq0, 'p_type':self.p_type, 'species':self.species, 'bands':self.bands}
		#self.print(_MPOgraph(MPOgraph1)



	def save_Markov_system_to_file(self, filename):
		"""Once a slycot exp. approximation has been used, this function will save its work to disk.
		Use load_Markov_system_from_file() to recover the MPOgraph.
			"""
		with open(filename, 'wb') as f:
			print ("Saving Markov system to file '{}'...".format(filename))
			pickle.dump(self.Vmk_MS, f, protocol=-1)

	def load_Markov_system_from_file(self, filename, verbose=0):
		"""Load a Markov system from file.
		This function is paired with save_Markov_system_to_file(), and to be used in place of make_MPOgraph1_from_Vmk_slycot()
			"""
		t0 = time.time()
		with open(filename, 'rb') as f:
			if verbose >= 1: print ("\tLoading Markov system from file '{}'...".format(filename))
			Vmk_MS = cPickle.load(f)
		if verbose >=1: print( "\tkeys: ", Vmk_MS.keys())	#TODO, check compatibility of species and bands
		if 'version' not in Vmk_MS: raise RuntimeError
		if Vmk_MS['version'] < 1.: raise RuntimeError
		if Vmk_MS['Nlayer'] != self.Nlayer: raise RuntimeError("Error!  Nlayer mismatch: {} != {}".format(Vmk_MS['Nlayer'], self.Nlayer))
		if Vmk_MS['Lx'] != self.Lx: print( "Warning!  Lx mismatch {} != {}".format(Vmk_MS['Lx'], self.Lx))
		if Vmk_MS['p_type'] != self.p_type: print( "Warning!  p_type mismatch {} != {}".format(Vmk_MS['p_type'], self.p_type))
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
		if verbose >= 1: print( "Making empty MPOgraph1.")
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
		if not isinstance(MPOgraph1, list): raise ValueError(type(MPOgraph1))
		N = len(MPOgraph1)
		self.MPOgraph = [ deepcopy(MPOgraph1[s%N]) for s in range(self.L) ]

	def add_chemical_pot(self, mu):
		"""	Given mu, add a chemical potential for each LL. """
		try:
			iter(mu)
		except TypeError:
			mu = [mu]
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
					for a, b, r, m in itertools.product( range(n_mu), range(n_mu), range(trm.shape[2]), range(trm.shape[3])  ):
					
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
				if verbose >= 3: print( '  ', V, '->', [(loc0%N, 'n')])

			elif loc0 > loc1:		# aOp to the left of AOp
				if self.ignore_herm_conj: Tstrength = 2. * Tstrength		# double
				##	Add a string 'a' from loc1 to locMid, and string '_A' from locMid to loc0.
				pre_Op2_ly,pre_Op2_node = self.addNodes(('a_', loc1%N), loc1%L, locMid-loc1, 'R', ('aOp',1.), 'StrOp')
				ly,post_Op2_node = self.addNodesB(('_A', loc0%N), loc0%L, loc0-locMid, 'F', ('AOp',1.), 'StrOp')
				XNode = MPOgraph[pre_Op2_ly][pre_Op2_node].setdefault(post_Op2_node, [])
				if locMid > loc1: XNode.append(('StrOp', Tstrength))
				if locMid == loc1: XNode.append(('aOp', Tstrength))
				if verbose >= 3: print( '  ', V, '->', [('a', loc1%N), loc0-loc1, ('A', loc0%N)])

			elif loc0 < loc1:		# AOp to the left of aOp
				if self.ignore_herm_conj: continue		# ignore this piece
				##	Add a string 'A' from loc0 to locMid, and string '_a' from locMid to loc1.
				pre_Op2_ly,pre_Op2_node = self.addNodes(('A_', loc0%N), loc0%L, locMid-loc0, 'R', ('AOp',1.), 'StrOp')
				ly,post_Op2_node = self.addNodesB(('_a', loc1%N), loc1%L, loc1-locMid, 'F', ('aOp',1.), 'StrOp')
				XNode = MPOgraph[pre_Op2_ly][pre_Op2_node].setdefault(post_Op2_node, [])
				if locMid > loc0: XNode.append(('StrOp', Tstrength))
				if locMid == loc0: XNode.append(('AOp', Tstrength))
				if verbose >= 3: print( '  ', V, '->', [('A', loc0%N), loc1-loc0, ('a', loc1%N)])


	def general_erfc(self,x,n):
		#if n==0: return sp.special.erfc(x)/2.

		if x<=-3.5: return 1
		elif x>=4: return 0	

		if not hasattr(self,"dat_general_erfc"):
			from numpy.polynomial.hermite import hermval
			xs=np.arange(-5,5,0.001)
			ns=[0]*(n)+[1]
			
			def psi(xx):
				return  hermval(xx,ns)**2*np.exp(-xx**2)/np.sqrt(np.pi)/2.**n/(1.*sp.special.factorial(n))
			
			integrated=np.zeros(xs.shape)
			for i,xx in enumerate(xs):
				integrated[i]=0.5-sp.integrate.quad(psi,0,xx)[0]
			
			self.dat_general_erfc=sp.interpolate.interp1d(xs,integrated)
			print( "init for x=",x,n)
		return self.dat_general_erfc(x)

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
	
	stacked = np.array([ np.tile(ls[i], max_len//len(ls[i])) for i in range(Nlayer) ])
	stacked = np.reshape(stacked.transpose(), (-1,))
	return stacked

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
			n(a, b, y, dy) = phi^*_a(y + dy) phi^*_b(y)  , y is array, dy is scalar
			
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
			
		for i in range(self.n):
			if self[i]!=other[i]:
				return False
				
		return True
	
	def __repr__(self):
		return self.__class__.__name__ + str(self.bands)
	
	def phi(self, a, y):
		raise NotImplementedError	# need to overload
	def eta(self, a):
		raise NotImplementedError	# need to overload
	def n(self, a, b, y):
		raise NotImplemented
	def P(self, a, b):
		raise NotImplementedError	# need to overload
	def pre_F(self, a, b):
		return 1.
	def F(self, a, b, qx, qy):
		raise NotImplementedError	# need to overload
	def orbital_spin(self, a):
		return 0.




def F_NM(a, b, qx, qy):
	"""Conventional form factors"""
	q2 = qy**2 + qx**2
	if a==0 and b==0:
		return  np.exp(-q2/4.)
	elif a > b:
		return   (  np.sqrt(1.*sp.special.factorial(b, exact=True)/sp.special.factorial(a, exact=True))*np.sqrt(2.)**(b-a)  )*np.exp(-q2/4.) * (qx-1j*qy)**(a-b) * sp.special.eval_genlaguerre(b, a-b, q2/2.)
	if a < b:
		return   (np.sqrt(1.*sp.special.factorial(a, exact=True)/sp.special.factorial(b, exact=True))*np.sqrt(2.)**(a-b))*np.exp(-q2/4.) * (-qx-1j*qy)**(b-a) * sp.special.eval_genlaguerre(a, b-a, q2/2.)
	else:
		return np.exp(-q2/4.) * sp.special.eval_genlaguerre(a, 0, q2/2.)

def phi_N(a, y):
	"""Conventional Landau gauge orbitals"""
	return np.exp(-0.5*y**2) * sp.special.eval_hermite(a, y) / np.sqrt(np.sqrt(np.pi) * sp.special.factorial(a, exact=True) * 2**a)
	
	

		
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
		#print(b)
		#print(sp.special.factorial(b, exact=True))
		if a==b:
			return 1.
		elif a > b:
			#misc replaced by special
			return np.sqrt(1.*sp.special.factorial(b, exact=True)/sp.special.factorial(a, exact=True))*np.sqrt(2.)**(b-a)
		else:
			#misc replaced by special
			return np.sqrt(1.*sp.special.factorial(a, exact=True)/sp.special.factorial(b, exact=True))*np.sqrt(2.)**(a-b)

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
		return phi_N(a, y)
	
	def phid_phi(self, a, b, y1, y2):
		#phi*_a()
		
		a = self[a]
		b = self[b]
		
		return phi_N(a, y1)*phi_N(b, y2)

	def orbital_spin(self, a):
		a = self[a]
		return a + 0.5

	@staticmethod
	def latex_F(a, b):
		from fractions import Fraction as Fr
		max_ab = max(a, b)
		min_ab = min(a, b)
		diff_ab = max_ab - min_ab
		sqrt_coef = Fr(sp.special.factorial(min_ab, exact=True), sp.special.factorial(max_ab, exact=True) * 2**diff_ab)
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

class general_LL(bands):
	""" Allows you set a linear combination of form factors.
		Band types are numbers - the LL.
		
		Consider a system with orthogonal orbitals A1, A2 . . . 
		
		Let |A, N> denote a GaAs Nth LL wavefunction localized on the "A" orbitals (I supress the moment index). A general-LL "layer" |a>
		can be expressed as
		
			|a> = \sum_{N, A}  u_{a, NA} |N, A>
			
		
		The form factors F_{ab} are then given as linear combinations of GaAs G_{NM}:
		
		F_{ab} = u*_{aAN} F_{NM} u_{b MA}
		
		
		Thus the structure is defined by the u_{aAN}. It is stored as follows. At the top level, it is a list over "layers":
		
		u = [ u_{a=0}, u_{a=1}, ...]
		
		Each layer u_a is list over "A", but a dictionary over N:
		
		u_a = [ { N :u_[a, A=0, N]  }, { N :u_[a, A=1, N]  },  . . .  ]
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
				norm2 += np.linalg.norm(np.array(uaA.values()))**2
				p = list(set((-1)**np.array(uaA.keys())))
				if len(p)==0:
					PaA.append(np.nan)
				elif len(p)==1:
					PaA.append(p[0])
				else:
					raise ValueError
			if np.abs(norm2 - 1.) > 1e-12:
				print( "WARNING: u has norm {}".format(norm2))
			Pa.append(np.array(PaA))
	
		Pab = np.ones( (len(u), len(u)) , dtype = np.int64)
		for a in range(len(u)):
			for b in range(len(u)):
				p = list(set( Pa[a]*Pa[b]))
				if len(p)==1:
					if np.isnan(p[0]):
						Pab[a, b] = 1 #won't be used since 0, so safe to default.
					else:
						Pab[a, b] = p[0]
				
				elif len(p)==2:
					#print( p[0], np.isnan(p[0])
					if np.isnan(p[0]):
						Pab[a, b] = p[1]
					elif np.isnan(p[1]):
						Pab[a, b] = p[0]
					else: #contains conflicting parities
						raise ValueError
				elif len(p)==3:
					raise ValueError
		#if self.verbose > 2:
		#	print( "F_ab parities", Pab
	
		self.Pab = Pab

	def eta(self, a):
		raise NotImplemented
	
	def P(self, a, b):
		return self.Pab[a, b]
	
	def pre_F(self, a, b):
		return 1.

	def F(self, a, b, qx, qy):
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		try:
			len(qx)
			F = np.zeros_like( qx )
		except:
			F = 0.
		
		for ua, ub in zip(self.u[a], self.u[b]):
			for k1, v1 in ua.items():
				for k2, v2 in ub.items():
					F = F + np.conj(v1)*v2*F_NM(k1, k2, qx, qy)
				
		return F

	def phi():
		a = self[a]
		return phi_N(a, y)
#class to test a tilted Dirac cone like in 1701.07836
#code only works for n=1 LL, and is not normalized
class tilted_Dirac_cone(LL):
	def __init__(self,tau):
		self.tau=tau
		self.bands=[1]
		self.n=1
		self.anisotropy=1
		
	def F(self, a, b, qx, qy):
		qx = qx / float(self.anisotropy)
		qy = qy * float(self.anisotropy)
		
		a = self[a]
		b = self[b]

		q2 = qy**2 + qx**2
		return np.exp(-q2/4.)*(1-q2/4.)*(1+1j*qx*self.tau/np.sqrt(2))

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
	def Pi_BLG(cls, a, b, qx, qy):
		q2 = qx*qx + qy*qy
		return -a*np.tanh(b*q2)
	
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
			#print( r, y[i], ya1[1], ya2[1], yb1[1], yb2[1], '\t', abserr[i], '\t', yW[0]-y[i], yW[1]
		x = np.hstack([[0], np.sqrt(np.arctan(r_samples)*2/np.pi), [1]])
		y = np.hstack([[0], y, [1]])
		from scipy.interpolate import interp1d
		mQHfunc.dat_InfSqWell_rVr = interp1d(x, y, kind=5)
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
			print( "WARNING: using Seperate_InfSqWell with 0 separation will likely cause vq_from_rVr to crash")
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
	def J0_integral(cls, rV, q, tol=1e-15, d = 1, dr = np.inf, full_output=False):
		"""Integrates  Vq = 2 \\pi \\int_0^\\inf  J_0(q r) rV(r) dr.
				rV is a callable function, with domain [0, +inf) and range in float.
				q is a float
				tol = Desired absolute error
				d = Length scale at which any r-->0 singularities may occur
				dr = Optional; length scale beyond which rV(r) is negligible. Can be important to know when 1 >> dr q
				full_output is a Boolean: then return number of periods integrated and #func evaluations

			Returns
				if full_output is False:
					Vq
				if full_output is True:
					Vq, number of periods integrated, number of function evaluation

			Example:
				>>>	q = 2.
				>>>	print( mod.mQHfunc.J0_integral(lambda r: np.exp(-r), q)
				>>>	print( 2 * np.pi / np.sqrt(q**2 + 1)
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
			binom = [ sp.special.comb(N, n, exact = True) for n in range(N+1) ]		# binomial coef
			return np.inner(S, binom) / 2**N
			#return np.inner(S, sp.stats.binom.pmf(range(N+1),N, 0.5))
		def I(r):
			#The integrand
			return sp.special.j0(q*r)*rV(r)
	
		n_Eval = 0		# number of rV() evaluations
		n_Quad = 0
		status = 'converged'
		if q==0.:
			#Break out near singular behaviour near 0
			a0, abserr, infodict = sp.integrate.quad(rV, 0., d, epsabs=tol/50., full_output = True)[:3]
			n_Eval+=infodict['neval']

			a1, abserr, infodict = sp.integrate.quad(rV, d, np.inf, epsabs=tol/50., full_output = True)[:3]
			n_Eval+=infodict['neval']
			n_Quad+=2
			if full_output:
				return (a0+a1) * two_pi, {'n_quad':n_Quad, 'n_eval':n_eval, 'converged':True}
			else:
				return (a0+a1) * two_pi
			
		#Seperate out first interval, [0, r_0], and garauntee subdivision for accuracy
		break_pt = min(0.375*np.pi/q, d)
		if np.pi*0.75/q < dr:
			break_pt = min(0.375*np.pi/q, d)
			a0, abserr, infodict =  sp.integrate.quad(I, 0., np.pi*0.75/q, epsabs=tol/50., full_output = True, points = [break_pt])[:3]
			n_Eval+=infodict['neval']
			n_Quad+=1
		else:
			break_pt = min(dr/2., d)
			a0, abserr, infodict =  sp.integrate.quad(I, 0., dr, epsabs=tol/50., full_output = True, points = [break_pt])[:3]
			n_Eval+=infodict['neval']
			a1, abserr, infodict =  sp.integrate.quad(I, dr, np.pi*0.75/q, epsabs=tol/50., full_output = True)[:3]
			n_Eval+=infodict['neval']
			n_Quad+=2
			a0+=a1
		a0 *= two_pi
		
		
		#Initialize S, ES
		r, abserr, infodict = sp.integrate.quad(I, np.pi*0.75/q, np.pi*1.75/q, full_output = True)[:3]
		r *= two_pi
		n_Eval+=infodict['neval']
		n_Quad+=1
		S = [r] #partial sums
		ES = [r] #euler sums
		
		j = 1.75
# 		while j < 50:
		while j < 1000:
			r, abserr, infodict = sp.integrate.quad(I, np.pi*j/q, np.pi*(j+1.)/q, full_output = True)[:3]
			r *= two_pi
			n_Eval+=infodict['neval']
			n_Quad+=1
			S.append( S[-1] + r )
			
			if abs(r) < max(abserr, tol): #Forget Euler summation - absolute convergence!
				if full_output:
					return S[-1] + a0, {'n_quad':n_Quad, 'n_eval':n_eval, 'converged':True}
				else:
					return S[-1] + a0
			ES.append(euler_sum(S))
		
			if	abs(ES[-1] - ES[-2]) < max(abserr, tol): #Good enough - our work is done.
				if full_output:
					return ES[-1] + a0, {'n_quad':n_Quad, 'n_eval':n_eval, 'converged':True}
				return ES[-1] + a0
			
			j+=1.
	
		print("\tJ0_integral: Poorly behaved integrand!  q = {}, d = {}, tol = {}, n_Quad = {}, n_Eval = {}, ".format(q, d, tol, n_Quad, n_Eval))
		print("\t\tS - S[-1]:", np.array(S) - S[-1])
		print("\t\tES - ES[-1]:", np.array(ES) - ES[-1])
		if full_output:
			return ES[-1] + a0, {'n_quad':n_Quad, 'n_eval':n_eval, 'converged':False, 'dS':np.array(S) - S[-1], 'dES':np.array(ES) - ES[-1]}
		else:
			return ES[-1] + a0

				

	@staticmethod
	def rVr_Haldane_pseudoP_LLL(rV, max_l):
		"""Given an arbitrary function, compute the Haldane pseudopotentials in the lowest LL.
	Return an 1-array (length max_l+1) with the Haldane pseudopotentials.

	The code evaluates the integral
		0.5 \\int_0^\\inf dr rV(r) (r/2)^{2l} e^{-r^2/4} / l!
	"""
		if type(max_l) is not int: raise ValueError
		Haldane_p = np.zeros(max_l + 1, float)
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
		HP2 = np.zeros(len(ls), float)
		for l in ls:
			HP2[l] = np.dot(transform_table[l], HP0[l: l+9])		# perform convolution
		return HP2


####################################################################################################
####################################################################################################


