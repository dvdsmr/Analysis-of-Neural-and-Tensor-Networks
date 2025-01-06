import torch
torch.set_default_dtype(torch.float64)

from utilities import prod

from copy import copy, deepcopy
from itertools import product
#    
from colorama import Fore, Style

import time

class TensorTrain:
    def __init__(self, dims, comp_list = None):
        
        self.n_comps = len(dims)
        self.dims = dims
        self.comps = [None] * self.n_comps

        self.rank = None
        self.core_position = None

        # upper bound for ranks
        self.uranks = [1] + [min(prod(dims[:k+1]), prod(dims[k+1:])) for k in range(len(dims)-1)] + [1]

        if comp_list is not None:
            self.set_components(comp_list)

    

    @staticmethod
    def ttsvd(A_full, ranks = None):
        """
        Obtains a TensorTrain from a full tensor via tt svd
        """
        d = len(A_full.shape)
        shapes = A_full.shape
        A_mat = A_full

        # if no ranks are provided, choose maximum possible ranks
        if ranks is None: 
            ranks = [1] + [min(prod(shapes[:mu+1]), prod(shapes[mu+1:])) for mu in range(d-1)] + [1]

        comps = []
        for mu in range(d-1):
            A_mat = A_mat.reshape((ranks[mu]*shapes[mu],-1))
            u, sigma, vt = torch.linalg.svd(A_mat)
            # truncation: 
            u, sigma, vt = u[:,:ranks[mu+1]], sigma[:ranks[mu+1]], vt[:ranks[mu+1],:]

            u_comp = u.reshape(ranks[mu], shapes[mu], ranks[mu+1])
            comps.append(u_comp)

            A_mat = torch.diag(sigma) @ vt

        comps.append(A_mat.unsqueeze(2))
        return TensorTrain(A_full.shape,comps)



    def set_components(self, comp_list):
        """ 
           @param comp_list: List of order 3 tensors representing the component tensors
                            = [C1, ..., Cd] with shape
                            Ci.shape = (ri, self.dims[i], ri+1)
                            
                            with convention r0 = rd = 1

        """
        # the length of the component list has to match 
        assert(len(comp_list) == self.n_comps)
    
        # each component must be a order 3 tensor object
        for pos in range(self.n_comps):
            assert(len(comp_list[pos].shape)==3)
        
        # the given components inner dimension must match the predefined fixed dimensions
        for pos in range(self.n_comps):
            assert(comp_list[pos].shape[1] == self.dims[pos])
            
        # neibourhood communication via rank size must match
        for pos in range(self.n_comps-1):
            assert(comp_list[pos].shape[2] == comp_list[pos+1].shape[0])

        # setting the components
        for pos in range(self.n_comps):
            self.comps[pos] = deepcopy(comp_list[pos])

    def fill_random(self, ranks, eps):
        """
            Fills the TensorTrain with random elements for a given structure of ranks.
            If entries in the TensorTrain object have been setted previously, they are overwritten 
            regardless of the existing rank structure.

            @param ranks #type list
        """
        self.rank = ranks
        
        for pos in range(self.n_comps):
            self.comps[pos] = eps * torch.rand(self.rank[pos], self.dims[pos], self.rank[pos+1])
        
    def full(self):
        """
            Obtain the underlying full tensor. 

            WARNING: This can become abitrarily slow and may exceed memory.
        """
        res = torch.zeros((self.dims))
        for idx in product(*[list(range(d)) for d in self.dims]):  
            val = torch.tensor([1.]) 
            for k, c in enumerate(self.comps):
                val = torch.matmul(val, c[:,idx[k],:].reshape(c.shape[0],-1))
            res[idx] = val

        return res

    def __shift_to_right(self,pos, variant):
        c = self.comps[pos]
        s = c.shape
        c = left_unfolding(c)
        if variant == 'qr': 
            q, r = torch.linalg.qr(c) 
            self.comps[pos] = q.reshape(s[0],s[1],q.shape[1])
            self.comps[pos+1] = torch.einsum('ij, jkl->ikl ', r, self.comps[pos+1] )
        else : # variant == 'svd'
            u, S, vh = torch.linalg.svd(c,  full_matrices=False)
            u, S, vh = u[:,:len(S)], S[:len(S)], vh[:len(S),:]

            # store orthonormal part at current position
            self.comps[pos] = u.reshape(s[0],s[1],u.shape[1])
            self.comps[pos+1] = torch.einsum('ij, jkl->ikl ', torch.diag(S)@vh, self.comps[pos+1] )
            

    def __shift_to_left(self, pos, variant):
        c = self.comps[pos]
    
        s = c.shape
        c = right_unfolding(c)
        if variant == 'qr':
            q, r = torch.linalg.qr(torch.transpose(c,1,0)) 
            qT = torch.transpose(q,1,0)
            self.comps[pos] = qT.reshape(qT.shape[0],s[1],s[2]) # refolding
            self.comps[pos-1] = torch.einsum('ijk, kl->ijl ', self.comps[pos-1], torch.transpose(r,1,0))

        else: # perform svd
            u, S, vh = torch.linalg.svd(c, full_matrices = False)
            # store orthonormal part at current position
            self.comps[pos] = vh.reshape(vh.shape[0], s[1],s[2])
            self.comps[pos-1] = torch.einsum('ijk, kl->ijl ', self.comps[pos-1], u@torch.diag(S) )

    def set_core(self, mu, variant = 'qr'):
        cc = [] # changes components

        if self.core_position is None:
            assert(variant in ['qr', 'svd'])
            self.core_position = mu
            # from left to right shift of the non-orthogonal component
            for pos in range(0, mu):
                self.__shift_to_right(pos, variant)
            # right to left shift of the non-orthogonal component          
            for pos in range(self.n_comps-1, mu, -1):
                self.__shift_to_left(pos, variant)
            #self.rank[mu+1] = self.comps[mu].shape[2]

            cc= list(range(self.n_comps))

        else:
            while self.core_position > mu:
                cc.append(self.core_position)
                self.shift_core('left')
            while self.core_position < mu:
                cc.append(self.core_position)
                self.shift_core('right')

            cc.append(mu)

        assert(self.comps[-1].shape[2] == 1)#self.comps[0].shape[0] == 1 and 

        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)] 
        return cc
   
    def shift_core(self, direction, variant = 'qr'):
        assert( direction in [-1,1,'left','right'])
        assert(self.core_position is not None)

        if direction == 'left':    shift = -1
        elif direction == 'right': shift = 1
        else:                      shift = direction
        # current core position
        mu = self.core_position
        if shift == 1:
            self.__shift_to_right(mu, variant)
        else:
            self.__shift_to_left(mu, variant)
        
        self.core_position += shift
  
    def dot_rank_one(self, rank1obj):
        """ 
          Implements the multidimensional contraction of the underlying Tensor Train object
          with a rank 1 object being product of vectors of sizes di 
          @param rank1obj: a list of vectors [vi i = 0, ..., modes-1] with len(vi)=di
                           vi is of shape (b,di) with bi > 0
        """
        # the number of vectors must match the component number
        assert(len(rank1obj) == self.n_comps)
        for pos in range(0, self.n_comps):
            # match of inner dimension with respective vector size
            assert(self.comps[pos].shape[1] == rank1obj[pos].shape[1])
            # vectors must be 2d objects 
            assert(len(rank1obj[pos].shape) == 2)
        
        G = [ torch.einsum('ijk, bj->ibk', c, v)  for  c,v in zip(self.comps, rank1obj) ]  
        #print(G)
        res = G[-1]
        # contract from right to left # TODO here we assume row-wise memory allocation of matrices in G
        for pos in range(self.n_comps-2, -1,-1):
            # contraction w.r.t. the 3d coordinate of G[pos]
            #res = lb.dot(G[pos], res)
            res = torch.einsum('ibj, jbk -> ibk', G[pos], res) # k = 1 only
        # res is of shape b x 1
        #print("in rank on dot", res.squeeze(2).shape)
        if res.shape[0]>1:
            return res.squeeze(2).permute(1,0)
        return res.reshape(res.shape[1], res.shape[2])
    

    def rank_truncation(self, max_ranks):

        if self.core_position != 0:
           self.set_core(0)

        for pos in range(self.n_comps-1):
            c = self.comps[pos]
            s = c.shape

            c = c.reshape(s[0]*s[1], s[2])
            u, sigma, vt = torch.linalg.svd(c, full_matrices=False)
            new_rank = max_ranks[pos+1]
            k = u.shape[1]

            # update informations
            u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]

            new_shape = (s[0], s[1], min(new_rank,k))

            self.comps[pos] = u.reshape(new_shape)

            self.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.matmul(torch.diag(sigma),vt), self.comps[pos+1] ) # Stimmt das noch ?

        self.core_position = self.n_comps-1
        assert(self.comps[-1].shape[2] == 1)
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps-1)] + [1] 


    def round(self, delta, verbose = False):

        rank_changed = False

        self.set_core(0)
        rule = Threshold(delta)
        for pos in range(self.n_comps-1):
            c = self.comps[pos]
            s = c.shape
            c = c.reshape(s[0]*s[1], s[2])
            u, sigma, vt = torch.linalg.svd(c, full_matrices=False) 
            new_rank =  rule(u, sigma, vt, pos) 

            # update informations
            u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
            new_shape = (s[0], s[1], min(new_rank,s[2]))
            self.comps[pos] = u.reshape(new_shape)

            ldtype = common_field_dtype([self.comps[pos+1]], [sigma, vt])
            self.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.diag(sigma).type(ldtype) @ vt.type(ldtype), self.comps[pos+1].type(ldtype) ) # Stimmt das noch ?

        self.core_position = self.n_comps-1
        assert(self.comps[-1].shape[2] == 1)#self.comps[0].shape[0] == 1 and 

        if verbose and self.rank is not None:
            for mu, c in enumerate(self.comps[:-1]):
                if self.rank[mu+1] > c.shape[2]: 
                    print(f" {Fore.GREEN} A rank changed : {Style.RESET_ALL}  \
                                {Fore.BLUE} r_{mu} :  {self.rank[mu+1]} -> {c.shape[2]}{Style.RESET_ALL}")
                    rank_changed = True
                    time.sleep(1)  
        
        # update the rank
        self.rank = [self.comps[0].shape[0]] + [self.comps[pos].shape[2] for pos in range(self.n_comps-1)] + [1]
        if verbose:
            print('New rank is ', self.rank)
            # time.sleep(1) 

        return rank_changed

    def modify_ranks(self, rule, verbose= False):
        # TODO handle the case if core is at last position
        if self.core_position != 0:
            self.set_core(0)

        # Possible modify ranks r2, ..., rM-2
        for pos in range(self.n_comps-1):
            c = self.comps[pos]
            s = c.shape
            c= c.reshape(s[0]*s[1], s[2])

            u, sigma, v = torch.linalg.svd(c, full_matrices=False) 
            # obtain the possible new rank according to truncation/retraction rule
            new_rank = rule(u, sigma, v, pos)  
            if verbose:
                print("{c}Update{r} : rank r{p} = {c1}{rank}{r} -> {c}{rankn}{r}".format(p = pos+1, rank=self.rank[pos+1], rankn = new_rank,c1=Fore.RED,c=Fore.GREEN, r=Style.RESET_ALL) )
                print("sing. vals for r{p} :\n".format(p=pos+1), sigma)

            if new_rank > len(sigma):
                #print("C[pos ].shape  = ", self.comps[pos ].shape)
                #print("C[pos +1].shape  = ",self.comps[pos +1 ].shape)

                #   u, sigma, v  = svd ( c )  with c = self.comps[pos]
                k =  new_rank - len(sigma)

                # 1. Add  k new columns to the left unfolding of u :
                #   - leftunfold(u)  is  M x r matrix 
                #   - add  k orthogonal columns called u_k to u to obtain  upk of shape M x ( r  + k )  
                #   - undo the left unfolding w.r.t. M  and store  self.comps[pos] = upk 
                # "add" random vectors from kernel of u^T as orthogonal projection of a random vectors
                u_k = torch.rand(u.shape[0], k)
                u_k -= (u@u.T) @ u_k

                # enlarged u plus k columns
                u_pk = torch.cat([u,u_k], dim = 1)
                self.comps[pos] = u_pk.reshape(s[0],s[1], u_pk.shape[1])    

                # 2. Enlarge the singular values s with k new very small entries.
                s_pk  = torch.cat([sigma, torch.tensor([1e-16]*k)])

                # 3.  K = v * self.comps[pos+1]    w.r.t. 3rd  and right unfolding 
                #      yields a   r x N orthogonal matrix. Add k orthgonal rows K_k
                #     to obtain a (r+k) x N orthogonal matrix K_kp = []
                K = torch.einsum('ir, rkl->ikl ', v, self.comps[pos+1] ) 
                s = K.shape
                K = K.reshape(s[0], s[1]*s[2])

                assert(abs(K.shape[0]-K.shape[1]) >= k)
                # get randomized orthogonal rows
                K_k = torch.rand(k, K.shape[1]) 
                K_k -= K_k @ (K.T@K)
                K_pk = torch.cat([K, K_k])

                # 4. Then undo the unfolding of   K_pk  and scale it with the enlarged sing. values
                #    to define the new right component self.comps[pos+1]
                K_pk = K_pk.reshape(K_pk.shape[0], s[1],s[2])
                self.comps[pos +1 ] = torch.einsum('ij,jkl->ikl', torch.diag(s_pk ), K_pk)

            else : 
                # update informations
                u, sigma, v = u[:,:new_rank], sigma[:new_rank], v[:new_rank,:]

                new_shape = (s[0], s[1], new_rank)
                self.comps[pos] = u.reshape(new_shape)

                self.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.dot(torch.diag(sigma),v), self.comps[pos+1] ) 
            
            # update the rank information
            #self.rank[pos+1] = self.comps[pos].shape[2]

        self.core_position = self.n_comps-1
        assert(self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)
        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)] 

