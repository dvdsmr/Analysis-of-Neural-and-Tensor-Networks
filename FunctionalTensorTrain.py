
import torch
torch.set_default_dtype(torch.float64)


from colorama import Fore, Style
from copy import deepcopy
from TensorTrain import TensorTrain
from math import pi, sqrt



class FourierBasisH1(object):   

    # TODO: allow different domains in different dimensions
    def __init__(self, maxIndices,domain):
        """
            Modified to permit now lists of domains for the different dimensions,
            as well as lists of regularization norms for the different dimensions.
            this means 'domain' should now either be a tuple [float,float] or a list
            [[float,float],...,[float,float]] of tuples. 'norm' should be either a string or 
            a list of strings.
        """
        # assert norm in ['L2','H1','H2']
        # assert (len(domain) == 2)

        self.d = len(maxIndices)
        self.maxIndices = maxIndices
        self.degs = [2*maxIndex for maxIndex in maxIndices]
        self.domain = domain
        
        self.T_list = [dom[1]-dom[0] for dom in self.domain]
        
        # for our TTHJB operator
        self.gradTmatrix_list = [torch.zeros((self.degs[i]+1,self.degs[i]+1)) for i in range(self.d)]
        for i in range(self.d):
            for k in range(self.maxIndices[i]):
                # diff sin getting cos
                self.gradTmatrix_list[i][1+k][1+self.maxIndices[i]+k] = 2*k*pi/self.T_list[i]
                # diff cos getting sin
                self.gradTmatrix_list[i][1+self.maxIndices[i]+k][1+k] = -2*k*pi/self.T_list[i]
        self.gradTmatrix_list = [mat.T for mat in self.gradTmatrix_list]
        
    # for the basis
    def sinBasis(self,k,dim,x):
        """returns the sine basis functions evaluated at input x

        Args:
            k (int): level up to which we compute the sine terms
            dim (int): component of the input to which x belongs
            x (torch.tensor): input of shape (batchsize,) which is a collection of the dim-th components of all "true" inputs

        Returns:
            sinResult (torch.tensor): shape (batch_size, k). Evaluations of the sine basis up to level k at all inputs
        """
        assert k >= 1
        sinResult = sqrt(2./self.T_list[dim])/sqrt(1+8*pi**2/self.T_list[dim]**3)*torch.sin(2*1*pi/self.T_list[dim] * x)[:,None]
        for j in range(2,k+1):
            sinResult = torch.cat([sinResult, sqrt(2./self.T_list[dim])/sqrt(1+8*j**2*pi**2/self.T_list[dim]**3)*torch.sin(2*j*pi/self.T_list[dim] * x)[:,None]],1)
        return sinResult
    
    def cosBasis(self,k,dim,x):
        """same as sinBasis"""
        assert k >= 1
        cosResult = sqrt(2./self.T_list[dim])/sqrt(1+8*pi**2/self.T_list[dim]**3)*torch.cos(2*1*pi/self.T_list[dim] * x)[:,None]
        for j in range(2,k+1):
            cosResult = torch.cat([cosResult, sqrt(2./self.T_list[dim])/sqrt(1+8*j**2*pi**2/self.T_list[dim]**3)*torch.cos(2*j*pi/self.T_list[dim] * x)[:,None]],1)
        return cosResult
    
    def constBasis(self,dim,x):
        return 1./sqrt(self.T_list[dim])*torch.ones((x.shape[0],1))
        
    def completeBasis(self,k,dim,x):
        # print(self.constBasis(dim,x).shape,self.sinBasis(k,dim,x).shape,self.cosBasis(k,dim,x).shape)
        return torch.cat([self.constBasis(dim,x),self.sinBasis(k,dim,x),self.cosBasis(k,dim,x)],1)
    
    # for the derivative
    def dsinBasis(self,k,dim,x):
        """returns the sine basis functions evaluated at input x

        Args:
            k (int): level up to which we compute the sine terms
            dim (int): component of the input to which x belongs
            x (torch.tensor): input of shape (batchsize,) which is a collection of the dim-th components of all "true" inputs

        Returns:
            sinResult (torch.tensor): shape (batch_size, k). Evaluations of the sine basis up to level k at all inputs
        """
        assert k >= 1
        sinResult = 2*1*pi/self.T_list[dim]*sqrt(2./self.T_list[dim])/sqrt(1+8*pi**2/self.T_list[dim]**3)*torch.cos(2*1*pi/self.T_list[dim] * x)[:,None]
        for j in range(2,k+1):
            sinResult = torch.cat([sinResult, 2*j*pi/self.T_list[dim] * sqrt(2./self.T_list[dim])/sqrt(1+8*j**2*pi**2/self.T_list[dim]**3)*torch.cos(2*j*pi/self.T_list[dim] * x)[:,None]],1)
        return sinResult
    
    def dcosBasis(self,k,dim,x):
        """same as sinBasis"""
        assert k >= 1
        cosResult = - 2*1*pi/self.T_list[dim] * sqrt(2./self.T_list[dim])/sqrt(1+8*pi**2/self.T_list[dim]**3)*torch.sin(2*1*pi/self.T_list[dim] * x)[:,None]
        for j in range(2,k+1):
            cosResult = torch.cat([cosResult, - 2*j*pi/self.T_list[dim] * sqrt(2./self.T_list[dim])/sqrt(1+8*j**2*pi**2/self.T_list[dim]**3)*torch.sin(2*j*pi/self.T_list[dim] * x)[:,None]],1)
        return cosResult
    
    def dconstBasis(self,dim,x):
        return 0.*torch.ones((x.shape[0],1))
        
    def dcompleteBasis(self,k,dim,x):
        return torch.cat([self.dconstBasis(dim,x),self.dsinBasis(k,dim,x),self.dcosBasis(k,dim,x)],1)
    

    def __call__(self, x):
        """lifts the inputs to feature space.

        Parameters
        ----------
        x : torch.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of torch.tensor
            inputs lifted to feature space defined by the feature and
            basis_coeffs attributes. 
            Query [i][j,k] is the k-th basis function evaluated at the j-th sample's
            i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []
        for dim in range(self.d):
            embedded_data.append(self.completeBasis(self.maxIndices[dim],dim,x[:,dim]))
        return embedded_data

    def grad(self, x):
        """lifts the inputs to feature-derivative space.

        Parameters
        ----------
        input_data : torch.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of torch.tensor
            inputs lifted to feature-derivative space defined by the feature and
            grad_coeffs attributes. 
            Query Query [i][j,k] is the first derivative of the k-th basis function evaluated 
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []
        for dim in range(self.d):
            embedded_data.append(self.dcompleteBasis(self.maxIndices[dim],dim,x[:,dim]))
        return embedded_data


class Extended_TensorTrain(object):

    def __init__(self, tfeatures, ranks, comps=None):
        """
            tfeatures should be a function returning evaluations of feature functions if given a data batch as argument,
            i.e. tfeatures(x), where x is an torch.array of size (batch_size, n_comps),
            is a list of torch.arrays such that tfeatures(x)[i][j,k] is the k-th feature function (in that dimension) 
            evaluated at the j-th samples i-th component
        """

        self.tfeatures = tfeatures
        self.d = self.tfeatures.d

        assert(len(ranks) == self.tfeatures.d+1)
        self.rank = ranks

        self.tt = TensorTrain([deg+1 for deg in tfeatures.degs])
        if comps is None:
            self.tt.fill_random(ranks,1.)
        else:
            # TODO allow ranks len d+1
            for pos in range(self.tfeatures.d-1):
                assert(comps[pos].shape[2] == ranks[pos+1])
            self.tt.set_components(comps)
        self.tt.rank = self.rank

    def __call__(self, x):
        assert(x.shape[1] == self.d)
        u = self.tfeatures(x)
        return self.tt.dot_rank_one(u)


    def set_ranks(self, ranks):
        self.tt.retract(self, ranks, verbose=False)


    def fit_ALS(self, x, y, iterations, rule = None, tol = 8e-6, verboselevel = 0, reg_param=None, xVal=None, yVal=None):
        """
            Fits the Extended Tensortrain to the given data (x,y) of some target function 
                     f : K\\subset IR^d to IR^m 
                                     x -> f(x) = y.

            @param x : input parameter of the training data set : x with shape (b,d)   b \\in \\mathbb{N}
            @param y : output data with shape (b,m)
        """

        # assert(x.shape[1] == self.d)
        solver = self.ALS_Regression
    
        residual = solver(x,y,iterations,tol,verboselevel, rule, reg_param, xVal, yVal)
        # self.tt.set_components(res.comps)
        return residual

    def ALS_Regression(self, x, y, iterations, tol, verboselevel, rule = None, reg_param=None, xVal=None, yVal=None):
        
        """
            @param loc_solver : 'normal', 'least_square',  
            x shape (batch_size, input_dim)
            y shape (batch_size, 1)
        """
        residuals = []
        # size of the data batch
        b = y.shape[0]

        # feature evaluation on input data
        u = self.tfeatures(x)

        # 0 - orthogonalize, s.t. sweeping starts on first component
        self.tt.set_core(mu = 0)

        # TODO: name stack instead of list
        # initialize lists for left and right contractions
        R_stack = [torch.ones((b, 1))]
        L_stack = [torch.ones((b, 1))]

        d = self.tt.n_comps

        def add_contraction(mu, list, side='left'):

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))

            core_tensor = self.tt.comps[mu]
            data_tensor = u[mu]
            contracted_core = torch.einsum('idr, bd -> bir', core_tensor, data_tensor)
            if (side == 'left' or side == -1):
                list.append(torch.einsum('bir, bi -> br', contracted_core, list[-1]))
            else: 
                list.append(torch.einsum('bir, br -> bi', contracted_core, list[-1]))


        def solve_local(mu,L,R):

            A = torch.einsum('bi,bj,br->bijr', L, u[mu], R)
            A = A.reshape(A.shape[0], A.shape[1]*A.shape[2]*A.shape[3])

            if reg_param is not None:
                assert isinstance(reg_param,float)

            

            # c, res, rank, sigma = torch.linalg.lstsq(A, y, rcond = None)  
            ATA, ATy = A.T@A, A.T@y

            if reg_param is not None:
                assert isinstance(reg_param,float)
                ATA += reg_param * torch.eye(ATA.shape[0])

            c = torch.linalg.solve(ATA,ATy)

            rel_err = torch.linalg.norm(A@c - y)/torch.linalg.norm(y)
            if rel_err > 1e-4:
                if reg_param is not None:
                    Ahat = torch.cat([A,sqrt(reg_param)*torch.eye(A.shape[1])],0)
                    yhat = torch.cat([y,torch.zeros((A.shape[1],1))],0)
                    c, res, rank, sigma = torch.linalg.lstsq(Ahat, yhat, rcond = None) 
                else:
                    c, res, rank, sigma = torch.linalg.lstsq(A, y, rcond = None)  

            s = self.tt.comps[mu].shape
            self.tt.comps[mu] = c.reshape(s[0],s[1],s[2])


        # initialize residual
        #TODO rename to rel res
        if xVal is None:
            curr_res = (torch.mean(torch.linalg.norm(self(x) - y,dim=1))/torch.mean(torch.linalg.norm(y,dim=1))).item()
        else:
            curr_res = (torch.mean(torch.linalg.norm(self(xVal) - yVal,dim=1)/torch.mean(torch.linalg.norm(yVal,dim=1)))).item()
        if verboselevel > 0: print("START relative error : ", curr_res)

        # initialize stop condition
        niter = 0
        stop_condition = niter > iterations or curr_res < tol

        # loc_solver =  solve_local_iterativeCG
        loc_solver = solve_local

        # before the first forward sweep we need to build the list of right contractions
        for mu in range(d-1,0,-1):
            add_contraction(mu, R_stack, side='right')

        history = []

        while not stop_condition:
            # forward half-sweep
            for mu in range(d-1):
                self.tt.set_core(mu)
                if mu > 0:
                    add_contraction(mu-1, L_stack, side='left')
                    del R_stack[-1]
                loc_solver(mu,L_stack[-1],R_stack[-1])

            # before back sweep
            self.tt.set_core(d-1)
            add_contraction(d-2, L_stack, side='left')
            del R_stack[-1]

            # backward half sweep
            for mu in range(d-1,0,-1):
                self.tt.set_core(mu)
                if mu < d-1:
                    add_contraction(mu+1, R_stack, side='right')
                    del L_stack[-1]
                loc_solver(mu,L_stack[-1],R_stack[-1])


            # before forward sweep
            self.tt.set_core(0)
            add_contraction(1, R_stack, side='right')
            del L_stack[-1]


            # update stop condition
            niter += 1
            if xVal is None:
                curr_res = (torch.mean(torch.linalg.norm(self(x) - y,dim=1))/torch.mean(torch.linalg.norm(y,dim=1))).item()
            else:
                curr_res = (torch.mean(torch.linalg.norm(self(xVal) - yVal,dim=1)/torch.mean(torch.linalg.norm(yVal,dim=1)))).item()
            # update reg_param
            # reg_param = reg_param*max(curr_res.item(),0.9)
            stop_condition = niter > iterations or  curr_res < tol
            if verboselevel > 0: # and  niter % 10 == 0: 
                print("{c}{k:<5}. iteration. {r} Relative error : {c2}{res}{r}".format(c=Fore.GREEN, c2=Fore.RED, r=Style.RESET_ALL, k = niter, res = curr_res))
                
        return residuals