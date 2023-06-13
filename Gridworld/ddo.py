import numpy as np
import torch
from torch.autograd import Variable



class DDO:
    def __init__(self, E, eta, pi, psi, generator):
        self.generateFromTheta = generator
        self.theta = [eta, pi, psi]

        self.H, self.eta = self.generateFromTheta(eta, pi, psi)
        self.E = E
        self.p = None

        self.phi_table = [{0: self.__phi(0,h)} for h,_ in enumerate(self.H)]
        self.omega_table = [{(len(self.E)-2): self.__omega((len(self.E)-2),h)} for h,_ in enumerate(self.H)]


    def __phi(self, t,h):
            s,_ = self.E[t]
            if t==0:
                return self.eta[s, h]
            
            s_,a_ = self.E[t-1]
            pi, psi = self.H[h]

            term_1 = sum(self.phi_table[h_][t-1] * pi_[s_][a_] * psi_[s] for h_,(pi_, psi_) in enumerate(self.H)) * self.eta[s, h]
            term_2 = self.phi_table[h][t-1] * pi[s_][a_] * (1-psi[s])

            return term_1 + term_2 
    def phi(self, h,t):
        if t not in self.phi_table[h]:
            if t-1 not in self.phi_table[h]:
                for h_, _ in enumerate(self.H):
                    self.phi_table[h_][t-1] = self.phi(h_,t-1, self.E, self.H, self.eta)
            self.phi_table[h][t] = self.__phi(t,h)
        return self.phi_table[h][t]
    
    def __omega(self, t,h):
            _s,_a = self.E[-2]
            pi, psi = self.H[h]
            
            if t==(len(self.E)-2):
                return pi[_s][_a]
            
            s,a = self.E[t]
            s_,_ = self.E[t+1]
            
            term_1 = psi[s_] * sum(self.eta[s_, h_] * self.omega_table[h_][t+1] for h_,_ in enumerate(self.H))
            term_2 = (1- psi[s_]) * self.omega_table[h][t+1]

            return pi[s][a] * (term_1 +  term_2)
    def omega(self, h, t):
        if t not in self.omega_table[h]:
            if t+1 not in self.omega_table[h]:
                for h_, _ in enumerate(self.H):
                    self.omega_table[h_][t+1] = self.omega(h_,t+1)
            self.omega_table[h][t] = self.__omega(t,h)
        return self.omega_table[h][t]
    
    def posterior(self, t):
        if self.p is not None:
            return self.p
        
        v = [self.phi(h,t) * self.omega(h,t) for h,_ in enumerate(self.H)]
        self.p = sum(v)
        return self.p
    

    def u(self, h, t):
        return 1/self.posterior(t) * self.phi(h,t) * self.omega(h,t)
    
    def v(self, h_, t_):
        if t_ == 0:
            return self.u(h_, t_)

        t = t_ -1 
        s,a = self.E[t]
        s_,_ = self.E[t_]

        somatoria = sum(self.phi(h,t) * pi[s][a] * psi[s_] for h,(pi, psi) in enumerate(self.H))

        return 1/self.posterior(t) * somatoria * self.eta[s_, h_] * self.omega(h_,t_)
    
    def w(self, h, t):
        s,a = self.E[t]
        s_,_ = self.E[t+1]
        pi, psi = self.H[h]

        return 1/self.posterior(t) * self.phi(h,t) * pi[s][a] * (1-psi[s_]) * self.omega(h,t+1)
    
    def expectation_gradient(self, mod=False):
        derivada_pi = lambda s, a, pi: 1/pi[s][a] * ((-1 + 1/len(pi[s])), 1/len(pi[s]))[bool(a == np.argmax(pi[s]))] * 0
        derivada_eta = lambda s, h: 1/self.eta[s, h] * 0
        derivada_psi = lambda s, psi: 1/psi[s] * 1
        derivada_psi_ = lambda s, psi: 1/(1-psi[s]) * -1

        term1 = lambda t,s,a,h,pi:  self.v(h, t) * derivada_eta(s,h) + self.u(h, t) * derivada_pi(s,a,pi)
        term2 = None
        if mod:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)*self.u(h, t)) * derivada_psi(s_, psi) + self.w(h, t) * derivada_psi_(s_, psi)
        else:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)) * derivada_psi(s_, psi) + self.w(h, t) * derivada_psi_(s_, psi)

        sum_1 = lambda h, pi: sum(term1(t,s,a,h,pi) for t,(s,a) in enumerate(self.E) if t < (len(self.E)-1))
        sum_2 = lambda h, psi: sum(term2(t-1,s_,h,psi) for t,(s_,_) in enumerate(self.E) if 0 < t <= (len(self.E)-2))

        return sum(sum_1(h, pi) + sum_2(h, psi) for h,(pi, psi) in enumerate(self.H))
    
    def expectation_gradient2(self,  mod=False):
        eta_s = {
                0: list(set(range(10, 25)) - {14}),
                1: list(range(10))+[14], 
            }

        derivada_pi = lambda s, a, pi: np.array([0, 1/pi[s][a] * (1/len(pi[s]), (-1 + 1/len(pi[s])))[bool(a == np.argmax(pi[s]))], 0])
        derivada_psi = lambda s, psi: np.array([0,0, 1/psi[s] * 1])
        derivada_psi_ = lambda s, psi: np.array([0,0, 1/(1-psi[s]) * -1])
        derivada_eta = lambda s, h: np.array([1/self.eta[s, h] * (-1,1)[s in eta_s[h]], 0,0])

        term1 = lambda t,s,a,h,pi:  self.v(h, t) * derivada_eta(s,h) + self.u(h, t) * derivada_pi(s,a,pi)

        term2 = None
        if mod:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)*self.u(h, t)) * derivada_psi(s_, psi) + self.w(h, t) * derivada_psi_(s_, psi)
        else:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)) * derivada_psi(s_, psi) + self.w(h, t) * derivada_psi_(s_, psi)

        sum_1 = lambda h, pi: np.sum([term1(t,s,a,h,pi) for t,(s,a) in enumerate(self.E) if t < (len(self.E)-1)], axis=0)
        sum_2 = lambda h, psi: np.sum([term2(t-1,s_,h,psi) for t,(s_,_) in enumerate(self.E) if 0 < t <= (len(self.E)-2)], axis=0)

        return np.sum([sum_1(h, pi) + sum_2(h, psi) for h,(pi, psi) in enumerate(self.H)], axis=0)
    
    def expectation_gradient3(self, functions, mod=False):
        func_eta, func_pi, func_psi = functions
        
        def gradientFunc(func, *params):
            theta = Variable(torch.Tensor(self.theta), requires_grad=True)

            valor = torch.log(func(theta, *params))

            valor.backward(retain_graph=True)
            grad = theta.grad.clone().numpy()
            # theta.grad.zero_()  
            return grad
        
        term1 = lambda t,s,a,h,pi:  self.v(h, t) * gradientFunc(func_eta, s, h) + self.u(h, t) * gradientFunc(func_pi, a, np.argmax(pi[s]), len(pi[s]))
        term2 = None
        if mod:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)*self.u(h, t)) * gradientFunc(func_psi) + self.w(h, t) * gradientFunc(lambda *x: 1-func_psi(*x))
        else:
            term2 = lambda t,s_,h,psi:  (self.u(h, t) - self.w(h, t)) * gradientFunc(func_psi) + self.w(h, t) * gradientFunc(lambda *x: 1-func_psi(*x))

        sum_1 = lambda h, pi: np.sum([term1(t,s,a,h,pi) for t,(s,a) in enumerate(self.E) if t < (len(self.E)-1)], axis=0)
        sum_2 = lambda h, psi: np.sum([term2(t-1,s_,h,psi) for t,(s_,_) in enumerate(self.E) if 0 < t <= (len(self.E)-2)], axis=0)

        return np.sum([sum_1(h, pi) + sum_2(h, psi) for h,(pi, psi) in enumerate(self.H)], axis=0)