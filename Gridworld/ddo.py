import numpy as np

class DDO:
    def __init__(self, E, H, eta):
        self.E = E
        self.H = H
        self.eta = eta
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
    
    def v(self, h, t):
        if t == 0:
            return self.u(h, t)

        s,a = self.E[t]
        s_,_ = self.E[t+1]

        somatoria = sum(self.phi(h,t) * pi[s][a] * psi[s_] for h,(pi, psi) in enumerate(self.H))

        return 1/self.posterior(t) * somatoria * self.eta[s_, h] * self.omega(h,t+1)
    
    def w(self, h, t):
        s,a = self.E[t]
        s_,_ = self.E[t+1]
        pi, psi = self.H[h]

        return 1/self.posterior(t) * self.phi(h,t) * pi[s][a] * (1-psi[s_]) * self.omega(h,t+1)
    
    def expectation_gradient(self):
        derivada_pi = lambda s, a, pi: 1/pi[s][a] * (-1 + 1/len(pi[s])) if a == np.argmax(pi[s]) else 1/len(pi[s]) * 0
        derivada_eta = lambda s, h: 1/self.eta[s, h] * 0
        derivada_psi = lambda s, psi: 1/psi[s] * 1

        term1 = lambda t,s,a,h,pi:  self.v(h, t) * derivada_eta(s,h) + self.u(h, t) * derivada_pi(s,a,pi)
        term2 = lambda t,h,psi:  (self.u(h, t) - self.w(h, t)) * derivada_psi(self.E[t+1][0], psi) + self.w(h, t) * 1/(1 - derivada_psi(self.E[t+1][0], psi))
        # term2 = lambda t,h,psi:  (self.u(h, t) - self.w(h, t)*self.u(h, t)) * derivada_psi(self.E[t+1][0], psi) + self.w(h, t) * 1/(1 - derivada_psi(self.E[t+1][0], psi))

        sum_1 = lambda h, pi: sum(term1(t,s,a,h,pi) for t,(s,a) in enumerate(self.E) if t < (len(self.E)-2))
        sum_2 = lambda h, psi: sum(term2(t,h,psi) for t,_ in enumerate(self.E) if t < (len(self.E)-2))

        return sum(sum_1(h, pi) + sum_2(h, psi) for h,(pi, psi) in enumerate(self.H))