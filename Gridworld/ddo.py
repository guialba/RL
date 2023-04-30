import numpy as np

class DDO:
    def __init__(self):
        pass

    def phi(self, h,t, E,H, eta):
        def __phi(t,h, E,H):
            s,_ = E[t]
            if t==0:
                return eta[s, h]
            
            s_,a_ = E[t-1]
            pi, psi = H[h]

            term_1 = sum(phi_table[h_][t-1] * pi_[s_][a_] * psi_[s] for h_,(pi_, psi_) in enumerate(H)) * eta[s, h]
            term_2 = phi_table[h][t-1] * pi[s_][a_] * (1-psi[s])

            return term_1 + term_2 

        phi_table = [{0: __phi(0,h, E,H)} for h,_ in enumerate(H)]

        def prefix(h,t, E,H):
            if t not in phi_table[h]:
                if t-1 not in phi_table[h]:
                    for h_, _ in enumerate(H):
                        phi_table[h_][t-1] = self.phi(h_,t-1, E,H, eta)
                phi_table[h][t] = __phi(t,h, E,H)
            return phi_table[h][t]
        return prefix(h,t, E,H)
    
    def omega(self, h, t, E, H, eta):
        def __omega(t,h, E,H):
            _s,_a = E[-2]
            pi, psi = H[h]
            
            if t==(len(E)-2):
                return pi[_s][_a]
            
            s,a = E[t]
            s_,_ = E[t+1]
            
            term_1 = psi[s_] * sum(eta[s_, h_] * w_table[h_][t+1] for h_,_ in enumerate(H))
            term_2 = (1- psi[s_]) * w_table[h][t+1]

            return pi[s][a] * (term_1 +  term_2)

        w_table = [{(len(E)-2): __omega((len(E)-2),h, E,H)} for h,_ in enumerate(H)]

        def suffix(h, t, E, H):
            if t not in w_table[h]:
                if t+1 not in w_table[h]:
                    for h_, _ in enumerate(H):
                        w_table[h_][t+1] = self.omega(h_,t+1, E,H, eta)
                w_table[h][t] = __omega(t,h, E,H)
            return w_table[h][t]
        return suffix(h, t, E, H)
    
    def posterior(self, t, E,H, eta):
        v = [self.phi(h,t, E,H, eta) * self.omega(h,t, E,H, eta) for h,_ in enumerate(H)]
        return sum(v)
    
    def u(self, h, t, E, H, eta):
        return 1/self.posterior(t, E, H, eta) * self.phi(h,t, E,H, eta) * self.omega(h,t, E,H, eta)
    
    def v(self, h, t, E, H, eta):
        if t == 0:
            return self.u(h, t, E, H, eta)

        s,a = E[t]
        s_,_ = E[t+1]

        somatoria = sum(self.phi(h,t, E,H, eta) * pi[s][a] * psi[s_] for h,(pi, psi) in enumerate(H))

        return 1/self.posterior(t, E, H, eta) * somatoria * eta[s_, h] * self.omega(h,t+1, E,H, eta)
    
    def w(self, h, t, E, H, eta):
        s,a = E[t]
        s_,_ = E[t+1]
        pi, psi = H[h]

        return 1/self.posterior(t, E, H, eta) * self.phi(h,t, E,H, eta) * pi[s][a] * (1-psi[s_]) * self.omega(h,t+1, E,H, eta)
    
    def expectation_gradient(self, E, H, eta):
        derivada_pi = lambda s, a, pi: 1/pi[s][a] * (-1 + 1/len(pi[s])) if a == np.argmax(pi[s]) else 1/len(pi[s]) * 0
        derivada_eta = lambda s, h, eta: 1/eta[s, h] * 0
        derivada_psi = lambda s, psi: 1/psi[s] * 1

        term1 = lambda t,s,a,h,pi:  self.v(h, t, E, H, eta) * derivada_eta(s,h,eta) + self.u(h, t, E, H, eta) * derivada_pi(s,a,pi)
        term2 = lambda t,h,psi:  (self.u(h, t, E, H, eta) - self.w(h, t, E, H, eta)) * derivada_psi(E[t+1][0], psi) + self.w(h, t, E, H, eta) * 1/(1 - derivada_psi(E[t+1][0], psi))

        sum_1 = lambda h, pi: sum(term1(t,s,a,h,pi) for t,(s,a) in enumerate(E) if t < (len(E)-2))
        sum_2 = lambda h, psi: sum(term2(t,h,psi) for t,_ in enumerate(E) if t < (len(E)-2))

        return sum(sum_1(h, pi) + sum_2(h, psi) for h,(pi, psi) in enumerate(H))