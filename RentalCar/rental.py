import numpy as np
from scipy.stats import poisson

# Define the parameters for the problem
max_cars = 20
max_move = 5
rental_reward = 10
move_cost = 2
gamma = 0.9

# Define the transition probabilities for the demand at each location
demand = [3, 4]
dropoff = [3, 2]

S = [(s1,s2) for s1 in range(max_cars+1) for s2 in range(max_cars+1)]
A = [a for a in range(-max_move, max_move+1)]

# Define the value function and policy arrays
V = np.zeros((max_cars + 1, max_cars + 1))
Pi = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
P = np.zeros((len(S), len(S), len(A)))
R = np.zeros((len(S), len(S), len(A)))


def p(s, s_, a):
    temp_s = np.array(s) + np.array([a, -a])
    p_1, r_1, p_2, r_2 = 0,0,0,0
    
    for requests_1, requests_2 in zip(range(temp_s[0]+1), range(temp_s[1]+1)):
        diff = temp_s[0] - s_[0]
        if requests_1 >= diff:
            p_1 += poisson.pmf(requests_1, demand[0]) * poisson.pmf(requests_1-diff, dropoff[0])
            r_1 += p_1*requests_1*10
    
        diff = temp_s[1] - s_[1]
        if requests_2 >= diff:
            p_2 += poisson.pmf(requests_2, demand[1]) * poisson.pmf(requests_2-diff, dropoff[1])
            r_2 += p_2*requests_2*10

    return p_1 * p_2, r_1+r_2


def policy_eval(theta=1e-4):
    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = sum(P[s, s_, Pi[s]]*R[s, s_, Pi[s]]+ P[s, s_, Pi[s]]*gamma*V[s_] for s_ in S])
            delta = max(delta, abs(v-V[s]))
        if delta < theta: break

def policy_improve():
    stable = True
    for s in S:
        old = Pi[s]
        Pi[s] = np.argmax([sum(P[s, s_, a]*R[s, s_, a]+ P[s, s_, a]*gamma*V[s_] for s_ in S) for a in A])
        if old != Pi[s]: stable = False
    if stable: return V, Pi 
    else: policy_eval()

for si,s in enumerate(S):
    for si_,s_ in enumerate(S):
        for ai,a in enumerate(A):
            P[si,si_,ai], R[si,si_,ai] = p(s, s_, a)


# print(policy_improve())