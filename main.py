
#Author: Julien Bouachrine
#Date: 08/05/26
# %%
# 
import numpy as np
import matplotlib.pyplot as plt

# defining parameters
T = 10
N = 10000
h = T / N
M= 100


increments = np.random.normal(0, np.sqrt(h), size=(M,N))
bm = np.insert(np.cumsum(increments,axis=1), 0, 0,axis=1)


t = np.linspace(0, T, N + 1)

# plotting
plt.figure(figsize=(10, 5))
plt.plot(t, bm.T, lw=0.8)
plt.title(f"standard brownian under Q($T={T}$, $N={N}$)")
plt.xlabel("time $t$")
plt.ylabel("$B_t$")
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', lw=1)
plt.legend()
#plt.show()
# %%

# %%
#Price of the risky asset 
S0=10
r=.02
vol=.3

S = S0*np.exp((r-vol**2/2)*t+vol*bm)
plt.plot(t,S.T)
# %%
# %%
#Price of a call
K=20
CT= np.maximum(S[:,-1] - K,0)
C0 = np.exp(-r*T)*np.mean(CT)
print(f"Premium: {C0}")
CT_at_random= np.random.choice(CT,size=1)[0]
print(f"1 kind of Payoff at maturity: {CT_at_random}")

nb_trajectoires_positives = np.count_nonzero(CT > 0)

proba_exercice = np.mean(CT > 0) * 100

print(f"Risk neutral probability to exercise the option at maturity : {proba_exercice:.2f}%")


#TODO after exams: 1)define a 'pricer' class to make things cleaner 2) compute the greeks (in progress, to do after exams) 3) compare with the following approach: numerically solve B-S PDE (using Euler formula imo). See what happens in terms of variance reduction and computational cost when the dimension is large. Apply pricer to compute straddle, butterfly and other vanilla derivatives. Also, extend to the case of more complexe  derivatives using implied vol. 4) Find a way to implement American options with Snell's envelope and Bellman approach. Experiment with sabr model for swaptions 
