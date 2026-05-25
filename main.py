
#Author: Julien Bouachrine
#Last Update: 25/05/26

### In progress
# %%
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# defining parameters
T = 10 #Number of periods
N = 5000 #number of timesteps
h = T / N #step
M= 5000 #number of trajectories


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
#Pricing a european call (warm-up)
K=20
CT= np.maximum(S[:,-1] - K,0)
C0 = np.exp(-r*T)*np.mean(CT)


print(f"Premium: {C0}")
CT_at_random= np.random.choice(CT,size=1)[0]
print(f"1 kind of Payoff at maturity: {CT_at_random}")

nb_trajectoires_positives = np.count_nonzero(CT > 0)

proba_exercice = np.mean(CT > 0) * 100

print(f"probability (under Q) to exercise the option at maturity : {proba_exercice:.2f}%")


#Pricing an Asian Option
#Naive approach by empirical mean
def PricerAsianNaive():
    AT = np.maximum(np.mean(S,axis=1) - K,0)
    A0 = np.exp(-r*T)*np.mean(AT)
    return(A0)

A0=PricerAsianNaive()
print(f"[Naive] The price of the Asian Option with strike {K} is {np.round(A0,3)}")

#Control variate method
#We will use the geometric asian call option price since its correlated to the (arithmetic) asian option price.
#h0 is the payoff of the geometric asian call option
#its expectation , m, is the future value of the price of such an option at time 0; we have a closed formula for this.

def d1(t,x,sigma, r, K, T):
    return((np.log(x/K)+r+(sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t)))

def d2(t,x, sigma, r, K , T):
    return(d1(t,x,sigma,r,K,T) - sigma*np.sqrt(T-t))

def PricerAsianCV():
    vol_star = vol/np.sqrt(3)
    r_star = 1/2 * (r-vol**2/6)
    m= S0*np.exp(r_star*T)*norm.cdf(d1(0,S0,vol_star,r_star,K,T))-K*norm.cdf(d2(0,S0,vol_star,r_star,K,T))
    h = np.maximum(np.mean(S,axis=1) - K,0)
    h0= np.maximum(np.exp(np.mean(np.log(S),axis=1))-K,0)
    b=np.cov(h0,h)[0,1]/np.var(h0)
    A0 = np.exp(-r*T)*np.mean(h-b*(h0-m))
    return(A0)

A0=PricerAsianCV()
print(f"[CV] The price of the Asian Option with strike {K} is {np.round(A0,3)}")



#TODO after exams: 1)define a 'pricer' class to make things cleaner 2) compute the greeks (in progress, to do after exams) 3) compare with the following approach: numerically solve B-S PDE (using Euler formula imo). See what happens in terms of variance reduction and computational cost when the dimension is large. Apply pricer to compute straddle, butterfly and other vanilla derivatives. Also, extend to the case of more complexe  derivatives using implied vol. 4) Find a way to implement American options with Snell's envelope and Bellman approach. Experiment with sabr model for swaptions 
