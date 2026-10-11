import csv
import numpy as np
from scipy.optimize import minimize

T = 12
BLOCK = 6
RUN_HOURS_PER_BLOCK = 3

def load_prices(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != T:
        raise ValueError('Exactly 12 consecutive hourly prices are required.')
    price = np.array([float(r['price_usd_per_mwh']) for r in rows])
    if not np.isfinite(price).all():
        raise ValueError('Prices must be finite; negative prices are permitted.')
    if [int(r['slot']) for r in rows] != list(range(T)):
        raise ValueError('Slots must be 0 through 11 in order.')
    return rows, price

def build_model(price, power_kw=1000., kwh_per_kg=57.5, switch_usd=.5):
    if power_kw <= 0 or kwh_per_kg <= 0 or switch_usd < 0:
        raise ValueError('Power and consumption must be positive; switching cost nonnegative.')
    price = np.asarray(price,dtype=float)
    if price.shape != (T,) or not np.isfinite(price).all():
        raise ValueError('Model needs 12 finite hourly prices.')
    costs = price*power_kw/1000.
    q = power_kw/kwh_per_kg
    penalty = float(np.abs(costs).sum()+switch_usd*(T+1)+1.)
    linear = costs.copy()
    pair = {}
    constant = 0.
    for start in (0,6):
        indices = range(start,start+6)
        constant += penalty*RUN_HOURS_PER_BLOCK**2
        for i in indices:
            linear[i] += penalty*(1-2*RUN_HOURS_PER_BLOCK)
        for i in indices:
            for j in range(i+1,start+6):
                pair[i,j] = pair.get((i,j),0.)+2*penalty
    linear[0] += switch_usd
    linear[-1] += switch_usd
    for t in range(1,T):
        linear[t-1] += switch_usd
        linear[t] += switch_usd
        pair[t-1,t] = pair.get((t-1,t),0.)-2*switch_usd
    ids = np.arange(2**T)
    bits = ((ids[:,None] >> np.arange(T)) & 1).astype(float)
    energy = constant+bits@linear
    for (i,j), value in pair.items():
        energy += value*bits[:,i]*bits[:,j]
    feasible = (bits[:,:6].sum(axis=1)==3)&(bits[:,6:].sum(axis=1)==3)
    physical_cost = bits@costs+switch_usd*(bits[:,0]+bits[:,-1]+np.abs(np.diff(bits,axis=1)).sum(axis=1))
    assert np.allclose(energy[feasible],physical_cost[feasible])
    h = -linear/2
    J = {}
    for (i,j), value in pair.items():
        h[i] -= value/4
        h[j] -= value/4
        J[i,j] = value/4
    offset = constant+linear.sum()/2+sum(pair.values())/4
    spins = 1-2*bits
    reconstructed = offset+spins@h
    for (i,j), value in J.items():
        reconstructed += value*spins[:,i]*spins[:,j]
    assert np.allclose(reconstructed,energy)
    scale = max(float(np.ptp(energy)),1.)
    return dict(price=price,costs=costs,q=q,power_kw=power_kw,kwh_per_kg=kwh_per_kg,
                switch_usd=switch_usd,penalty=penalty,linear=linear,pair=pair,
                constant=constant,bits=bits,energy=energy,feasible=feasible,
                physical_cost=physical_cost,h=h,J=J,offset=offset,scale=scale)

def simulate(theta,model):
    theta = np.asarray(theta)
    p = len(theta)//2
    psi = np.ones(2**T,dtype=complex)/np.sqrt(2**T)
    centered = (model['energy']-model['energy'].min())/model['scale']
    for layer in range(p):
        gamma,beta = theta[layer],theta[p+layer]
        psi *= np.exp(-1j*gamma*centered)
        for q in range(T):
            blocks = psi.reshape(-1,2,2**q)
            left,right = blocks[:,0,:].copy(),blocks[:,1,:].copy()
            blocks[:,0,:] = np.cos(beta)*left-1j*np.sin(beta)*right
            blocks[:,1,:] = np.cos(beta)*right-1j*np.sin(beta)*left
    return psi

def train(model,p=1,seed=7,maxfev=160):
    if p not in (1,2,3):
        raise ValueError('p must be 1, 2, or 3')
    rng = np.random.default_rng(seed)
    scaled = (model['energy']-model['energy'].min())/model['scale']
    def loss(theta):
        return float(np.abs(simulate(theta,model))**2@scaled)
    starts = [np.r_[rng.uniform(1,25,p),rng.uniform(.1,1.2,p)] for _ in range(3)]
    fits = [minimize(loss,s,method='Powell',options={'maxfev':maxfev,'maxiter':30}) for s in starts]
    best = min(fits,key=lambda f:f.fun)
    return best.x,dict(evaluations=sum(f.nfev for f in fits),scaled_expectation=float(best.fun),
                      optimizer_converged=bool(best.success))

def validate(schedule,model):
    x = np.asarray(schedule,dtype=int)
    inventory = 0.
    capacity = 3*model['q']
    records = []
    valid = True
    for t in range(T):
        produced = model['q']*x[t]
        before_delivery = inventory+produced
        delivered = 3*model['q'] if t in (5,11) else 0.
        inventory = before_delivery-delivered
        valid &= before_delivery <= capacity+1e-8 and inventory >= -1e-8
        records.append(dict(slot=t,on=int(x[t]),power_kw=float(model['power_kw']*x[t]),
                            hydrogen_produced_kg=float(produced),delivery_kg=float(delivered),
                            inventory_before_delivery_kg=float(before_delivery),
                            inventory_after_delivery_kg=float(inventory),
                            electricity_cost_usd=float(model['costs'][t]*x[t])))
    valid &= abs(inventory)<1e-8
    return bool(valid),records

def encode(x):
    return int(np.dot(np.asarray(x,dtype=int),2**np.arange(T)))

def baselines(model,seed=7):
    bits,feasible,cost = model['bits'],model['feasible'],model['physical_cost']
    valid_ids = np.flatnonzero(feasible)
    exact = int(valid_ids[np.argmin(cost[valid_ids])])
    fixed = np.zeros(T,dtype=int); fixed[[0,2,4,6,8,10]]=1
    greedy = np.zeros(T,dtype=int)
    for start in (0,6):
        greedy[start+np.argsort(model['costs'][start:start+6])[:3]]=1
    rng = np.random.default_rng(seed)
    random_id = int(rng.choice(valid_ids))
    return {'exact':exact,'fixed_alternating':encode(fixed),'price_greedy':encode(greedy),
            'random_feasible':random_id}
