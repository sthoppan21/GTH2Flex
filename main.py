import csv
import io
import json
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from core import T, load_prices, build_model, train, simulate, validate, baselines

SYNTHETIC_FILE = 'prices_synthetic.csv'
REAL_FILE = 'prices_real.csv'

def make_circuit(model,theta,p):
    circuit = QuantumCircuit(T)
    circuit.h(range(T))
    for layer in range(p):
        gamma,beta = float(theta[layer]),float(theta[p+layer])
        for i,hi in enumerate(model['h']):
            circuit.rz(2*gamma*hi/model['scale'],i)
        for (i,j),Jij in model['J'].items():
            circuit.rzz(2*gamma*Jij/model['scale'],i,j)
        circuit.rx(2*beta,range(T))
    return circuit

def export(name,mime,content):
    try:
        from js import sendFile
    except ImportError:
        with open(name,'w') as f:
            f.write(content)
    else:
        sendFile(name,mime,content)

def main(shots: int = 4096, p: int = 1, seed: int = 7, use_real_prices: bool = False, switch_usd: float = 0.5):
    if shots < 1:
        raise ValueError('shots must be positive')
    start = time.perf_counter()
    filename = REAL_FILE if use_real_prices else SYNTHETIC_FILE
    rows,prices = load_prices(filename)
    expected = 'real_historical' if use_real_prices else 'synthetic'
    if any(r['source_kind'] != expected for r in rows):
        raise ValueError('Data provenance does not match selected mode.')
    model = build_model(prices,switch_usd=switch_usd)
    baseline_start = time.perf_counter()
    refs = baselines(model,seed)
    baseline_seconds = time.perf_counter()-baseline_start
    training_start = time.perf_counter()
    theta,training = train(model,p,seed)
    training_seconds = time.perf_counter()-training_start
    raw = make_circuit(model,theta,p)
    from qiskit.quantum_info import Statevector
    qiskit_prob = Statevector.from_instruction(raw).probabilities()
    numpy_prob = np.abs(simulate(theta,model))**2
    if not np.allclose(qiskit_prob,numpy_prob,atol=1e-9):
        raise AssertionError('Qiskit and NumPy circuit simulations disagree.')
    measured = raw.copy(); measured.measure_all()
    device = globals().get('backend')
    if device is None:
        from qiskit.providers.basic_provider import BasicSimulator
        device = BasicSimulator()
    # Universal QIS decomposition; keep Qollab's injected backend unchanged.
    compiled = transpile(measured,basis_gates=['rx','ry','rz','cx'],optimization_level=1)
    run_start = time.perf_counter()
    counts = device.run(compiled,shots=shots).result().get_counts()
    run_seconds = time.perf_counter()-run_start
    def decode(key):
        if isinstance(key,(int,np.integer)):
            return int(key)
        text = str(key).replace(' ','')
        return int(text,16) if text.startswith('0x') else int(text,2)
    sampled = {decode(key):float(value) for key,value in counts.items()}
    good = [i for i in sampled if model['feasible'][i]]
    exact = refs['exact']
    optimum = float(model['physical_cost'][exact])
    total = sum(sampled.values())
    result = {'data_source_kind':expected,'source':rows[0]['source'],
              'model':'12-hour fixed-load electrolysis; 3 running hours in each 6-hour delivery block',
              'power_kw':model['power_kw'],'kwh_per_kg':model['kwh_per_kg'],
              'kg_per_on_hour':model['q'],'total_delivery_kg':6*model['q'],
              'tank_capacity_kg':3*model['q'],'terminal_inventory_kg':0,
              'qubits':T,'p':p,'shots_requested':shots,'theta':theta.tolist(),
              'compiled_depth':compiled.depth(),'compiled_gates':dict(compiled.count_ops()),
              'training':training,'training_seconds':training_seconds,
              'reference_selection_seconds':baseline_seconds,'backend_seconds':run_seconds,
              'exact_feasible_schedule_count':int(model['feasible'].sum()),
              'feasible_sample_probability':sum(sampled[i] for i in good)/total,
              'optimal_sample_probability':sum(v for i,v in sampled.items() if model['feasible'][i] and np.isclose(model['physical_cost'][i],optimum))/total,
              'baselines':{},'quantum':None,
              'limitations':['Power, hydrogen demand, tank size and switching charges are scenario assumptions.',
                            'No compression, minimum-up-time, ramp, degradation, emissions or tariff model.',
                            'Parameters were trained with exact classical state simulation.',
                            'This small problem is classically easy; no quantum advantage or customer savings claimed.']}
    for name,i in refs.items():
        valid,_ = validate(model['bits'][i],model)
        result['baselines'][name] = {'schedule':model['bits'][i].astype(int).tolist(),
                                     'valid':valid,'cost_usd':float(model['physical_cost'][i])}
    if good:
        chosen = min(good,key=lambda i:model['physical_cost'][i])
        valid,schedule_rows = validate(model['bits'][chosen],model)
        if not valid:
            raise AssertionError('Quantum schedule failed physical validation.')
        cost = float(model['physical_cost'][chosen])
        fixed_cost = result['baselines']['fixed_alternating']['cost_usd']
        result['quantum'] = {'schedule':model['bits'][chosen].astype(int).tolist(),
                             'valid':valid,'cost_usd':cost,'cost_usd_per_kg':cost/(6*model['q']),
                             'absolute_optimality_gap_usd':cost-optimum,
                             'cost_change_vs_fixed_usd':fixed_cost-cost,
                             'cost_reduction_vs_fixed_pct':100*(fixed_cost-cost)/fixed_cost if fixed_cost>0 else None}
        for t,row in enumerate(schedule_rows):
            row['timestamp_utc'] = rows[t]['timestamp_utc']
            row['price_usd_per_mwh'] = float(prices[t])
        stream = io.StringIO()
        writer = csv.DictWriter(stream,fieldnames=list(schedule_rows[0]))
        writer.writeheader(); writer.writerows(schedule_rows)
        export('schedule.csv','text/csv',stream.getvalue())
    else:
        result['sampling_status'] = 'No feasible sample; no classical solution substituted.'
    result['total_seconds'] = time.perf_counter()-start
    print(json.dumps(result,indent=2))
    export('results.json','application/json',json.dumps(result,indent=2))

if __name__ == '__main__' and 'backend' not in globals():
    main()
