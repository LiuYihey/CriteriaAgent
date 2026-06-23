import json
from pathlib import Path

summary_path = Path(r'c:\Users\ASUS\Desktop\trial_graph\outputs\bench_profiles_graphs\build_summary.json')
data = json.loads(summary_path.read_text(encoding='utf-8'))

print(f'Total cases: {len(data)}')
print()

nodes_list = [d['nodes'] for d in data]
print('--- Overall stats ---')
print(f'Min nodes : {min(nodes_list)}')
print(f'Max nodes : {max(nodes_list)}')
print(f'Mean nodes: {sum(nodes_list)/len(nodes_list):.1f}')
print(f'Median    : {sorted(nodes_list)[len(nodes_list)//2]}')
print(f'Sum       : {sum(nodes_list)}')
print()

print('--- Distribution by bucket ---')
buckets = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 700), (700, 1000)]
for lo, hi in buckets:
    cnt = sum(1 for n in nodes_list if lo <= n < hi)
    label = f'[{lo:>3}, {hi:>3})'
    print(f'  {label}: {cnt:>3} cases')
print()

print('--- Cases with nodes > 500 (sorted desc) ---')
over = sorted([d for d in data if d['nodes'] > 500], key=lambda x: x['nodes'], reverse=True)
for d in over:
    nct = d['nct_id']
    n = d['nodes']
    e = d['edges']
    print(f'  {nct}: {n} nodes, {e} edges')
print(f'Total: {len(over)} cases')
print()

print('--- Cases with nodes < 50 (sorted asc) ---')
under = sorted([d for d in data if d['nodes'] < 50], key=lambda x: x['nodes'])
for d in under:
    nct = d['nct_id']
    n = d['nodes']
    e = d['edges']
    print(f'  {nct}: {n} nodes, {e} edges')
print(f'Total: {len(under)} cases')
