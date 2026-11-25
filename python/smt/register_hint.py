"""Register-aware topological seed for CP-SAT, never an IR scheduling pass.

Keeps the proposed memory-event order so its periodic LDS packing remains
valid. Computational order is only a search hint; the solver still proves
all dependencies, periodic capacities and register limits independently.
"""
from collections import defaultdict


def register_hint(problem, stages, sequence, budget, reserve, *, policy='depth', order_edges=()):
    info = problem.get('register_liveness')
    if not info or not info.get('supported') or budget is None:
        return None
    n = len(sequence)
    maximum = max(stages)
    # This seed supports data on the steady stage and DMA-only lookahead.
    # The actual constraint supports general stage assignments.
    if any(stages[v['producer']] != maximum or any(
            u['op'] < n and stages[u['op']] != maximum for u in v['users'])
           for v in info['values']):
        return None
    predecessors = [set() for _ in range(n)]
    for e in problem['dependencies']:
        if e['kind'] != 'intra_sync' and stages[e['src']] == stages[e['dst']]:
            predecessors[e['dst']].add(e['src'])
    memory = [i for i in sequence if problem['operations'][i]['kind'] in ('dma', 'wait', 'shm_reg')]
    for a, b in zip(memory, memory[1:]):
        predecessors[b].add(a)
    for a, b in order_edges:
        predecessors[b].add(a)
    successors = [set() for _ in range(n)]
    for b, pred in enumerate(predecessors):
        for a in pred:
            successors[a].add(b)
    born, uses = defaultdict(set), defaultdict(set)
    remaining, words, live = {}, {}, set()
    for key, v in enumerate(info['values']):
        words[key] = v['words']
        born[v['producer']].add(key)
        remaining[key] = {u['op'] for u in v['users'] if u['distance'] == 0}
        if any(u['distance'] for u in v['users']):
            remaining[key].add(n)  # output must survive the backedge
            previous = key + len(info['values'])
            words[previous] = v['words']
            remaining[previous] = {u['op'] for u in v['users'] if u['distance'] == 1}
            live.add(previous)
        for atom in (key, key + len(info['values'])):
            for i in remaining.get(atom, ()):
                uses[i].add(atom)
    count = info['invariant_words'] + sum(words[k] for k in live)
    ranks = {i: rank for rank, i in enumerate(sequence)}
    ready = {i for i, p in enumerate(predecessors) if not p}
    result, emitted = [], {}
    while ready:
        def priority(i):
            new = sum(words[k] for k in born[i])
            freed = sum(words[k] for k in uses[i] if remaining[k] == {i})
            excess = max(0, count + new + reserve - budget)
            recent = max((emitted[j] for j in predecessors[i]), default=-1)
            # Prefer retiring data, then finish a just-started chain. Preserve
            # source rank only as the last tie breaker, not as a constraint.
            delta = new - freed
            # Among growing operations, preferring a 2-word load over a
            # 4-word consumer repeatedly would read the entire matrix first.
            # Treat growth as a class, then continue the most recent chain.
            is_read = problem['operations'][i]['kind'] == 'shm_reg'
            if policy == 'source':
                return excess, min(0, delta), delta > 0, is_read, i
            return excess, min(0, delta), delta > 0, is_read, -recent, ranks[i]
        i = min(ready, key=priority)
        ready.remove(i)
        result.append(i)
        emitted[i] = len(result)
        for k in born[i]:
            live.add(k)
            count += words[k]
        for k in uses[i]:
            remaining[k].discard(i)
            if not remaining[k] and k in live:
                live.remove(k)
                count -= words[k]
        for j in successors[i]:
            if j not in emitted and predecessors[j] <= emitted.keys():
                ready.add(j)
    return result if len(result) == n else None
