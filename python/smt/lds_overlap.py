"""Order constraints for bounded LDS read-ahead inside the CP-SAT model.

Timing gaps disappear when a schedule is materialized. These rules instead
put the next read batch before the current batch's consumers, and finish the
current MMACs before opening a third batch. Register and LDS capacities remain
independent hard constraints. This module never rewrites IR or a solved order.
"""
from collections import defaultdict, deque


def _acyclic(n, edges):
    successors = [set() for _ in range(n)]
    degree = [0] * n
    for a, b in edges:
        if b not in successors[a]:
            successors[a].add(b)
            degree[b] += 1
    ready = deque(i for i, d in enumerate(degree) if not d)
    visited = 0
    while ready:
        a = ready.popleft()
        visited += 1
        for b in successors[a]:
            degree[b] -= 1
            if not degree[b]:
                ready.append(b)
    return visited == n


def lds_overlap_rules(problem, enabled=True):
    ops = problem['operations']
    result = dict(enabled=enabled, edges=[], groups=[], skipped_groups=0,
                  contract='bounded-lds-read-ahead-v1')
    if not enabled:
        return result
    # Restrict to explicitly exported, anchored Fragment reads. Older Base
    # exports have neither this instruction shape nor a data liveness model.
    if not problem.get('register_liveness', {}).get('supported'):
        return result
    successors = defaultdict(set)
    original = set()
    for e in problem['dependencies']:
        if e['kind'] != 'intra_sync':
            original.add((e['src'], e['dst']))
        if e['kind'] == 'ssa':
            successors[e['src']].add(e['dst'])

    # A matrix read can feed dynamic vector extracts before reaching MMAC.
    # Include those real consumers in the prefetch boundary as well as MMAC;
    # views alone are not useful independent instructions to hide latency.
    views = {'vector.bitcast', 'vector.extract', 'vector.shape_cast',
             'vector.from_elements', 'frisk.pack_frag', 'frisk.unpack_frag',
             'frisk.extract_frag'}
    def consumers(read):
        pending, seen, first, mmacs = list(successors[read]), set(), set(), set()
        while pending:
            i = pending.pop()
            if i in seen:
                continue
            seen.add(i)
            if ops[i]['kind'] == 'wmma':
                first.add(i)
                mmacs.add(i)
            elif ops[i]['name'] in views:
                if not ops[i].get('zero_issue', False):
                    first.add(i)
                pending.extend(successors[i])
            else:
                # A different use needs a different policy, not an assumed
                # MMAC lifetime or an accidental ordering restriction.
                return None
        if not mmacs or any(not ops[i].get('pipeline_anchor') for i in mmacs):
            return None
        return first, mmacs

    grouped = defaultdict(list)
    for i, op in enumerate(ops):
        if op['kind'] == 'shm_reg' and op.get('pipeline_anchor') and len(op.get('reads', [])) == 1:
            grouped[(op['reads'][0], op['name'])].append(i)
    accepted = set()
    for (buffer, name), reads in grouped.items():
        uses = [consumers(i) for i in reads]
        if any(u is None for u in uses):
            result['skipped_groups'] += 1
            continue
        # Two small fragment reads form a scheduling batch. Keeping both
        # ahead of their consumers also leaves LLVM's ds_read2 combining
        # opportunity open; this is not a claim about address coalescibility.
        width = 2 if name == 'frisk.load_frag' and all(ops[i]['read_bytes'] <= 8 for i in reads) else 1
        batches = [reads[i:i + width] for i in range(0, len(reads), width)]
        if len(batches) < 2:
            continue
        use_map = dict(zip(reads, uses))
        proposed = set(zip(reads, reads[1:]))
        for j, batch in enumerate(batches):
            first = set().union(*(use_map[i][0] for i in batch))
            mmacs = set().union(*(use_map[i][1] for i in batch))
            if j + 1 < len(batches):
                proposed.update((batches[j + 1][-1], u) for u in first)
            if j + 2 < len(batches):
                proposed.update((u, batches[j + 2][0]) for u in mmacs)
        # Address/data dependencies may make a lookahead illegal (e.g. a
        # gather indexed by the current result). Never override those edges.
        if not _acyclic(len(ops), original | accepted | proposed):
            result['skipped_groups'] += 1
            continue
        accepted.update(proposed)
        result['groups'].append(dict(buffer=buffer, reads=reads, batch_size=width,
                                     max_open_batches=2))
    result['edges'] = [list(pair) for pair in sorted(accepted)]
    return result


def check_lds_overlap(rules, schedule):
    entries = {o['id']: o for o in schedule['operations']}
    violations = [[a, b] for a, b in rules['edges']
                  if entries[a]['stage'] != entries[b]['stage'] or
                  entries[a]['order'] >= entries[b]['order']]
    return dict(contract=rules['contract'], enabled=rules['enabled'],
                groups=len(rules['groups']), constraints=len(rules['edges']),
                within_rules=not violations, violations=violations)
