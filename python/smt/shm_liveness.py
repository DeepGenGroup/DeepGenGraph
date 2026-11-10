"""Schedule-dependent logical SHM lifetimes, independent of estimated cycles.

The timeline is stage * op_count + order. Ends are exclusive, so inputs and
outputs touched by one op coexist. Periodic copies cover all steady-state
overlap. Unknown alias/escape/live-in/live-out facts pin storage for the entire
period. This is a logical capacity check, not an aligned physical allocation.
"""
from __future__ import annotations


def dma_transfers(problem: dict) -> list[dict]:
    """Recover token pairs from completion edges, also for older v1 exports."""
    ops = problem['operations']
    result = []
    for op in ops:
        if op['kind'] != 'dma':
            continue
        waits = sorted({e['dst'] for e in problem['dependencies']
                        if e['src'] == op['id'] and e['kind'] == 'completion'})
        if len(waits) != 1 or ops[waits[0]]['kind'] != 'wait':
            raise ValueError(f"DMA op {op['id']} needs exactly one completion/wait edge")
        if len(set(op['writes'])) != 1:
            raise ValueError(f"DMA op {op['id']} needs one destination buffer")
        result.append(dict(issue=op['id'], wait=waits[0],
                           reads=sorted(set(op['reads'])), writes=sorted(set(op['writes']))))
    return result


def lifetime_specs(problem: dict) -> list[dict]:
    """Require compiler ownership/full-write evidence before allowing reuse."""
    transfers = dma_transfers(problem)
    ops = problem['operations']
    specs = []
    for b in problem['buffers']:
        if b['memory_space'] != 3:
            continue
        info = b.get('liveness', {})
        # rotatable already certifies owned full-tile DMA, no aliases/escape,
        # one writer and one local wait in the original v1 exchange contract.
        local = info.get('iteration_local', b['rotatable'])
        reason = info.get('reason', 'owned full-tile DMA' if local else 'missing ownership/lifetime proof')
        users = set(b['users']) | set(b['readers']) | set(b['writers'])
        users |= {o['id'] for o in ops if b['id'] in o['reads'] + o['writes']}
        # DMA keeps both its source and destination alive through completion,
        # even if no synchronous consumer uses the data afterward.
        for transfer in transfers:
            if b['id'] in transfer['reads'] + transfer['writes']:
                users.update((transfer['issue'], transfer['wait']))
        writers = set(b['writers'])
        if local and (len(writers) != 1 or not users or
                      any(ops[u]['opaque'] for u in users)):
            local, reason = False, 'ambiguous writer or opaque access'
        if local and any(u < next(iter(writers)) for u in users):
            local, reason = False, 'live-in read precedes full write'
        specs.append(dict(id=b['id'], bytes=b['bytes'], local=bool(local), reason=reason,
                          writer=next(iter(writers)) if local else None, users=sorted(users)))
    return specs


def interval_count(specs: list[dict], stage_limit: int) -> int:
    return sum(2 * stage_limit + 1 for b in specs if b['local']) + 1


def analyze_shm(problem: dict, schedule: dict, *, budget: int | None = None,
                specs: list[dict] | None = None) -> dict:
    """Sweep periodic lifetimes and return peak bytes + a peak witness."""
    specs = lifetime_specs(problem) if specs is None else specs
    n = len(problem['operations'])
    ops = sorted(schedule['operations'], key=lambda o: o['id'])
    slots = {b['id']: b['slots'] for b in schedule['buffers']}
    positions = [o['stage'] * n + o['order'] for o in ops]
    radius = max(o['stage'] for o in ops) + 1
    reserved = {b['id'] for b in specs if not b['local'] or slots[b['id']] > 1}
    # The current backend excludes async FIFO allocations from physical reuse.
    # Report their true lifetimes, but reserve every materialized version for
    # the budget: a logical single-live-instance does not erase the second slot.
    pinned = sum(b['bytes'] * slots[b['id']] for b in specs if b['id'] in reserved and b['bytes'] >= 0)
    unknown = [b['id'] for b in specs if b['bytes'] < 0]
    events: dict[int, int] = {0: 0, n: 0}
    records, intervals = [], []
    for b in specs:
        record = dict(b)
        record['slots'] = slots[b['id']]
        record['budget_reserved'] = b['id'] in reserved
        if b['local']:
            begin = positions[b['writer']]
            end = max(positions[u] + 1 for u in b['users'])
            if any(positions[u] < begin for u in b['users']):
                raise ValueError(f"buffer {b['id']} used before its full write")
            # A slot must survive until all users complete before reuse by a
            # later logical iteration, including same-op endpoint overlap.
            if end - begin > slots[b['id']] * n:
                raise ValueError(f"buffer {b['id']} has too few versions for its lifetime")
            record.update(begin=begin, end=end)
            if b['id'] in reserved:
                records.append(record)
                continue
            for iteration in range(-radius, radius + 1):
                left, right = max(0, begin + iteration * n), min(n, end + iteration * n)
                if left >= right:
                    continue
                intervals.append((left, right, b['id'], iteration))
                if b['bytes'] >= 0:
                    events[left] = events.get(left, 0) + b['bytes']
                    events[right] = events.get(right, 0) - b['bytes']
        records.append(record)
    current, peak, at = pinned, pinned, 0
    for position, delta in sorted(events.items()):
        current += delta
        if position < n and current > peak:
            peak, at = current, position
    known = not unknown
    return dict(
        scope='selected-loop-logical-shm', timeline='stage * op_count + order; end exclusive',
        peak_bytes=peak if known else None, known_peak_bytes=peak,
        total_versioned_bytes=sum(b['bytes'] * slots[b['id']] for b in specs) if known else None,
        persistent_bytes=pinned, unknown_size_buffers=unknown, budget_bytes=budget,
        within_budget=(peak <= budget if known else False) if budget is not None else None,
        peak_at_order=at,
        peak_live_instances=[dict(buffer=b, iteration=k) for left, right, b, k in intervals if left <= at < right],
        persistent_buffers=sorted(reserved), buffers=records,
        requires_backend_allocation_check=True,
        note='Logical peak excludes alignment/packing, future scratch allocations and unrelated outer storage; '
             'backend must allocate storage and synchronize cross-warp handoffs.')


def add_shm_constraint(model, specs, stages, ranks, slots, *, n, stage_limit, budget):
    """Exact cumulative capacity for the same conservative logical lifetimes."""
    if any(b['bytes'] < 0 for b in specs):
        raise ValueError('cannot enforce SHM budget with unknown buffer size')
    max_position = stage_limit * n
    intervals, demands, persistent = [], [], []
    for b in specs:
        if not b['local']:
            persistent.append(b['bytes'] * slots[b['id']])
            continue
        writer = b['writer']
        single = model.new_bool_var(f"shm_single_version_{b['id']}")
        model.add(slots[b['id']] == 1).only_enforce_if(single)
        model.add(slots[b['id']] > 1).only_enforce_if(single.Not())
        versioned = model.new_int_var(0, budget, f"shm_reserved_versions_{b['id']}")
        model.add(versioned == 0).only_enforce_if(single)
        model.add(versioned == b['bytes'] * slots[b['id']]).only_enforce_if(single.Not())
        persistent.append(versioned)
        begin = model.new_int_var(0, max_position, f"shm_begin_{b['id']}")
        end = model.new_int_var(1, max_position, f"shm_end_{b['id']}")
        size = model.new_int_var(1, max_position, f"shm_lifetime_{b['id']}")
        model.add(begin == stages[writer] * n + ranks[writer])
        model.add_max_equality(end, [stages[u] * n + ranks[u] + 1 for u in b['users']])
        model.add(end == begin + size)
        model.add(size <= slots[b['id']] * n)
        for u in b['users']:
            model.add(stages[u] * n + ranks[u] >= begin)
        for iteration in range(-stage_limit, stage_limit + 1):
            left = model.new_int_var(-max_position, 2 * max_position, f"shm_left_{b['id']}_{iteration}")
            right = model.new_int_var(-max_position, 2 * max_position, f"shm_right_{b['id']}_{iteration}")
            model.add(left == begin + iteration * n)
            model.add(right == end + iteration * n)
            intervals.append(model.new_optional_interval_var(left, size, right, single,
                                                            f"shm_{b['id']}_{iteration}"))
            demands.append(b['bytes'])
    if persistent:
        fixed = model.new_int_var(0, budget, 'shm_persistent_bytes')
        model.add(fixed == sum(persistent))
        intervals.append(model.new_fixed_size_interval_var(-max_position, 3 * max_position + 1,
                                                          'shm_persistent'))
        demands.append(fixed)
    if intervals:
        model.add_cumulative(intervals, demands, budget)
