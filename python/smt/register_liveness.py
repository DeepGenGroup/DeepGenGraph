"""Periodic Fragment data pressure, in 32-bit words per thread.

This is a scheduling constraint, not a physical register allocation guarantee.
The compiler exports deduplicated origin slices and loop-carried uses. Address
and backend temporaries are covered only by an explicit reserve; final assembly
must independently pass check_assembly_resources.py.
"""


def validate_register_liveness(problem):
    info = problem.get('register_liveness')
    if not info or info.get('contract') not in ('fragment-origin-words-v1', 'fragment-origin-words-v2') or not info.get('supported'):
        raise ValueError('register budget requires supported Fragment register_liveness export')
    n = len(problem['operations'])
    if type(info['invariant_words']) is not int or info['invariant_words'] < 0:
        raise ValueError('invalid invariant register words')
    for value in info['values']:
        if (type(value['producer']) is not int or not 0 <= value['producer'] < n or
                type(value['words']) is not int or value['words'] <= 0 or not value['users']):
            raise ValueError('invalid register lifetime')
        for user in value['users']:
            if (type(user['op']) is not int or not 0 <= user['op'] <= n or
                    type(user['distance']) is not int or user['distance'] not in (0, 1)):
                raise ValueError('invalid register lifetime user')
    return info


def add_register_constraint(model, problem, stages, ranks, *, stride, stage_limit,
                            budget, reserve):
    info = validate_register_liveness(problem)
    n = len(stages)
    capacity = budget - reserve - info['invariant_words']
    if capacity < 0:
        raise ValueError('register invariants and reserve exceed VGPR budget')
    # Double rank positions: dying inputs and freshly defined outputs coexist
    # at an instruction, while yield lies just BEFORE the next iteration.
    period = 2 * stride
    positions = [stage * period + 2 * rank for stage, rank in zip(stages, ranks)]
    maximum = model.new_int_var(0, stage_limit - 1, 'register_max_stage')
    model.add_max_equality(maximum, stages)
    intervals, demands = [], []
    radius = stage_limit + 1
    bound = (stage_limit + 2) * period
    for i, value in enumerate(info['values']):
        begin = model.new_int_var(0, bound, f'reg_begin_{i}')
        model.add(begin == positions[value['producer']])
        ends = [(maximum + 1) * period if u['op'] == n else
                positions[u['op']] + u['distance'] * period + 1 for u in value['users']]
        end = model.new_int_var(0, bound, f'reg_end_{i}')
        model.add_max_equality(end, ends)
        size = model.new_int_var(1, bound, f'reg_size_{i}')
        model.add(size == end - begin)
        for copy in range(-radius, radius + 1):
            intervals.append(model.new_interval_var(begin + copy * period, size,
                             end + copy * period, f'reg_{i}_{copy}'))
            demands.append(value['words'])
    if intervals:
        model.add_cumulative(intervals, demands, capacity)


def analyze_registers(problem, schedule, *, budget=250, reserve=32):
    """Independent sweep over the emitted order, not the solver's phase gaps."""
    info = validate_register_liveness(problem)
    ops = sorted(schedule['operations'], key=lambda o: o['id'])
    n, maximum = len(ops), max(o['stage'] for o in ops)
    period = 2 * n
    positions = [o['stage'] * period + 2 * o['order'] for o in ops]
    events = {0: 0, period: 0}
    radius = maximum + 2
    for value in info['values']:
        begin = positions[value['producer']]
        end = max((maximum + 1) * period if u['op'] == n else
                  positions[u['op']] + u['distance'] * period + 1 for u in value['users'])
        if end <= begin:
            raise ValueError('register use precedes definition')
        for copy in range(-radius, radius + 1):
            left, right = max(0, begin + copy * period), min(period, end + copy * period)
            if left < right:
                events[left] = events.get(left, 0) + value['words']
                events[right] = events.get(right, 0) - value['words']
    live = peak = info['invariant_words']
    witness = 0
    for position, delta in sorted(events.items()):
        live += delta
        if position < period and live > peak:
            peak, witness = live, position // 2
    return dict(data_peak_words=peak, reserve_words=reserve, estimated_peak_words=peak + reserve,
                budget_words=budget, within_budget=peak + reserve <= budget,
                peak_order=witness, invariant_words=info['invariant_words'],
                accounting_contract=info['contract'],
                physical_registers_verified=False)
