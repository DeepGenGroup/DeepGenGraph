"""Finite-horizon objective for the periodic schedule, in relative model ticks.

This covers the selected loop's real tasks and their completion latencies.
It is not a model of the whole GPU launch, or of unexported lowering work.
"""


def resolve_trip_count(problem, upper_bound=None, *, required=True):
    loop = problem['loop']
    bounds = []
    for source, value in (('explicit_upper_bound', upper_bound),
                          ('exported_upper_bound', loop.get('trip_count_upper_bound'))):
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError('trip count upper bound must be a nonnegative integer')
            bounds.append((value, source))
    if 'lower' in loop and 'upper' in loop:
        lower, upper, step = loop['lower'], loop['upper'], loop.get('step')
        if any(type(v) is not int for v in (lower, upper, step)) or step <= 0:
            raise ValueError('constant loop bounds require integers and a positive step')
        count = max(0, (upper - lower + step - 1) // step)
        if any(bound < count for bound, _ in bounds):
            raise ValueError('trip count upper bound is smaller than the known iteration count')
        return count, 'constant_bounds'
    if bounds:
        # Both values assert safe bounds over ALL legal block/thread IDs.
        return min(bounds)
    if required:
        raise ValueError('makespan requires a finite trip count: export constant bounds or '
                         'loop.trip_count_upper_bound, or pass --trip-count-upper-bound '
                         'over all legal bid/tid values')
    return None, 'unknown'


def add_makespan(model, starts, period, latencies, real_tasks, *, trip_count, horizon,
                 stage_limit):
    """T=(N-1)*II + max(start+latency)-min(start), excluding zero-issue views.

    Starts already include stage*II. Keeping their minimum variable avoids
    charging the unused prefix of the prologue to the objective. Completion
    includes asynchronous results, even when they cross a period boundary.
    """
    end_bound = horizon * (stage_limit + 2)
    total_bound = end_bound + max(0, trip_count - 1) * horizon
    if total_bound >= 1 << 60:
        raise ValueError('finite makespan horizon exceeds the safe integer range')
    first = model.new_int_var(0, end_bound, 'first_real_issue')
    last = model.new_int_var(0, end_bound, 'last_real_completion')
    span = model.new_int_var(0, end_bound, 'iteration_span')
    total = model.new_int_var(0, total_bound, 'finite_makespan')
    if not real_tasks or trip_count == 0:
        for var in (first, last, span, total):
            model.add(var == 0)
    else:
        model.add_min_equality(first, [starts[i] for i in real_tasks])
        model.add_max_equality(last, [starts[i] + latencies[i] for i in real_tasks])
        model.add(span == last - first)
        model.add(total == (trip_count - 1) * period + span)
    return dict(makespan=total, iteration_span=span, first_issue=first, last_completion=last)
