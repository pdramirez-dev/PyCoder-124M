def mean(values):
    """Return the arithmetic mean of a nonempty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def clamp(value, lower, upper):
    if lower > upper:
        raise ValueError("invalid interval")
    return max(lower, min(value, upper))
