import typing
import bisect


class GlobalPosInt(typing.NamedTuple):
    """Data we need from a GLOBAL_POSITION_INT message."""

    t_now: float
    alt: float
    vn: float
    ve: float


class Attitude(typing.NamedTuple):
    """Data we need from an ATTITUDE message."""

    t_now: float
    roll: float
    pitch: float
    yaw: float


# Some generic magic
class Interpolatable(typing.Protocol):
    t_now: float
    _fields: tuple[str, ...]


T = typing.TypeVar("T", bound=Interpolatable)


def lookup(data: list[T], t_now: float) -> T | None:
    """
    Look up an object by timestamp, interpolating if necessary.
    Works with any NamedTuple containing a 't_now' field and a bunch of floats.
    Uses binary search for O(log n) performance instead of O(n).
    """
    if not data:
        return None

    # Binary search to find the insertion point
    # bisect_left returns the index where t_now would be inserted
    idx = bisect.bisect_left(data, t_now, key=lambda x: x.t_now)

    if idx < len(data) and data[idx].t_now == t_now:
        # Exact match found
        return data[idx]

    if idx == 0:
        # Before the first measurement
        return None

    if idx >= len(data):
        # After the last measurement
        return None

    # Interpolate between data[idx-1] and data[idx]
    d0 = data[idx - 1]
    d1 = data[idx]

    t0 = d0.t_now
    t1 = d1.t_now

    # Avoid division by zero
    if t1 == t0:
        return d0

    ratio = (t_now - t0) / (t1 - t0)

    # Dynamically interpolate all fields
    values = []
    for field in d0._fields:
        if field == "t_now":
            values.append(t_now)
        else:
            v0 = getattr(d0, field)
            v1 = getattr(d1, field)
            values.append(v0 + ratio * (v1 - v0))

    # Construct a new instance of the correct class
    return type(d0)(*values)
