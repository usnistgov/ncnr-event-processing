import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
#%matplotlib inline
def oscillating_motor(x_min, x_max, v_max, a, d=None, scale=1):
    """
    Returns the period and the motion profile for an oscillating motor. The
    motion profile returns (position, velocity) at time t.

    *x_min* and *x_max* are the motor limits (software position).

    *v_max* > 0 is the maximum velocity (raw units).

    *a* and *d* are the acceleration and deceleration (raw units)
    with *d* defaulting to *a*.

    *scale* converts raw units (velocity and acceleration) into software
    units.

    Applying maximum acceleration and deceleration we get times
        ta, td = v_max/a, v_max/d
    This gives travel distance
        xa, xd = a ta^2/2, d td^2/2
    The remaining distance is traversed at maximum velocity
        tc = ((x_max - x_min) - (xa + xd)) / v_max
    which gives oscillation time for the full cycle as
        period = 2 (ta + tc + td)

    If the acceleration/deceleration distances are too long for the x range
    then it chooses the crossover point such that it is always accelerating
    or decelerating, and never at constant velocity.

    Since peak velocity is the same for acceleration and deceleration,
        a ta = d td = v
    Distance covers full range so
        Δx = xa + xd = a ta^2/2 + d td^2 / 2 = v^2/2 * (1/a + 1/d)
    And therefore peak velocity is
        v = √ ( 2Δx / (1/a + 1/d) )
    From this we get acceleration and deceleration times
        ta, td = v/a, v/d
    There is no constant velocity time, so the full cycle is
        period = 2 (ta + td)
    """
    if x_min >= x_max:
        raise ValueError(f"Cannot have {x_min=:.1f} above {x_max=:.1f}")
    if d is None:
        d = a
    Δx = (x_max - x_min)/scale  # motor range (raw)
    ta, td = v_max/a, v_max/d # acceleration time
    xa, xd = a*ta**2/2, d*td**2/2  # acceleration distance
    if xa + xd > Δx:
        # acceration is too slow so set a lower v_max
        v_max = sqrt(2*Δx / (1/a + 1/d))
        ta, td = v_max/a, v_max/d
        xa, xd = a*ta**2/2, d*td**2/2
        #print("too big")
    #print(f"{v_max=:.2f} {xa=:.2f} {xd=:.2f} {ta=:.2f} {td=:.2f}")
    tc = (Δx - (xa + xd) ) / v_max
    # half cycle time
    period = ta + tc + td
    def motion(t):
        """return (position,velocity) at time t"""
        cycle = int(t // period)
        phase = t - cycle * period
        if phase < ta:
            tp = phase
            x, v = a*tp**2/2, a*phase
        elif phase <= ta + tc:
            tp = phase - ta
            x, v = xa + v_max*tp, v_max
        else:
            tp = phase - (ta+tc)
            x, v = (Δx - xd) + v_max*tp - d*tp**2/2, v_max - d*tp
        return (x_min + scale*x, v) if cycle%2 == 0 else (x_max - scale*x, -v)

    return period, motion

def poll_motor(motion, t_max, Δt):
    """
    Returns poll time, motor position and motor velocity.
    """
    t = np.arange(0, t_max, Δt)
    x, v = zip(*(motion(tk) for tk in t))
    return t, np.array(x), np.array(v)

def plot_motor(motion, t_max, Δt):
    t, x, v = poll_motor(motion, t_max, 0.01)
    plt.plot(t, x, label='position')
    plt.plot(t, v, label='velocity')
    t, x, v = poll_motor(motion, t_max, Δt)
    plt.plot(t, x, '.', label='poll')
    plt.legend()
    plt.grid()

def do():
    # macs A3 has a = d = 20, vmax = 10
    period, motion = oscillating_motor(x_min=0, x_max=57, v_max=10, a=20, d=20, scale=1)
    plot_motor(motion, 200, 1.5)

if __name__ == "__main__":
    do()
    plt.show()