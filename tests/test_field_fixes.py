"""Mission-controller fixes for flying real drones.

Run with:  python -m tests.test_field_fixes   (or: python -m pytest tests)
"""
import numpy as np

from env.forest_env import ForestEnv
from env.grid import FREE, OBSTACLE
from planning.mission import DOCKED, EXPLORE, LANDED, RETURN, TRACK, MissionConfig, MissionController
from planning.routing import cell_of, field_at, route_cells, safe_distance_field
from planning.safety import SafetySupervisor
from planning.surveillance import Event


def _setup(walls=(), **cfg):
    """A controller on an open 50 x 50 map (plus the given obstacle slices), all UAVs exploring."""
    np.random.seed(0)
    env = ForestEnv("configs/default.yaml")
    env.reset()
    env.grid = np.full((env.grid_size, env.grid_size), FREE, dtype=env.grid.dtype)
    for w in walls:
        env.grid[w] = OBSTACLE
    env.coverage_map[:] = False
    ctrl = MissionController(MissionConfig(launch_gap=0, safety=True, **cfg), env.grid_size, env.n_agents)
    ctrl.reset(env)
    return env, ctrl


def _place(env, ctrl, i, p, mode=EXPLORE):
    env.uavs[i].pos = np.array(p, dtype=np.float32)
    env.uavs[i].is_active = mode != DOCKED
    ctrl.mode[i] = mode


# --- 1. a position reading inside a tree must not make home look unreachable
def test_home_distance_from_a_reading_inside_a_tree():
    env, ctrl = _setup(walls=[(slice(20, 23), slice(20, 23))])
    uav = env.uavs[0]
    outside = ctrl._battery_needed(0, uav, np.array([19.95, 21.5]))
    inside = ctrl._battery_needed(0, uav, np.array([20.02, 21.5]))     # 0.07 cells of error
    assert np.isfinite(inside), "a reading inside a tree made home unreachable"
    assert abs(inside - outside) < 5.0


def test_low_battery_must_be_read_twice_with_position_error():
    env, ctrl = _setup(position_noise=0.1)
    for i in range(1, env.n_agents):
        _place(env, ctrl, i, (1.5, 1.5 + 2 * i), DOCKED)
    _place(env, ctrl, 0, (30.5, 30.5))
    env.uavs[0].battery = 60.0                                        # far below what the trip home needs
    ctrl.actions(env, [np.zeros(2)] * env.n_agents)
    assert ctrl.mode[0] == EXPLORE, "one low reading already sent it home"
    ctrl.actions(env, [np.zeros(2)] * env.n_agents)
    assert ctrl.mode[0] == RETURN


# --- 4. a target given up on is only banned for a while
def test_given_up_target_comes_back():
    env, ctrl = _setup(gain_targets=True)
    gain = np.zeros((env.grid_size, env.grid_size)); gain[30, 30] = 50
    owner = np.zeros_like(gain, dtype=int)
    ctrl.blocked_until[30, 30] = 150
    ctrl._assign_gain(env, 0, np.array([10.5, 10.5]), gain, owner, t=100)
    assert ctrl.target[0] is None
    ctrl._assign_gain(env, 0, np.array([10.5, 10.5]), gain, owner, t=151)
    assert ctrl.target[0] == (30, 30)


# --- 5. a returning UAV that makes no progress escalates, then lands
def test_stuck_return_escalates_then_lands():
    env, ctrl = _setup()
    _place(env, ctrl, 0, (20.5, 20.5), RETURN)
    _place(env, ctrl, 1, (40.5, 40.5), EXPLORE)                      # its pad is empty
    for i in (2, 3, 4):
        _place(env, ctrl, i, (1.5, 1.5 + 2 * i), DOCKED)
    ctrl._start_return(0)
    p, k = np.array([20.5, 20.5]), ctrl.cfg.return_patience
    ctrl._watch_return(env, 0, env.uavs[0], p)                       # first reading sets the best so far
    for _ in range(k):
        ctrl._watch_return(env, 0, env.uavs[0], p)
    assert ctrl.boosted[0]
    before = list(ctrl.pad_of)
    for _ in range(k):
        ctrl._watch_return(env, 0, env.uavs[0], p)
    assert ctrl.pad_of[0] == before[1] and ctrl.pad_of[1] == before[0], "did not swap to the free pad"
    for _ in range(2 * k):
        ctrl._watch_return(env, 0, env.uavs[0], p)
        if ctrl.mode[0] == LANDED:
            break
    assert ctrl.mode[0] == LANDED and not env.uavs[0].is_active and ctrl.emergency_landings == 1


def test_new_no_fly_zone_does_not_look_like_being_stuck():
    env, ctrl = _setup(events=True)
    _place(env, ctrl, 0, (20.5, 20.5), RETURN)
    ctrl._last_pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
    ctrl.home_best[0], ctrl.home_stall[0] = 10.0, 7                  # a route home that just got longer
    e = Event(0, "fire", np.array([12.5, 12.5], dtype=np.float32), spawn=0, radius=2.0)
    e.detected = 0
    ctrl.events.events.append(e)
    ctrl._update_hazard(env, 1)
    # progress is measured on the new route from here (no false alarm), and a real stall keeps counting
    assert ctrl.home_stall[0] == 7 and ctrl.home_best[0] > 10.0 and np.isfinite(ctrl.home_best[0])


def _walled_base(fire_radius):
    """Trees along column 10 shut the base off except a gap at rows 20-22, with a fire next to the gap."""
    env, ctrl = _setup(walls=[(slice(0, 20), 10), (slice(23, 50), 10)], events=True)
    for i in range(1, env.n_agents):
        _place(env, ctrl, i, (1.5, 1.5 + 2 * i) if i < 3 else (3.5 + 2 * (i - 3), 1.5), DOCKED)
    _place(env, ctrl, 0, (21.5, 30.5), RETURN)
    ctrl._start_return(0)
    e = Event(0, "fire", np.array([21.5, 12.5], dtype=np.float32), spawn=0, radius=fire_radius)
    e.detected = 0
    ctrl.events.events.append(e)
    ctrl._update_hazard(env, 1)
    return env, ctrl


def test_cut_off_uav_crosses_a_fire_margin_but_not_the_fire():
    env, ctrl = _walled_base(0.3)                                     # the gap is in the margin, not the fire
    p = env.uavs[0].pos
    field, crossing = ctrl._return_field(0, p)
    assert field is not None and crossing, "no way home offered across the margin"
    path = route_cells(field, cell_of(p, 50), max_len=80)
    assert not any(ctrl.core[c] for c in path), "route crosses burning ground"
    act = ctrl.actions(env, [np.zeros(2)] * env.n_agents)[0]
    assert ctrl.mode[0] == RETURN and ctrl.crossing[0] and np.linalg.norm(act) > 0.5


def test_uav_cut_off_by_burning_ground_lands():
    env, ctrl = _walled_base(1.5)                                     # the fire itself blocks the gap
    assert ctrl._return_field(0, env.uavs[0].pos)[0] is None
    ctrl.actions(env, [np.zeros(2)] * env.n_agents)
    assert ctrl.mode[0] == LANDED and ctrl.emergency_landings == 1


def test_burning_pad_is_closed():
    env, ctrl = _setup(events=True)
    _place(env, ctrl, 0, (20.5, 20.5)); _place(env, ctrl, 1, (25.5, 25.5)); _place(env, ctrl, 4, (30.5, 30.5), RETURN)
    _place(env, ctrl, 2, (3.5, 1.5), DOCKED); _place(env, ctrl, 3, (1.5, 5.5), DOCKED)
    ctrl._last_pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
    e = Event(0, "fire", np.array([5.5, 5.0], dtype=np.float32), spawn=0, radius=3.5)  # burns pads (5,1), (3,1), (1,5)
    e.detected = 0
    ctrl.events.events.append(e)
    ctrl._update_hazard(env, 1)
    assert ctrl._pad_burning(2) and ctrl._pad_burning(3) and ctrl._pad_burning(4)
    ctrl.actions(env, [np.zeros(2)] * env.n_agents)
    assert not ctrl._pad_burning(4), "the UAV flying home kept its burning pad"
    assert ctrl.pad_switches == 1, "pads passed around instead of switched once"
    ctrl.after_step(env, 2)
    assert ctrl.mode[2] == DOCKED and ctrl.mode[3] == DOCKED, "launched from a pad on fire"


# --- 3. incidents go to the UAV that gets there soonest and has the battery
def test_responder_by_flying_distance_and_battery():
    env, ctrl = _setup(walls=[(slice(0, 46), 25)], events=True)      # a wall with a gap at the bottom
    e = Event(0, "fire", np.array([10.5, 30.5], dtype=np.float32), spawn=0, radius=1.0)
    _place(env, ctrl, 0, (10.5, 20.5))                               # 10 cells away, but behind the wall
    _place(env, ctrl, 1, (25.5, 36.5))                               # 16 cells away, open ground
    for i in (2, 3, 4):
        _place(env, ctrl, i, (1.5, 1.5 + 2 * i), DOCKED)
    pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
    assert ctrl._pick_responder(env, e, pos) == 1
    env.uavs[1].battery = 120.0                                      # not enough to get there and home
    assert ctrl._pick_responder(env, e, pos) == 0


def test_incident_is_handed_over_while_it_still_needs_watching():
    env, ctrl = _setup(events=True)
    e = Event(0, "fire", np.array([30.5, 30.5], dtype=np.float32), spawn=0, radius=1.0)
    e.detected, e.confirmed, e.watch_until, e.tracker = 2, 5, 45, 0
    ctrl.mode[0], ctrl.track_event[0], ctrl.t = TRACK, e, 10
    ctrl._stop_tracking(0)                                           # e.g. it had to go home
    assert e.tracker is None, "the incident was dropped instead of re-opened"
    _place(env, ctrl, 1, (20.5, 20.5))
    ctrl._start_tracking(env, 1, e, 10, env.uavs[1].pos)
    assert ctrl.track_until[1] == 45 and ctrl.handovers == 1


# --- 7. fires are watched from a stand-off ring, upwind
def test_fire_watched_from_upwind_standoff():
    env, ctrl = _setup(events=True)                                  # wind blows towards +column
    e = Event(0, "fire", np.array([25.5, 25.5], dtype=np.float32), spawn=0, radius=2.0)
    cell = ctrl._watch_cell(env, e, np.array([25.5, 10.5]))
    d = np.hypot(cell[0] + 0.5 - 25.5, cell[1] + 0.5 - 25.5)
    lo, hi = e.radius + ctrl.cfg.fire_standoff - 0.75, e.radius + ctrl.cfg.fire_standoff + 1.25
    assert lo <= d <= hi, f"watch point {d:.1f} cells from the fire"
    assert cell[1] + 0.5 < 25.5, "watching from downwind, in the smoke"
    east = ctrl._watch_cell(env, e, np.array([25.5, 40.5]))         # approaching from downwind
    v = np.array([east[0] + 0.5 - 25.5, east[1] + 0.5 - 25.5])
    assert v[1] / np.linalg.norm(v) < 0.75, "parked straight downwind"


# --- 8. a detected fire is a no-fly zone; a UAV caught inside is steered out
def test_detected_fire_is_no_fly_and_uav_escapes():
    env, ctrl = _setup(events=True)
    e = Event(0, "fire", np.array([25.5, 25.5], dtype=np.float32), spawn=0, radius=2.0)
    e.detected = 0
    ctrl.events.events.append(e)
    ctrl._update_hazard(env, 1)
    assert ctrl.hazard[25, 25] and ctrl.hazard[25, 28] and not ctrl.hazard[25, 30]
    field = safe_distance_field(ctrl.plan_grid, (25, 40))
    path = route_cells(field, (25, 10), max_len=80)
    assert path[-1] == (25, 40) and not any(ctrl.hazard[c] for c in path), "route runs through the fire"
    for i in range(1, env.n_agents):
        _place(env, ctrl, i, (1.5, 1.5 + 2 * i), DOCKED)
    _place(env, ctrl, 0, (25.5, 26.5))
    act = ctrl.actions(env, [np.zeros(2)] * env.n_agents)[0]
    assert ctrl.escaping[0] and float(np.dot(act, [0.0, 1.0])) > 0.5, "not steered out of the fire zone"


def test_supervisor_respects_no_fly_cells():
    g = np.zeros((10, 10), dtype=np.int32)
    no_fly = np.zeros((10, 10), dtype=bool); no_fly[5, 5:8] = True
    sup = SafetySupervisor()
    pos = np.array([[5.5, 4.5]], dtype=np.float32)
    out = sup.filter(g, pos, [np.array([0.0, 1.0])], [True], [0], no_fly=no_fly)[0]
    assert not no_fly[cell_of(pos[0] + out, 10)], "flew into the no-fly zone"


# --- 9. intruders are followed at a distance and kept clear of
def test_supervisor_keeps_out_of_intruder_circle():
    g = np.zeros((10, 10), dtype=np.int32)
    q = np.array([5.0, 5.0], dtype=np.float32)
    sup = SafetySupervisor()
    pos = np.array([[5.0, 2.5]], dtype=np.float32)                   # 2.5 cells away, heading straight at it
    out = sup.filter(g, pos, [np.array([0.0, 1.0])], [True], [0], keep_out=[(q, 2.0)])[0]
    assert np.linalg.norm(pos[0] + out - q) >= 2.0 - 1e-6
    inside = np.array([[5.0, 4.0]], dtype=np.float32)                # the intruder walked up to it
    out = sup.filter(g, inside, [np.zeros(2)], [True], [0], keep_out=[(q, 2.0)])[0]
    assert np.linalg.norm(inside[0] + out - q) > 1.0, "did not back away"


def test_passing_uavs_keep_normal_spacing_from_intruders():
    env, ctrl = _setup(events=True)
    e = Event(0, "intruder", np.array([20.5, 22.5], dtype=np.float32), spawn=0)
    e.detected = 0
    ctrl.events.events.append(e)
    _place(env, ctrl, 0, (20.5, 20.5))
    circles = ctrl._keep_out(env, np.array([u.pos for u in env.uavs], dtype=np.float32))
    assert len(circles) == 1 and circles[0][1] == ctrl.cfg.separation  # a wider circle closes narrow gaps


def test_intruder_followed_from_standoff():
    env, ctrl = _setup(events=True)
    e = Event(0, "intruder", np.array([25.5, 25.5], dtype=np.float32), spawn=0)
    cell = ctrl._watch_cell(env, e, np.array([25.5, 10.5]))
    d = np.hypot(cell[0] + 0.5 - 25.5, cell[1] + 0.5 - 25.5)
    assert abs(d - ctrl.cfg.intruder_standoff) <= 0.8


# --- 2. with position error the controller plans from what it believes
def test_noisy_controller_keeps_its_own_coverage_map():
    env, ctrl = _setup(position_noise=0.1)
    assert ctrl._coverage(env) is ctrl.est_cov
    env.step(ctrl.actions(env, [np.zeros(2)] * env.n_agents))
    ctrl.after_step(env, 1)
    assert ctrl.est_cov.any(), "nothing marked as seen from the measured positions"
    believed = ctrl.coverage_estimate(env, {"coverage_rate": -1.0})
    assert 0.0 < believed < 1.0
    exact_env, exact = _setup()
    assert exact._coverage(exact_env) is exact_env.coverage_map     # no error: the true map is used


def test_field_at_without_error_is_the_plain_lookup():
    f = np.arange(100, dtype=float).reshape(10, 10)
    assert field_at(f, np.array([3.4, 7.9])) == f[3, 7]


if __name__ == "__main__":
    from tests.runner import run_tests
    run_tests(globals(), "FIELD-FIX")
