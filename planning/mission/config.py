from __future__ import annotations

from dataclasses import dataclass

# what a UAV is doing
DOCKED, EXPLORE, RETURN, STRANDED, TRACK, LANDED = "docked", "explore", "return", "stranded", "track", "landed"
AIRBORNE = (EXPLORE, RETURN, TRACK)
# who produced a UAV's action in a step (stored in recordings as "flown_by")
FLOWN_BY = {"policy": 0, "planner": 1, RETURN: 2, DOCKED: 3, STRANDED: 4, TRACK: 5, LANDED: 6}

# one landing pad per UAV beside the base; rows and columns below 2 never hold trees
LANDING_PADS = ((1, 1), (1, 3), (3, 1), (1, 5), (5, 1))


@dataclass
class MissionConfig:
    # ---- basic mission
    coverage_override: bool = True   # a UAV finding no new ground is routed to uncovered ground
    return_home: bool = True         # fly home before the battery runs out
    recharge: bool = True            # docked UAVs recharge and relaunch
    launch_gap: int = 40             # steps between take-offs
    recharge_steps: int = 100        # steps for an empty battery to refill
    reserve_steps: int = 10          # spare steps kept on top of the trip home (without safety)
    stall_limit: int = 10            # steps without new coverage before the override
    # ---- safety layer
    safety: bool = False             # supervisor, corner-free routes, landing pads, robust following
    separation: float = 1.0          # minimum distance between flying UAVs, in cells
    reserve_fraction: float = 0.2    # battery share kept for the trip home
    trip_margin: float = 1.3         # safety factor on the trip home
    position_noise: float = 0.0      # std of the simulated position error, in cells
    noise_seed: int = 0              # seed of the private RNG for that error (and for events)
    # ---- smarter override
    gain_targets: bool = False       # target the spot revealing most uncovered cells per distance
    gain_offset: float = 10.0        # score = cells revealed / (distance + gain_offset)
    chain_targets: bool = False      # keep the UAV, target after target, until it reaches fresh ground
    hand_back_cells: int = 30        # uncovered cells nearby that count as fresh ground ...
    hand_back_radius: int = 10       # ... within this many cells
    # ---- persistent surveillance
    patrol: bool = False             # no finish line: keep revisiting the ground seen longest ago
    stagger_first_sortie: bool = True    # in patrol, UAV i ends its first flight early
    spare_packs: int = 0             # charged spare batteries at the base
    swap_steps: int = 5              # steps to swap a battery
    events: bool = False             # fires and intruders appear (hidden from the policy)
    event_rate: float = 1 / 40       # expected new events per step
    track_steps: int = 40            # steps a UAV watches an event after confirming it
    fresh_window: int = 100          # "seen recently" means within this many steps
    # ---- fixes for flying real drones
    confirm_low_battery: int = 2     # with position error: low readings in a row before turning home
    block_steps: int = 100           # a target given up on can be picked again after this many steps
    return_patience: int = 8         # steps without getting closer to home before escalating
    fire_standoff: float = 2.5       # cells between a watching UAV and the fire's edge
    fire_margin: float = 1.5         # no-fly margin around a detected fire, in cells
    intruder_standoff: float = 3.0   # distance at which an intruder is followed, in cells
    wind: tuple = (0.0, 1.0)         # direction the wind blows towards; fires are watched from upwind
