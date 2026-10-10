"""Layout, colours and codes shared by the dashboard."""
# layout, in pixels
WINDOW = (1280, 760)
CELL = 14
GRID_ORIGIN = (10, 44)
PANEL_X = 730
PANEL_W = 540
TRAIL_LEN = 60             # frames of trail behind each drone
SPEEDS = [1, 2, 5, 10, 15, 30, 60, 120]   # replay speeds, steps per second
FLASH_FRAMES = 12          # a safety correction stays highlighted this many frames
PIN_FRAMES = 80            # a detection alert keeps its label this many frames
LANES = 4                  # rows of the incident timeline

# colours
BG = (13, 19, 26)
PANEL_BG = (19, 27, 36)
SLOT = (45, 56, 68)
TEXT = (230, 237, 243)
MUTED = (139, 152, 165)
GROUND = (26, 54, 34)
GROUND_DRY = (58, 70, 38)
CANOPY = [(30, 78, 42), (36, 90, 48), (42, 100, 52), (27, 70, 39)]
COVER = (95, 208, 138, 80)
FOG = (6, 10, 16)
VORONOI = (255, 255, 255, 80)
GREEN = (95, 208, 138)
AMBER = (255, 191, 0)
DEAD = (110, 116, 124)
HOME = (120, 200, 255)
LOST = (228, 87, 46)
PLAN_C = (255, 214, 64)
TRACK_C = (255, 92, 138)
RADIO = (120, 220, 255)
SAFETY = (255, 214, 64)
FIRE_C = (255, 120, 40)
INTRUDER_C = (200, 110, 255)
INCIDENT_COLORS = {"fire": FIRE_C, "intruder": INTRUDER_C}
# mission mode codes written by record_episode.py
EXPLORE, RETURN, DOCKED, STRANDED, TRACK, LANDED = 0, 1, 2, 3, 4, 5
# who flew a drone, by the names record_episode.py stores in metadata["flown_by_codes"]
FLOWN_STYLE = {"policy": ("POLICY", "AI", GREEN), "planner": ("PLANNER", "PL", PLAN_C),
               "return": ("RETURN", "RTH", HOME), "docked": ("DOCKED", "", MUTED),
               "stranded": ("LOST", "", LOST), "track": ("TRACKING", "TRK", TRACK_C),
               "landed": ("LANDED", "", AMBER)}
UAV_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
TYPE_COLORS = {"animal": (74, 163, 255), "fire": (228, 87, 46), "poi": (175, 183, 191)}
TYPE_LABELS = {"animal": "Animals", "fire": "Fire", "poi": "Points of interest"}
