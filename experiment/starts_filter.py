import csv, math, sys
import numpy as np
import gymnasium as gym, gym_pusht
from shapely.geometry import Polygon, Point

SCALE = 0.5
GOAL_GYM = (256.0, 256.0, math.pi / 4)
G = (185.0, 0.0)
RMIN, RMAX = 130.0, 220.0
TIP_R = 14.88 / 2 + 2.0
Y_LIM = 150.0
PLACE_TOL_MM = 2.0
N_GRADED, N_PRACTICE = 20, 10

OUT = [(-60,0),(60,0),(60,30),(15,30),(15,120),(-15,120),(-15,30),(-60,30)]

BAR_L, BAR_W, STEM_W, TOT_H = 59.91, 15.03, 14.96, 59.90
def tracker_template():
    stem = TOT_H - BAR_W
    ba, sa = BAR_L*BAR_W, STEM_W*stem
    c = (ba*BAR_W/2 + sa*(BAR_W+stem/2))/(ba+sa)
    w, s = BAR_L/2, STEM_W/2
    return c, np.array([(-c,-w),(BAR_W-c,-w),(BAR_W-c,-s),(TOT_H-c,-s),(TOT_H-c,s),(BAR_W-c,s),(BAR_W-c,w),(-c,w)])
CENTROID, TEMPLATE = tracker_template()

def to_desk_poly(x, y, th):
    c, s = math.cos(th), math.sin(th)
    pts = []
    for vx, vy in OUT:
        wx, wy = x + c*vx - s*vy, y + s*vx + c*vy
        pts.append((G[0] + (wx-256)*SCALE, G[1] + (wy-256)*SCALE))
    return Polygon(pts)

def tracker_pose(x, y, th):
    a = th + math.pi/2
    ox, oy = G[0] + (x-256)*SCALE, G[1] + (y-256)*SCALE
    px, py = ox + CENTROID*math.cos(a), oy + CENTROID*math.sin(a)
    deg = (math.degrees(a) + 180) % 360 - 180
    return px, py, deg

def tracker_outline(px, py, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return TEMPLATE @ np.array([[c, s], [-s, c]]) + (px, py)

def margins(poly):
    ring = list(poly.exterior.coords)[:-1]
    rmax = max(math.hypot(*p) for p in ring)
    rmin = poly.exterior.distance(Point(0, 0))
    inside = poly.contains(Point(0, 0))
    ymax = max(abs(p[1]) for p in ring)
    return rmin, rmax, ymax, inside

def passes(poly):
    rmin, rmax, ymax, inside = margins(poly)
    return (not inside) and rmin - TIP_R >= RMIN and rmax + TIP_R <= RMAX and ymax + TIP_R <= Y_LIM

def wrap(d): return (d + 180) % 360 - 180

if __name__ == "__main__":
    env = gym.make("gym_pusht/PushT-v0", obs_type="state").unwrapped
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    env.reset(seed=0)
    env.block.angle = GOAL_GYM[2]; env.block.position = GOAL_GYM[:2]
    print("gym coverage with T on the goal:", round(env._get_coverage(), 6))
    gpoly = to_desk_poly(*GOAL_GYM)
    gx, gy, ga = tracker_pose(*GOAL_GYM)
    print("goal desk pose (tracker convention): x %.2f y %.2f angle %.2f" % (gx, gy, ga))
    print("goal margins (rmin, rmax, ymax, contains base):", [round(v, 2) if not isinstance(v, bool) else v for v in margins(gpoly)], "passes:", passes(gpoly))
    print("tracker T outline vs gym-derived goal polygon: Hausdorff gap %.3f mm" % Polygon(tracker_outline(gx, gy, ga)).hausdorff_distance(gpoly))
    from shapely.geometry import Polygon as P2
    worst = 0.0
    for sd in range(0, 50):
        o, _ = env.reset(seed=sd)
        bx, by, bth = (float(v) for v in o[2:5])
        mine = to_desk_poly(bx, by, bth)
        goal_poly = to_desk_poly(*GOAL_GYM)
        cov_mine = mine.intersection(goal_poly).area / goal_poly.area
        worst = max(worst, abs(cov_mine - env._get_coverage()))
    print("max |my coverage - gym _get_coverage| over seeds 0-49: %.2e" % worst)
    rows, passing, first200 = [], 0, 0
    for seed in range(n_seeds):
        o, _ = env.reset(seed=seed)
        x, y, th = (float(v) for v in o[2:5])
        poly = to_desk_poly(x, y, th)
        ok = passes(poly)
        if seed < 200 and ok: first200 += 1
        if ok:
            passing += 1
            px, py, deg = tracker_pose(x, y, th)
            rmin, rmax, ymax, _ = margins(poly)
            role = "graded" if passing <= N_GRADED else ("practice" if passing <= N_GRADED + N_PRACTICE else "spare")
            rows.append([passing, seed, role, round(x,2), round(y,2), round(math.degrees(th) % 360, 2), round(px,2), round(py,2), round(deg,2), round(abs(wrap(deg-ga)),1), round(rmin-TIP_R,1), round(rmax+TIP_R,1)])
    print("seeds tried:", n_seeds, " passing:", passing, " first 200 seeds passing:", first200)
    for k in (N_GRADED, N_GRADED + N_PRACTICE):
        print("seed number of passing start #%d:" % k, rows[k-1][1])
    with open("passing_seeds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank","seed","role","gym_x_px","gym_y_px","gym_angle_deg","desk_x_mm","desk_y_mm","desk_angle_deg","turn_to_goal_deg","min_radius_with_tip_mm","max_radius_with_tip_mm"])
        w.writerows(rows[:60])
    graded = [r for r in rows if r[2] == "graded"]
    print("graded turn-to-goal degrees:", sorted(r[9] for r in graded))
    print("graded desk x range %.0f-%.0f y range %.0f-%.0f" % (min(r[6] for r in graded), max(r[6] for r in graded), min(r[7] for r in graded), max(r[7] for r in graded)))
