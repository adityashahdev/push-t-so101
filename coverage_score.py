from shapely.affinity import rotate, translate
from shapely.geometry import Polygon



"""
input:
0 0 0
10 0 0

standard t:
(-20.306, -29.955), (-5.276, -29.955), (-5.276, -7.48), (39.594, -7.48),
(39.594, 7.48), (-5.276, 7.48), (-5.276, 29.955), (-20.306, 29.955)
"""


t_points = [
    (-20.306, -29.955),
    (-5.276, -29.955),
    (-5.276, -7.48),
    (39.594, -7.48),
    (39.594, 7.48),
    (-5.276, 7.48),
    (-5.276, 29.955),
    (-20.306, 29.955)
]


def place(x, y, angle):
    t = rotate(Polygon(t_points), angle, origin=(0, 0))
    return translate(t, xoff=x, yoff=y)


def coverage(goal_pose, pose):
    goal = place(*goal_pose)
    return goal.intersection(place(*pose)).area / goal.area


if __name__ == "__main__":
    goal_pose = [float(v) for v in input().split()]
    pose = [float(v) for v in input().split()]
    print(round(coverage(goal_pose, pose), 3))
