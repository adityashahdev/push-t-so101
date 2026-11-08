"""Lightweight clone of the PushT pymunk physics for forward simulation.

Replicates exactly the dynamics of gym_pusht.envs.PushTEnv (walls, kinematic
circle agent under PD control, T-shaped block, damping=0) without rendering or
coverage computation, so candidate pushes can be rolled out quickly.
"""

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d


class SimWorld:
    K_P, K_V = 100, 20
    DT = 0.01
    CONTROL_HZ = 10

    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0.0

        walls = [
            pymunk.Segment(self.space.static_body, (5, 506), (5, 5), 2),
            pymunk.Segment(self.space.static_body, (5, 5), (506, 5), 2),
            pymunk.Segment(self.space.static_body, (506, 5), (506, 506), 2),
            pymunk.Segment(self.space.static_body, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # agent (kinematic circle, r=15)
        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.position = (256, 400)
        # NOTE: the env sets body.friction (a no-op); its shapes keep the
        # pymunk default friction of 0, so contacts are frictionless.
        shape = pymunk.Circle(self.agent, 15)
        self.space.add(self.agent, shape)

        # tee block (same construction as env)
        scale, length, mass = 30, 4, 1
        vertices1 = [(-length * scale / 2, scale), (length * scale / 2, scale),
                     (length * scale / 2, 0), (-length * scale / 2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale / 2, scale), (-scale / 2, length * scale),
                     (scale / 2, length * scale), (scale / 2, scale)]
        self.block = pymunk.Body(mass, inertia1 + inertia1)
        s1 = pymunk.Poly(self.block, vertices1)
        s2 = pymunk.Poly(self.block, vertices2)
        self.block.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        self.block.position = (256, 300)
        self.space.add(self.block, s1, s2)

    def set_state(self, agent_pos, block_pos, block_angle):
        self.agent.position = tuple(agent_pos)
        self.agent.velocity = (0, 0)
        # set angle FIRST: pymunk rotates the body about its COG when setting
        # angle, which moves the origin; setting position afterwards pins it.
        self.block.angle = float(block_angle)
        self.block.position = tuple(block_pos)
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0.0

    def control_step(self, action):
        """One 10 Hz control step = 10 physics substeps with PD control."""
        action = Vec2d(*action)
        for _ in range(int(1 / (self.DT * self.CONTROL_HZ))):
            acc = self.K_P * (action - self.agent.position) + self.K_V * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity = self.agent.velocity + acc * self.DT
            self.space.step(self.DT)

    @property
    def state(self):
        return (np.array(self.agent.position),
                np.array(self.block.position),
                float(self.block.angle))
