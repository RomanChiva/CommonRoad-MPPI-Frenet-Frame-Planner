import numpy as np
from mppi_planner.mppi import MPPIPlanner
from typing import Callable, Optional
import io
import math
import os
import yaml
from yaml.loader import SafeLoader

import torch

torch.set_printoptions(precision=2, sci_mode=False)


def torch_to_bytes(t: torch.Tensor) -> bytes:
    buff = io.BytesIO()
    torch.save(t, buff)
    buff.seek(0)
    return buff.read()


def bytes_to_torch(b: bytes) -> torch.Tensor:
    buff = io.BytesIO(b)
    return torch.load(buff)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):
        self.cfg = cfg
        self.objective = objective
        self.nx = 4  # number of states
        self.nu = 2  # number of control variables

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg.mppi,
            cfg.nx,
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )

        # Note: place_holder variable to pass to mppi, so it doesn't complain, while the real state is actually the
        # isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg.mppi.num_samples, self.cfg.nx))

    def update_objective(self, objective):
        self.objective = objective

    def dynamics2(self, old_state, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the
        # state is already saved in the simulator itself, so we ignore it. Note: t is an unused step dependent
        # dynamics variable
        """
        v = u[:, 0]
        w = u[:, 1]
        psi = old_state[:, 2]
        trans = torch.stack([
            0.2 * v * torch.cos(psi),
            0.2 * v * torch.sin(psi),
            0.2 * w,
            0.2 * v
        ], dim=-1)
        state = torch.add(old_state, trans)
        return state, u
        """

        # vehicle bicycle model with dynamic steering
        l_f = (2.9 / 2.595)
        l_r = (4.95 / 2.595)
        a = u[:, 0]     # acceleration
        ddelta = u[:, 1]    # steering angle rate
        psi = old_state[:, 2]
        v = old_state[:, 3]
        delta = old_state[:, 4]
        ratio = l_r / (l_r + l_f)
        beta = torch.arctan(ratio * torch.tan(delta))
        trans = torch.stack([
            0.2 * v * torch.cos(psi + beta),
            0.2 * v * torch.sin(psi + beta),
            0.2 * (v/l_r) * torch.sin(beta),
            0.2 * a,
            0.2 * ddelta
        ], dim=-1)
        state = torch.add(old_state, trans)
        return state, u


    def dynamics(self, old_state, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the
        # state is already saved in the simulator itself, so we ignore it. Note: t is an unused step dependent
        # dynamics variable
        """
        v = u[:, 0]
        w = u[:, 1]
        psi = old_state[:, 2]
        trans = torch.stack([
            0.2 * v * torch.cos(psi),
            0.2 * v * torch.sin(psi),
            0.2 * w,
            0.2 * v
        ], dim=-1)
        state = torch.add(old_state, trans)
        return state, u
        """

        t = 0.1

        # vehicle bicycle model with dynamic steering
        l_f = (2.9 / 2.595)
        l_r = (4.95 / 2.595)
        a = u[:, 0]     # acceleration
        ddelta = u[:, 1]    # steering angle rate
        psi = old_state[:, 2]
        v = old_state[:, 3]
        delta = old_state[:, 4]
        ratio = l_r / (l_r + l_f)
        beta = torch.arctan(ratio * torch.tan(delta))
        trans = torch.stack([
            t*v*torch.cos(psi + beta),
            t*v*torch.sin(psi + beta),
            t*(v/l_r) * torch.sin(beta),
            t*a,
            t*ddelta
        ], dim=-1)
        state = torch.add(old_state, trans)


        # Set velocity limits
        state[:, 3] = torch.clamp(state[:, 3], -13.6, 50.8)
        # Set steering angle limits
        state[:, 4] = torch.clamp(state[:, 4], -1.066, 1.066)


        return state, u


    def running_cost(self, state):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the
        # state is already saved and accessible in the simulator itself, so we ignore it and pass a handle to the
        # simulator.
        return self.objective.compute_cost(state)
    
    def convert_to_frenet(self, state):

        fs, fd = self.objective.convert_to_frenet(state)

        return fs, fd
    

    def get_samples(self, q, qdot, obst=None, obst_tensor=None):

        self.state_place_holder = torch.tensor(q * self.cfg.mppi.num_samples).view(self.cfg.mppi.num_samples, -1)
        actions, states = self.mppi.get_trajectories(self.state_place_holder)
        actions = actions.cpu()
        states = states.cpu()
        return actions, states


    def compute_traj(self, drop, predictions, pred_cov, plan_cov, n_samples, discount, weight):

        actions, states = self.mppi.evaluate_trajectories(drop, predictions, pred_cov, plan_cov, n_samples, discount, weight)

        actions = actions.cpu()
        # loop over the actions and forward propagate the dynamics to get the trajectory
        old_state = self.state_place_holder[0, :].unsqueeze(0)
        traj = []
        for i in range(actions.size(0)):
            element = actions[i, :]
            new_state, _ = self.dynamics(old_state, element.unsqueeze(0))
            traj.append(old_state)
            old_state = new_state

        traj = torch.cat(traj, dim=0)

        return actions, traj, states[:, :, 0:2], self.mppi.COST


    def compute_action(self, q, qdot, obst=None, obst_tensor=None):
    
        self.state_place_holder = torch.tensor(q * self.cfg.mppi.num_samples).view(self.cfg.mppi.num_samples, -1)
        actions, states = self.mppi.command(self.state_place_holder, obst)
        actions = actions.cpu()
       
        # loop over the actions and forward propagate the dynamics to get the trajectory
        old_state = self.state_place_holder[0, :].unsqueeze(0)
        traj = []
        for i in range(actions.size(0)):
            element = actions[i, :]
            new_state, _ = self.dynamics(old_state, element.unsqueeze(0))
            traj.append(old_state)
            old_state = new_state

        traj = torch.cat(traj, dim=0)

        return actions, traj, states[:, :, 0:2]

    def command(self):
        return torch_to_bytes(self.mppi.command(self.state_place_holder))
