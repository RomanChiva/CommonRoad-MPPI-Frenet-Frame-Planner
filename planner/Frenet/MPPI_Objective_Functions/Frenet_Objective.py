import torch
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import pickle
import time


class Objective(object):
    def __init__(self, cfg, device, reference_spline, global_path):
        self.nav_goal = torch.tensor(cfg.goal, device=device)
        self.v_ref = torch.tensor(cfg.v_ref, device=device)
        self.reference_spline = reference_spline
        self.global_path = global_path

    # Function to find the closest point on the global path to the given (x, y)
    def find_closest_s(self, x, y):
        result = minimize(
            lambda s: np.linalg.norm(np.array([self.reference_spline.sx(s) - x, self.reference_spline.sy(s) - y])), 0.0)
        return result.x

    def orthogonal_projection_on_spline(self, x, y, spline_x, spline_y):
        # Vector between consecutive spline points
        dp = np.column_stack((np.diff(spline_x), np.diff(spline_y)))

        # Vector from the point to the spline start point
        p_start = np.column_stack((x - spline_x[:-1], y - spline_y[:-1]))

        # Project the point onto the spline segment
        dot_products = np.sum(p_start * dp, axis=1)
        norm_squared = np.sum(dp ** 2, axis=1)
        projections = dot_products / norm_squared

        # Find the parameter 's' that minimizes the distance
        closest_s = self.reference_spline.s[np.argmin(np.abs(projections))]
        return closest_s

    def cart2frt(self, fX, fY, fPsi, faRefX, faRefY):
        nClosestRefPoint, nClosest2ndRefPoint = self.closest_ref_point(fX, fY, faRefX, faRefY)

        if nClosestRefPoint > nClosest2ndRefPoint:
            nNextRefPoint = nClosestRefPoint
        else:
            nNextRefPoint = nClosest2ndRefPoint

        nPrevRefPoint = nNextRefPoint - 1
        if nNextRefPoint == 0:
            nPrevRefPoint = 0
            nNextRefPoint = 1

        fTangentX = faRefX[nNextRefPoint] - faRefX[nPrevRefPoint]
        fTangentY = faRefY[nNextRefPoint] - faRefY[nPrevRefPoint]

        fVecX = fX - faRefX[nPrevRefPoint]
        fVecY = fY - faRefY[nPrevRefPoint]

        fTangentLength = np.linalg.norm([fTangentX, fTangentY])
        fProjectedVecNorm = np.dot([fVecX, fVecY], [fTangentX, fTangentY]) / fTangentLength
        fProjectedVecX = fProjectedVecNorm * fTangentX / fTangentLength
        fProjectedVecY = fProjectedVecNorm * fTangentY / fTangentLength

        fD = self.distance(fVecX, fVecY, fProjectedVecX, fProjectedVecY)

        fX1, fY1 = faRefX[nPrevRefPoint], faRefY[nPrevRefPoint]
        fX2, fY2 = faRefX[nNextRefPoint], faRefY[nNextRefPoint]
        fd = (fX - fX1) * (fY2 - fY1) - (fY - fY1) * (fX2 - fX1)
        nSide = np.sign(fd)
        if nSide > 0:
            fD *= -1

        fS = 0
        for i in range(nPrevRefPoint):
            fS += self.distance(faRefX[i], faRefY[i], faRefX[i + 1], faRefY[i + 1])

        fS += fProjectedVecNorm

        return fS, fD

    def closest_ref_point(self, fX, fY, faRefX, faRefY):
        fClosestLen = np.inf
        nClosestRefPoint = 0

        for i in range(len(faRefX)):
            fRefX, fRefY = faRefX[i], faRefY[i]
            fDist = self.distance(fX, fY, fRefX, fRefY)

            if fDist < fClosestLen:
                fClosestLen = fDist
                nClosestRefPoint = i  # + 1  # MATLAB indexing starts from 1
            else:
                break

        if nClosestRefPoint == len(faRefX) - 1:
            nClosest2ndRefPoint = nClosestRefPoint - 1
        elif nClosestRefPoint == 0:
            nClosest2ndRefPoint = nClosestRefPoint + 1
        else:
            fRefXp1, fRefYp1 = faRefX[nClosestRefPoint + 1], faRefY[nClosestRefPoint + 1]
            fDistp1 = self.distance(fX, fY, fRefXp1, fRefYp1)

            fRefXm1, fRefYm1 = faRefX[nClosestRefPoint - 1], faRefY[nClosestRefPoint - 1]
            fDistm1 = self.distance(fX, fY, fRefXm1, fRefYm1)

            if fDistm1 < fDistp1:
                nClosest2ndRefPoint = nClosestRefPoint - 1
            else:
                nClosest2ndRefPoint = nClosestRefPoint + 1

        return nClosestRefPoint, nClosest2ndRefPoint

    def distance(self, fX1, fY1, fX2, fY2):
        return np.sqrt((fX1 - fX2) ** 2 + (fY1 - fY2) ** 2)

    def frenet_s(self, external_points, reference_path):
        _, _, closest_point_indices, second_closest_point_indices = self.find_closest_points_vectorized(external_points,
                                                                                                        reference_path)
        # Adjust indices to get nPrevRefPoint and nNextRefPoint
        nPrevRefPoint = closest_point_indices
        nNextRefPoint = second_closest_point_indices

        # Handle boundary cases where nNextRefPoint exceeds the array size
        nNextRefPoint[nNextRefPoint == len(reference_path)] = 0

        # Swap indices if nClosest2ndRefPoint is greater than nClosestRefPoint
        swap_indices = nPrevRefPoint > nNextRefPoint
        nPrevRefPoint[swap_indices], nNextRefPoint[swap_indices] = nNextRefPoint[swap_indices], nPrevRefPoint[
            swap_indices]

        # Calculate tangent vectors
        fTangentX = reference_path[nNextRefPoint, 0] - reference_path[nPrevRefPoint, 0]
        fTangentY = reference_path[nNextRefPoint, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate vectors to external points
        fVecX = external_points[:, 0] - reference_path[nPrevRefPoint, 0]
        fVecY = external_points[:, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate lengths and dot products
        fTangentLength = np.linalg.norm([fTangentX, fTangentY], axis=0)
        fProjectedVecNorm = np.sum(np.stack([fVecX, fVecY], axis=1) * np.stack([fTangentX, fTangentY], axis=1),
                                   axis=1) / fTangentLength

        reference_path_x = reference_path[:, 0]
        reference_path_y = reference_path[:, 1]

        # Calculate distances between consecutive points
        distances = np.sqrt(np.diff(reference_path_x) ** 2 + np.diff(reference_path_y) ** 2)

        # Initialize an empty array to store the cumulative distances
        fS = np.zeros_like(nPrevRefPoint, dtype=float)

        # Loop through each entry in nPrevRefPoint and calculate cumulative distances
        for i, index in enumerate(nPrevRefPoint):
            fS[i] = np.sum(distances[:index])

        # Add the projected vectors to get the final S values
        fS += fProjectedVecNorm

        return fS

    def find_closest_points_vectorized(self, external_points, reference_path):
        # Calculate the Euclidean distance between each external point and all points in the reference path
        distances = np.linalg.norm(reference_path[:, np.newaxis, :] - external_points, axis=2)
        # Find the indices of the points with the minimum distances
        closest_point_indices = np.argmin(distances, axis=0)
        # Retrieve the closest points
        closest_points = reference_path[closest_point_indices]

        # Find the second-closest points
        n_closest_ref_point = closest_point_indices
        n_closest_2nd_ref_point = np.zeros_like(external_points[:, 0], dtype=np.int)

        # Cases where nClosestRefPoint is at the beginning or end
        mask_end = n_closest_ref_point == len(reference_path) - 1
        mask_start = n_closest_ref_point == 0

        n_closest_2nd_ref_point[mask_end] = n_closest_ref_point[mask_end] - 1
        n_closest_2nd_ref_point[mask_start] = n_closest_ref_point[mask_start] + 1

        # Cases where nClosestRefPoint is neither at the beginning nor end
        mask_mid = ~mask_end & ~mask_start

        fRefXp1 = reference_path[n_closest_ref_point[:] + 1, 0]
        fRefYp1 = reference_path[n_closest_ref_point[:] + 1, 1]
        fDistp1 = np.linalg.norm(
            external_points[:] - np.stack([fRefXp1, fRefYp1], axis=1), axis=1
        )

        fRefXm1 = reference_path[n_closest_ref_point[:] - 1, 0]
        fRefYm1 = reference_path[n_closest_ref_point[:] - 1, 1]
        fDistm1 = np.linalg.norm(
            external_points[:] - np.stack([fRefXm1, fRefYm1], axis=1), axis=1
        )

        # Update mask_choose_p1 based on distances
        mask_choose_p1 = fDistm1 < fDistp1
        n_closest_2nd_ref_point[mask_mid] = n_closest_ref_point[mask_mid]

        # Subtract 1 where mask_choose_p1 is True in the mask_mid region
        n_closest_2nd_ref_point[:][mask_mid & mask_choose_p1] -= 1
        # Add 1 where mask_choose_p1 is False in the mask_mid region
        n_closest_2nd_ref_point[:][mask_mid & ~mask_choose_p1] += 1

        # Retrieve the second-closest points
        second_closest_points = reference_path[n_closest_2nd_ref_point]
        return closest_points, second_closest_points, closest_point_indices, n_closest_2nd_ref_point

    def frenet_s_tensor(self, external_points, reference_path):
        closest_point_indices, second_closest_point_indices = self.find_closest_points_vectorized_tensor(
            external_points,
            reference_path)
        # Adjust indices to get nPrevRefPoint and nNextRefPoint
        nPrevRefPoint = closest_point_indices
        nNextRefPoint = second_closest_point_indices

        # Handle boundary cases where nNextRefPoint exceeds the array size
        nNextRefPoint[nNextRefPoint == len(reference_path)] = 0

        # Swap indices if nClosest2ndRefPoint is greater than nClosestRefPoint
        swap_indices = nPrevRefPoint > nNextRefPoint
        nPrevRefPoint_temp = nPrevRefPoint.clone()
        nPrevRefPoint[swap_indices] = nNextRefPoint[swap_indices].to(torch.long)
        nNextRefPoint[swap_indices] = nPrevRefPoint_temp[swap_indices].to(torch.int)

        # Calculate tangent vectors
        fTangentX = reference_path[nNextRefPoint, 0] - reference_path[nPrevRefPoint, 0]
        fTangentY = reference_path[nNextRefPoint, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate vectors to external points
        fVecX = external_points[:, 0] - reference_path[nPrevRefPoint, 0]
        fVecY = external_points[:, 1] - reference_path[nPrevRefPoint, 1]

        # Calculate lengths and dot products
        fTangentLength = torch.norm(torch.stack([fTangentX, fTangentY], dim=0), dim=0)
        fProjectedVecNorm = torch.sum(torch.stack([fVecX, fVecY], dim=1) * torch.stack([fTangentX, fTangentY], dim=1),
                                      dim=1) / fTangentLength

        # Calculate cumulative distances
        distances = torch.sqrt(torch.diff(reference_path[:, 0]) ** 2 + torch.diff(reference_path[:, 1]) ** 2)
        # Initialize an empty array to store the cumulative distances
        fS = torch.zeros_like(fProjectedVecNorm, device='cuda:0')

        # Loop through each entry in nPrevRefPoint and calculate cumulative distances
        for i, index in enumerate(nPrevRefPoint):
            fS[i] = torch.sum(distances[:index])

        # Add the projected vectors to get the final S values
        fS += fProjectedVecNorm

        return fS

    def find_closest_points_vectorized_tensor(self, external_points, reference_path):
        # Calculate the Euclidean distance between each external point and all points in the reference path
        distances = torch.norm(reference_path[:, None, :] - external_points, dim=2)

        # Find the indices of the points with the minimum distances
        closest_point_indices = torch.argmin(distances, dim=0)

        # Find the second-closest points
        n_closest_ref_point = closest_point_indices
        n_closest_2nd_ref_point = torch.zeros_like(external_points[:, 0], device='cuda:0', dtype=torch.int)

        # Cases where nClosestRefPoint is at the beginning or end
        mask_end = n_closest_ref_point == len(reference_path) - 1
        mask_start = n_closest_ref_point == 0

        n_closest_2nd_ref_point[mask_end] = (n_closest_ref_point[mask_end] - 1).to(torch.int)
        n_closest_2nd_ref_point[mask_start] = (n_closest_ref_point[mask_start] + 1).to(torch.int)

        # Cases where nClosestRefPoint is neither at the beginning nor end
        mask_mid = ~mask_end & ~mask_start

        fRefXp1 = reference_path[n_closest_ref_point[:] + 1, 0]
        fRefYp1 = reference_path[n_closest_ref_point[:] + 1, 1]
        fDistp1 = torch.norm(
            external_points[:] - torch.stack([fRefXp1, fRefYp1], dim=1), dim=1
        )

        fRefXm1 = reference_path[n_closest_ref_point[:] - 1, 0]
        fRefYm1 = reference_path[n_closest_ref_point[:] - 1, 1]
        fDistm1 = torch.norm(
            external_points[:] - torch.stack([fRefXm1, fRefYm1], dim=1), dim=1
        )

        # Update mask_choose_p1 based on distances
        mask_choose_p1 = fDistm1 < fDistp1
        n_closest_2nd_ref_point[mask_mid] = n_closest_ref_point[mask_mid].to(torch.int)
        n_closest_2nd_ref_point[:][mask_mid & mask_choose_p1] -= 1
        n_closest_2nd_ref_point[:][mask_mid & ~mask_choose_p1] += 1

        return closest_point_indices, n_closest_2nd_ref_point

    def compute_cost(self, state, u):
        pos = state[:, 0:2]
        velocity_cost = torch.square(state[:, 3] - self.v_ref)

        pos_x = state[:, 0]
        pos_y = state[:, 1]
        pos_x_cpu = pos_x.cpu().numpy()
        pos_y_cpu = pos_y.cpu().numpy()

        with open('trajectory.pkl', 'wb') as file:
            pickle.dump(pos_x_cpu, file)
            pickle.dump(pos_y_cpu, file)

        external_points = np.array(list(zip(pos_x_cpu, pos_y_cpu)))
        reference_path = np.array(list(zip(self.global_path[:, 0], self.global_path[:, 1])))

        # closest_point, closest_point_index, _, _ = find_closest_points_vectorized(external_points, reference_path)
        start_time = time.time()
        fS = self.frenet_s(external_points, reference_path)
        d_values_traj = np.abs(
            [np.linalg.norm([x - self.reference_spline.sx(s), y - self.reference_spline.sy(s)]) for s, x, y
             in
             zip(fS, pos_x_cpu, pos_y_cpu)])
        end_time = time.time()
        '''
        reference_path = np.array(list(zip(self.global_path[:, 0], self.global_path[:, 1])))
        reference_path = torch.tensor(reference_path, device='cuda:0')
        fS = self.frenet_s_tensor(pos, reference_path)
        fS = fS.cpu().numpy()

        d_values_traj = [np.linalg.norm([x - self.reference_spline.sx(s), y - self.reference_spline.sy(s)]) for s, x, y
                         in
                         zip(fS, pos_x_cpu, pos_y_cpu)]

        '''

        #print("computational time for reference tracking: ", end_time - start_time)

        return torch.tensor(d_values_traj)
        '''

        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )
        '''