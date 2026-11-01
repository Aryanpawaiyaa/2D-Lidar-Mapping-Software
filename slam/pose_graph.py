"""SE(2) Pose Graph Optimization using Gauss-Newton on the manifold."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PoseNode:
    node_id: int
    pose: np.ndarray             # shape (3,), [x, y, theta]
    session_id: str = "session_0"
    submap_id: int = 0
    timestamp: float = 0.0


@dataclass
class PoseEdge:
    from_id: int
    to_id: int
    relative_pose: np.ndarray    # shape (3,), [dx, dy, dtheta] in 'from' frame
    info_matrix: np.ndarray      # shape (3, 3), precision matrix (inverse covariance)
    is_loop_closure: bool = False


class PoseGraph:
    """Non-linear SE(2) Pose Graph."""

    def __init__(self):
        self.nodes: Dict[int, PoseNode] = {}
        self.edges: List[PoseEdge] = []

    def add_node(
        self,
        node_id: int,
        pose: np.ndarray,
        session_id: str = "session_0",
        submap_id: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        self.nodes[node_id] = PoseNode(
            node_id=node_id,
            pose=pose.copy().astype(np.float64),
            session_id=session_id,
            submap_id=submap_id,
            timestamp=timestamp,
        )

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        info_matrix: Optional[np.ndarray] = None,
        is_loop_closure: bool = False,
    ) -> None:
        if info_matrix is None:
            # Default information matrix: high confidence
            info_matrix = np.diag([100.0, 100.0, 500.0]).astype(np.float64)

        self.edges.append(
            PoseEdge(
                from_id=from_id,
                to_id=to_id,
                relative_pose=relative_pose.copy().astype(np.float64),
                info_matrix=info_matrix.copy().astype(np.float64),
                is_loop_closure=is_loop_closure,
            )
        )

    def optimize(self, max_iterations: int = 15, fixed_node_id: Optional[int] = 0) -> float:
        """Run Gauss-Newton optimization over all active nodes.
        
        Args:
            max_iterations: Maximum iterations.
            fixed_node_id: ID of the anchor node kept constant.
            
        Returns:
            Final total chi-squared error.
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return 0.0

        node_ids = sorted(list(self.nodes.keys()))
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)
        dim = n_nodes * 3

        if fixed_node_id is None or fixed_node_id not in id_to_idx:
            fixed_node_id = node_ids[0]
        fixed_idx = id_to_idx[fixed_node_id]

        total_chi2 = 0.0

        for it in range(max_iterations):
            H = sparse.lil_matrix((dim, dim), dtype=np.float64)
            b = np.zeros(dim, dtype=np.float64)
            total_chi2 = 0.0

            for edge in self.edges:
                if edge.from_id not in id_to_idx or edge.to_id not in id_to_idx:
                    continue

                i = id_to_idx[edge.from_id]
                j = id_to_idx[edge.to_id]

                pi = self.nodes[edge.from_id].pose
                pj = self.nodes[edge.to_id].pose
                dp = edge.relative_pose
                Omega = edge.info_matrix

                # SE(2) error vector: e = [R_i^T (p_j - p_i) - dp_meas, wrap(theta_j - theta_i - dtheta)]
                xi, yi, thi = pi[0], pi[1], pi[2]
                xj, yj, thj = pj[0], pj[1], pj[2]

                cos_i = math.cos(thi)
                sin_i = math.sin(thi)

                dx = xj - xi
                dy = yj - yi

                e_x = cos_i * dx + sin_i * dy - dp[0]
                e_y = -sin_i * dx + cos_i * dy - dp[1]
                e_th = (thj - thi - dp[2] + math.pi) % (2 * math.pi) - math.pi

                e = np.array([e_x, e_y, e_th], dtype=np.float64)
                chi2 = float(e.T @ Omega @ e)
                total_chi2 += chi2

                # Jacobians A_ij = de / dpi, B_ij = de / dpj
                # de / dpi:
                # de_pos / dpi_pos = -R_i^T
                # de_pos / dthi = [-sin_i * dx + cos_i * dy, -cos_i * dx - sin_i * dy]
                # de_th / dthi = -1
                A = np.zeros((3, 3), dtype=np.float64)
                A[0, 0] = -cos_i
                A[0, 1] = -sin_i
                A[0, 2] = -sin_i * dx + cos_i * dy
                A[1, 0] = sin_i
                A[1, 1] = -cos_i
                A[1, 2] = -cos_i * dx - sin_i * dy
                A[2, 2] = -1.0

                # de / dpj:
                # de_pos / dpj_pos = R_i^T
                # de_th / dthj = 1
                B = np.zeros((3, 3), dtype=np.float64)
                B[0, 0] = cos_i
                B[0, 1] = sin_i
                B[1, 0] = -sin_i
                B[1, 1] = cos_i
                B[2, 2] = 1.0

                # H contributions:
                # H_ii += A^T Omega A
                # H_ij += A^T Omega B
                # H_ji += B^T Omega A
                # H_jj += B^T Omega B
                idx_i = i * 3
                idx_j = j * 3

                H[idx_i : idx_i + 3, idx_i : idx_i + 3] += A.T @ Omega @ A
                H[idx_i : idx_i + 3, idx_j : idx_j + 3] += A.T @ Omega @ B
                H[idx_j : idx_j + 3, idx_i : idx_i + 3] += B.T @ Omega @ A
                H[idx_j : idx_j + 3, idx_j : idx_j + 3] += B.T @ Omega @ B

                # b contributions: b_i += A^T Omega e, b_j += B^T Omega e
                b[idx_i : idx_i + 3] += A.T @ Omega @ e
                b[idx_j : idx_j + 3] += B.T @ Omega @ e

            # Fix anchor node by adding huge penalty to diagonal
            anchor_idx = fixed_idx * 3
            for k in range(3):
                H[anchor_idx + k, anchor_idx + k] += 1e9

            # Solve H * delta = -b
            H_csr = H.tocsr()
            try:
                delta = spsolve(H_csr, -b)
            except Exception:
                break

            # Apply delta updates to poses
            max_step = 0.0
            for nid, idx in id_to_idx.items():
                if idx == fixed_idx:
                    continue
                d = delta[idx * 3 : idx * 3 + 3]
                step = math.hypot(d[0], d[1])
                if step > max_step:
                    max_step = step

                self.nodes[nid].pose[0] += d[0]
                self.nodes[nid].pose[1] += d[1]
                self.nodes[nid].pose[2] = (self.nodes[nid].pose[2] + d[2] + math.pi) % (2 * math.pi) - math.pi

            if max_step < 1e-4:
                break

        return total_chi2

