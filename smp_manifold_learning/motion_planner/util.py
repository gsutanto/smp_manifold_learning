import numpy as np
import copy
import configparser
from math import sqrt, pi, gamma, inf
from sklearn.neighbors import KDTree as cKDTree
from heapq import nsmallest
from scipy.spatial.transform.rotation import Rotation
import plotly.graph_objs as go
from plotly.offline import plot


def unit_ball_measure(n):
    return (sqrt(pi) ** n) / gamma(float(n) / 2.0 + 1.0)


def is_on_manifold(m, q, eps=1e-4):
    return np.linalg.norm(m.y(q)) < eps


def read_cfg(cfg_name):
    cfg = configparser.ConfigParser()
    if cfg_name[-3:] == 'cfg':
        cfg.read(cfg_name)
    else:
        cfg.read_string(cfg_name)

    c = dict()
    c['SEED'] = cfg.getint('general', 'SEED')
    c['CONV_TOL'] = cfg.getfloat('general', 'CONV_TOL')
    c['N'] = cfg.getint('general', 'N')
    c['ALPHA'] = cfg.getfloat('general', 'ALPHA', fallback=1.0)
    c['BETA'] = cfg.getfloat('general', 'BETA', fallback=0.0)
    c['COLLISION_RES'] = cfg.getfloat('general', 'COLLISION_RES', fallback=1.0)
    c['EPS'] = cfg.getfloat('general', 'EPS', fallback=1e-2)
    c['RHO'] = cfg.getfloat('general', 'RHO', fallback=5e-1)
    c['R_MAX'] = cfg.getfloat('general', 'R_MAX', fallback=1e-1)
    c['GREEDY'] = cfg.getboolean('general', 'GREEDY', fallback=False)
    c['PROJ_STEP_SIZE'] = cfg.getfloat('general', 'PROJ_STEP_SIZE', fallback=1.0)

    return c


def plot_box(pd, pos, quat, size):
    d = -size
    p = size
    X = np.array([[d[0], d[0], p[0], p[0], d[0], d[0], p[0], p[0]],
                  [d[1], p[1], p[1], d[1], d[1], p[1], p[1], d[1]],
                  [d[2], d[2], d[2], d[2], p[2], p[2], p[2], p[2]]])

    R = Rotation.from_quat(quat)
    X = R.apply(X.T) + pos

    pd.append(go.Mesh3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        flatshading=True,
        lighting=dict(facenormalsepsilon=0),
        lightposition=dict(x=2000, y=1000),
        color='black',
        # i, j and k give the vertices of triangles
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='y',
        showscale=False
        )
    )


class Tree:
    def __init__(self, d=2, exact_nn=True):
        self.d = d
        self.V = dict()
        self.E = dict()
        self.path = []
        self.edge_count = 0
        self.node_count = 0
        self.exact_nn = exact_nn

        self.kd_buffer_limit = 100
        self.kd_tree = None
        self.kd_off_ids = []

    def add_node(self, node_id, node_value, node_cost, inc_cost):
        if node_id in self.V:
            raise Exception('[Tree/add_node] Node id already exists.')

        self.V[node_id] = Node(node_id, node_value, node_cost, inc_cost)
        self.node_count += 1

        self.kd_off_ids.append(node_id)

    def add_edge(self, edge_id, node_a, node_b):
        if edge_id in self.E:
            raise Exception('[Tree/add_edge] Edge id already exists.')

        self.E[edge_id] = Edge(edge_id, node_a, node_b)
        self.V[node_a].children.append(node_b)
        self.V[node_b].parent = node_a
        self.edge_count += 1

    def is_node(self, node_id):
        if node_id in self.V:
            return True
        else:
            return False

    def is_node_value(self, node_value, tol=1e-12):
        ids = [v.id for v in self.V.values() if np.linalg.norm(v.value - node_value) < tol]

        if len(ids) == 0:
            return False
        else:
            return True

    def is_edge(self, node_a, node_b):
        ids = [e.id for e in self.E.values() if (e.node_a == node_a and e.node_b == node_b)]

        if len(ids) == 0:
            return False
        else:
            return True

    def update_kd_tree(self):
        if self.kd_tree is None or len(self.V) < self.kd_buffer_limit or (
                len(self.V) - len(self.kd_tree.data)) > self.kd_buffer_limit:
            # data = np.array()
            data = np.stack([v.value for v in self.V.values()])
            self.kd_tree = cKDTree(data)
            self.kd_off_ids.clear()

    def get_nearest_neighbor(self, node_value):
        if self.exact_nn:
            near_node = min(self.V.values(), key=lambda v: np.linalg.norm(v.value - node_value))
        else:
            self.update_kd_tree()

            near_idx = self.kd_tree.query(X=[node_value], return_distance=False)
            near_idx = min(self.kd_off_ids + list(near_idx.flatten()), key=lambda idx: np.linalg.norm(self.V[idx].value - node_value))
            near_node = self.V[near_idx]

        return near_node.value, near_node.id

    def get_nearest_neighbors(self, node_value, radius):
        if self.exact_nn:
            near_ids = [v.id for (k, v) in self.V.items() if np.linalg.norm(v.value - node_value) <= radius]
        else:
            self.update_kd_tree()

            near_ids = self.kd_tree.query_radius(X=[node_value], r=np.max([radius, 1e-12]))
            # near_ids = self.kd_tree.query_ball_point(x=node_value, r=np.max([radius, 1e-12]))
            if len(near_ids[0]) > 0:
                near_ids = [idx for idx in self.kd_off_ids + list(near_ids[0].flatten())
                            if np.linalg.norm(self.V[idx].value - node_value) <= radius]
            else:
                near_ids = [idx for idx in self.kd_off_ids
                            if np.linalg.norm(self.V[idx].value - node_value) <= radius]

        return near_ids

    def get_k_nearest_neighbors(self, node_value, k):
        if self.exact_nn:
            nl = nsmallest(k, [v for (k, v) in self.V.items()], key=lambda v: np.linalg.norm(v.value - node_value))
            nl = [v.id for v in nl]
        else:
            self.update_kd_tree()
            # k_near_ids = self.kd_tree.query(x=node_value, k=k)
            k_near_ids = self.kd_tree.query(X=[node_value], k=min(k, len(self.kd_tree.data)), return_distance=False)

            nl = nsmallest(k, [idx for idx in self.kd_off_ids + list(k_near_ids[0].flatten())],
                           key=lambda idx: np.linalg.norm(self.V[idx].value - node_value))
        return nl

    def remove_edge_from_id(self, edge_id):
        if edge_id not in self.E:
            raise Exception('[graph/remove_edge] Edge is not in Graph')
        self.V[self.E[edge_id].node_a].children.remove(self.E[edge_id].node_b)
        del self.E[edge_id]

    def remove_edge(self, node_a_id, node_b_id):
        ids = [e.id for e in self.E.values() if (e.node_a == node_a_id and e.node_b == node_b_id) or
               (e.node_a == node_b_id and e.node_b == node_a_id)]
        if len(ids) == 0:
            raise Exception('[graph/remove_edge] Edge is not in Graph')
        self.V[node_a_id].children.remove(node_b_id)
        del self.E[ids[0]]

    # computes path from start to a given node
    def comp_path(self, node_idx):
        # find nodes that reached the goal
        if not self.is_node(node_idx):
            raise Exception('node_idx is not in the Tree')

        # iterate through parent list until start is reached
        path = [node_idx]
        node_idx = self.V[node_idx].parent

        while node_idx != 0:
            path.append(node_idx)
            node_idx = self.V[node_idx].parent

        # check if root node is virtual
        if not np.isinf(self.V[0].value[0]):
            path.append(node_idx)

        path = list(reversed(path))
        return path

    # computes optimal path to a goal
    def comp_opt_path(self, goal_value, conv_tol=None):
        # find nodes that reached the goal
        if conv_tol:
            V_near = [v for v in self.V.values() if np.linalg.norm(v.value - goal_value) <= conv_tol]
            if len(V_near) == 0:
                return inf
            min_node = min(V_near, key=lambda v: v.cost)
        else:
            min_node = min([v for (k, v) in self.V.items() if k >= 0], key=lambda v: np.linalg.norm(v.value - goal_value))

        # iterate through parent list until start is reached
        self.path = [min_node.id]
        node_id = min_node.parent
        # while node_id is not None:
        while node_id is not None and node_id >= 0:
            self.path.append(node_id)
            node_id = self.V[node_id].parent

        self.path = list(reversed(self.path))
        return min_node.cost

    def update_child_costs(self, node_id):
        # update child costs via depth first strategy
        curr_s = copy.copy(self.V[node_id].children)

        while len(curr_s) > 0:
            child_id = curr_s.pop()
            parent_id = self.V[child_id].parent
            self.V[child_id].cost = self.V[parent_id].cost + \
                                    np.linalg.norm(self.V[child_id].value - self.V[parent_id].value)
            curr_s.extend(self.V[child_id].children)


class Node:
    def __init__(self, id, value, cost, inc_cost):
        self.id = id
        self.value = value
        self.parent = None
        self.children = []
        self.cost = cost           # costs from root to node
        self.inc_cost = inc_cost   # costs from parent to node
        self.con_extend = False    # property used by SMP to determine if node was already extended

    def __str__(self):
        return "id: " + str(self.id) + ", value: " + str(self.value) + ", parent: " + \
               str(self.parent) + ", cost: " + str(self.cost)


class Edge:
    def __init__(self, id, node_a, node_b):
        self.id = id
        self.node_a = node_a
        self.node_b = node_b

    def __str__(self):
        return "id: " + str(self.id) + ", node_a: " + str(self.node_a) + ", node_b: " + str(self.node_b)
