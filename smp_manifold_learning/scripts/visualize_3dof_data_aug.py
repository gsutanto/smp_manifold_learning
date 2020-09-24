import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
from scipy.spatial.transform.rotation import Rotation
from scipy.spatial.ckdtree import cKDTree


def plot_box(pd, pos, quat, size):
    d = -size
    p = size
    X = np.array([[d[0], d[0], p[0], p[0], d[0], d[0], p[0], p[0]],
                  [d[1], p[1], p[1], d[1], d[1], p[1], p[1], d[1]],
                  [d[2], d[2], d[2], d[2], p[2], p[2], p[2], p[2]]])

    R = Rotation.from_quat(quat)
    X = R.apply(X.T) + pos

    pd.append(go.Mesh3d(
        # 8 vertices of a cube
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        flatshading=True,
        lighting=dict(facenormalsepsilon=0),
        lightposition=dict(x=2000, y=1000),
        color='yellow',
        opacity=0.2,
        # i, j and k give the vertices of triangles
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        name='y',
        showscale=False
        )
    )


if __name__ == '__main__':
    # load dataset
    dataset_filepath = '../data/trajectories/3dof_v2_traj'
    dataset_dict = dict()
    dataset_dict['data'] = np.load(dataset_filepath + '.npy')
    dim_ambient = dataset_dict['data'].shape[1]
    N_local_neighborhood = 2 * (2 ** dim_ambient)
    kd_tree = cKDTree(data=dataset_dict['data'])

    i = 2
    epsilon = 1.25e-1
    dist, idx = kd_tree.query(dataset_dict['data'][i], k=N_local_neighborhood + 1)
    idx = idx[1:]
    vecs_from_curr_pt_to_nearest_neighbors = (dataset_dict['data'][idx] -
                                              dataset_dict['data'][i])

    X = vecs_from_curr_pt_to_nearest_neighbors
    XTX = X.T @ X
    S = XTX / (N_local_neighborhood - 1)

    _, s, V = np.linalg.svd(S)

    quat = Rotation.from_matrix(V.T)
    line_width = 10
    line_length = 1e-1
    arrow_cone_length = 0.3 * line_length
    t1 = quat.as_matrix()[:, 0]
    t2 = quat.as_matrix()[:, 1]
    n = quat.as_matrix()[:, 2]

    pd = list()

    # plot all data points:
    pd.append(go.Scatter3d(x=dataset_dict['data'][:, 0],
                           y=dataset_dict['data'][:, 1],
                           z=dataset_dict['data'][:, 2],
                           name='x', mode='markers',
                           marker=dict(size=2)))

    # plot the tangent space (or here tangent plane):
    plot_box(pd=pd, pos=dataset_dict['data'][i], quat=quat.as_quat(),
             size=np.array([line_length, line_length, 1e-6]))

    # plot the tangent space eigenvector #1:
    pd.append(go.Scatter3d(x=[dataset_dict['data'][i, 0], dataset_dict['data'][i, 0] + line_length * t1[0]],
                           y=[dataset_dict['data'][i, 1], dataset_dict['data'][i, 1] + line_length * t1[1]],
                           z=[dataset_dict['data'][i, 2], dataset_dict['data'][i, 2] + line_length * t1[2]],
                           mode='lines', line=dict(width=line_width, color='red')))
    # drawing the arrowhead (is there an easier way?):
    pd.append(go.Cone(x=[dataset_dict['data'][i, 0] + line_length * t1[0]],
                      y=[dataset_dict['data'][i, 1] + line_length * t1[1]],
                      z=[dataset_dict['data'][i, 2] + line_length * t1[2]],
                      u=[arrow_cone_length * t1[0]],
                      v=[arrow_cone_length * t1[1]],
                      w=[arrow_cone_length * t1[2]],
                      colorscale='Reds'))

    # plot the tangent space eigenvector #2:
    pd.append(go.Scatter3d(x=[dataset_dict['data'][i, 0], dataset_dict['data'][i, 0] + line_length * t2[0]],
                           y=[dataset_dict['data'][i, 1], dataset_dict['data'][i, 1] + line_length * t2[1]],
                           z=[dataset_dict['data'][i, 2], dataset_dict['data'][i, 2] + line_length * t2[2]],
                           mode='lines', line=dict(width=line_width, color='green')))
    # drawing the arrowhead (is there an easier way?):
    pd.append(go.Cone(x=[dataset_dict['data'][i, 0] + line_length * t2[0]],
                      y=[dataset_dict['data'][i, 1] + line_length * t2[1]],
                      z=[dataset_dict['data'][i, 2] + line_length * t2[2]],
                      u=[arrow_cone_length * t2[0]],
                      v=[arrow_cone_length * t2[1]],
                      w=[arrow_cone_length * t2[2]],
                      colorscale='Greens'))

    # plot the normal space eigenvector:
    pd.append(go.Scatter3d(x=[dataset_dict['data'][i, 0], dataset_dict['data'][i, 0] + line_length * n[0]],
                           y=[dataset_dict['data'][i, 1], dataset_dict['data'][i, 1] + line_length * n[1]],
                           z=[dataset_dict['data'][i, 2], dataset_dict['data'][i, 2] + line_length * n[2]],
                           mode='lines', line=dict(width=line_width, color='blue')))
    # drawing the arrowhead (is there an easier way?):
    pd.append(go.Cone(x=[dataset_dict['data'][i, 0] + line_length * n[0]],
                      y=[dataset_dict['data'][i, 1] + line_length * n[1]],
                      z=[dataset_dict['data'][i, 2] + line_length * n[2]],
                      u=[arrow_cone_length * n[0]],
                      v=[arrow_cone_length * n[1]],
                      w=[arrow_cone_length * n[2]],
                      colorscale='Blues'))

    # plot the point of interest:
    pd.append(go.Scatter3d(x=[dataset_dict['data'][i, 0]],
                           y=[dataset_dict['data'][i, 1]],
                           z=[dataset_dict['data'][i, 2]],
                           mode='markers', marker=dict(size=20, color='orange')))

    # plot nearest neighbors of the point of interest:
    pd.append(go.Scatter3d(x=dataset_dict['data'][idx, 0],
                           y=dataset_dict['data'][idx, 1],
                           z=dataset_dict['data'][idx, 2],
                           mode='markers', marker=dict(size=10, color='magenta')))

    # plot an example of the augmented data points:
    pd.append(go.Scatter3d(x=[dataset_dict['data'][i, 0] + epsilon * n[0]],
                           y=[dataset_dict['data'][i, 1] + epsilon * n[1]],
                           z=[dataset_dict['data'][i, 2] + epsilon * n[2]],
                           mode='markers', marker=dict(size=10, color='brown')))

    fig = go.Figure(data=pd)
    plot(fig, filename='../plot/sphere_aug.html', auto_open=True)
