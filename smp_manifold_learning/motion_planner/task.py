import numpy as np
from smp_manifold_learning.motion_planner.feature import SphereFeature, ParaboloidFeature, PointFeature


class Task:
    def __init__(self, name):
        self.name = name
        self.d = 3
        self.start = np.zeros(self.d)
        self.lim_lo = np.array([-6., -6., -6.])
        self.lim_up = np.array([6., 6., 6.])
        self.manifolds = []
        self.obstacles = []

        if name == 'sphere':
            if self.d != 3:
                raise Exception('Sphere task only works with an ambient space dimensionality of 3')

            self.start = np.array([1.0, 0.0, 0.0])
            self.goal = np.array([-1.0, 0.0, 0.0])
            self.manifolds.append(SphereFeature(r=1.0))
            self.manifolds.append(PointFeature(goal=self.goal))

        elif name == 'hourglass_sphere':
            if self.d != 3:
                raise Exception('Hourglass task only works with an ambient space dimensionality of 3')

            if self.d != 3:
                raise Exception('Hourglass task only works with an ambient space dimensionality of 3')

            lim = 2.0
            self.lim_lo = np.array([-lim, -lim, -2*lim])
            self.lim_up = np.array([lim, lim, 2*lim])
            self.start = np.array([1.5, 1.5, 2.75])
            A = np.eye(2) * 0.5
            b = np.zeros(2)
            self.manifolds.append(ParaboloidFeature(A=A, b=b, c=0.5))
            self.manifolds.append(SphereFeature(r=1.0))
            self.manifolds.append(ParaboloidFeature(A=-A, b=b, c=-0.5))
            self.manifolds.append(PointFeature(goal=-self.start))

    def getJointSpaceVolume(self):
        vol = 1.0
        for i in range(self.d):
            vol = vol * (self.lim_up[i] - self.lim_lo[i])
        return vol

    def is_collision_conf(self, q):
        return False

    def plot(self, plot_dir, G_list, V_goal_list, opt_path=None):
        import plotly.graph_objs as go
        from plotly.offline import plot
        from smp_manifold_learning.motion_planner.util import plot_box
        colorscales = ['Reds', 'Greens', 'Blues', 'Magentas']
        color = ['red', 'green', 'blue', 'magenta']
        pd = []

        if self.d == 3:
            X = []
            Y = []
            Z = []
            if opt_path:
                for i, path in enumerate(opt_path):
                    X.clear(), Y.clear(), Z.clear()
                    for state in path:
                        X += [state[0]]
                        Y += [state[1]]
                        Z += [state[2]]
                    pd.append(go.Scatter3d(x=X, y=Y, z=Z, marker=dict(color=color[i], size=5), name='Path_M' + str(i)))

            X.clear(), Y.clear(), Z.clear()
            for G in G_list:
                for e in G.E.values():
                    X += [G.V[e.node_a].value[0], G.V[e.node_b].value[0], None]
                    Y += [G.V[e.node_a].value[1], G.V[e.node_b].value[1], None]
                    Z += [G.V[e.node_a].value[2], G.V[e.node_b].value[2], None]
            pd.append(go.Scatter3d(x=X, y=Y, z=Z, mode='lines', showlegend=True,
                                   line=dict(color='rgb(125,125,125)', width=0.5),
                                   hoverinfo='none', name='Tree'))
            pd.append(go.Scatter3d(x=[self.start[0]], y=[self.start[1]], z=[self.start[2]],
                                   mode='markers', marker=dict(color='red', size=5), name='Start'))

            X.clear(), Y.clear(), Z.clear()
            for i, V in enumerate(V_goal_list):
                for j in V:
                    X += [G_list[i].V[j].value[0]]
                    Y += [G_list[i].V[j].value[1]]
                    Z += [G_list[i].V[j].value[2]]
            pd.append(go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                                   marker=dict(color='magenta', size=5),
                                   name='Intersection nodes'))

        for i, m in enumerate(self.manifolds):
            limits = [self.lim_lo[0], self.lim_up[0], self.lim_lo[1], self.lim_up[1]]
            X_m, Y_m, Z_m = m.draw(limits=limits)

            if m.draw_type is "Scatter":
                pd.append(go.Scatter3d(x=X_m, y=Y_m, z=Z_m, showlegend=False, mode='markers',
                                       marker=dict(color=color[i], size=5)))
            elif m.draw_type is "Surface":
                pd.append(go.Surface(x=X_m, y=Y_m, z=Z_m, opacity=0.8, showscale=False,
                                     colorscale=colorscales[i]))

        for obs in self.obstacles:
            plot_box(pd=pd, pos=np.array([0., 0., obs[0]]), quat=np.array([0., 0., 0., 1.]), size=np.array(obs[1:]))

        fig = go.Figure(data=pd, layout=go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1)))
        plot(fig, filename=plot_dir + '/task_' + self.name + '.html', auto_open=True)
