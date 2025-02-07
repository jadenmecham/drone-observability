import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import figurefirst as fifi

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap

import util
import figure_functions as ff

from pybounds import SlidingEmpiricalObservabilityMatrix, SlidingFisherObservability, colorline
from drone_model_body_level import DroneSimulator


class DroneObservability:
    """ Run empirical observability analysis on a fly trajectory.
    """

    def __init__(self, dt=0.1, states=None, sensors=None, time_steps=None,
                 mpc_horizon=10, mpc_control_penalty=1e-1):
        """ Run.
        """

        # Set the states, sensors, & time-steps to use
        if states is None:
            self.states = [['x', 'y', 'z',
                            'v_x', 'v_y', 'v_z',
                            'phi', 'theta', 'psi',
                            'omega_x', 'omega_y', 'omega_z',
                            'w', 'zeta',
                            'm', 'I_x', 'I_y', 'I_z', 'C'], ]
        else:
            self.states = states.copy()

        if sensors is None:
            self.sensors = [['psi', 'gamma'],
                            ['psi', 'gamma', 'beta'],
                            ['psi', 'gamma', 'beta', 'phi', 'theta'],
                            ['psi', 'gamma', 'beta', 'r'],
                            ['psi', 'gamma', 'beta', 'a'],
                            ['psi', 'gamma', 'beta', 'g'],
                            ['psi', 'gamma', 'beta', 'r', 'a'],
                            ['psi', 'gamma', 'beta', 'a', 'g'],
                            ['psi', 'gamma', 'beta', 'a', 'g', 'phi', 'theta'],
                            ['psi', 'gamma', 'beta', 'v_x', 'v_y'],
                            ]
        else:
            self.sensors = sensors.copy()

        if time_steps is None:
            self.time_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            self.time_steps = time_steps.copy()

        # Unique sensors
        sensors_all = []
        for s in self.sensors:
            sensors_all = sensors_all + s

        self.sensors_all = list(set(sensors_all))

        # Every combination of states, sensors, & time-steps
        self.comb = list(itertools.product(self.states, self.sensors, self.time_steps))
        self.n_comb = len(self.comb)

        # Create simulator
        self.simulator = DroneSimulator(dt=dt, mpc_horizon=mpc_horizon, r_u=mpc_control_penalty,
                                        control_mode='velocity_body_level')

        # Set Observability & Fisher Information parameters
        self.SEOM = []

    def run(self, v_x=None, v_y=None, psi=None, z=None, w=None, zeta=None,
            R=0.1, lam=1e-6):
        """ Run.
        """

        # Update the setpoints
        self.simulator.update_setpoint(v_x=v_x, v_y=v_y, psi=psi, z=z, w=w, zeta=zeta)

        # Reconstruct trajectory with MPC
        t_sim, x_sim, u_sim, y_sim = self.simulator.simulate(x0=None, mpc=True, return_full_output=True)

        # Get simulation data
        sim_data = pd.DataFrame(y_sim)
        sim_data.insert(loc=0, column='time', value=t_sim)
        # sim_data['time'] = sim_data['time'].values - sim_data['time'].values[0]

        # Construct observability matrix in sliding windows
        time_window_max = np.max(np.array(self.time_steps)) + 1
        self.SEOM = SlidingEmpiricalObservabilityMatrix(self.simulator, t_sim, x_sim, u_sim,
                                                        w=time_window_max, eps=1e-4)

        # Dictionary to store data for trajectory
        data_dict = {'states': [], 'sensors': [], 'time_steps': [], 'sim_data': [], 'error_variance': [],
                     'O_sliding': self.SEOM.O_df_sliding,}

        for n, c in enumerate(self.comb):  # each state, sensor, time-step combination
            o_states = c[0]
            o_sensors = c[1]
            time_window = c[2]
            o_time_steps = np.atleast_1d(np.arange(1, time_window + 1))

            # Run Fisher Information observability for each sliding window
            # O_sliding = self.SEOM.get_observability_matrix()
            SFO = SlidingFisherObservability(self.SEOM.O_df_sliding, time=self.SEOM.t_sim, lam=lam, R=R,
                                             states=o_states, sensors=o_sensors, time_steps=o_time_steps, w=time_window)

            EV_aligned = SFO.get_minimum_error_variance()

            # Align & rename observability data
            EV_aligned_no_nan = EV_aligned.copy()
            EV_aligned_no_nan = EV_aligned_no_nan.fillna(method='bfill').fillna(method='ffill')
            EV_aligned_rename = EV_aligned_no_nan.copy()
            EV_aligned_rename.columns = ['o_' + item for item in EV_aligned_no_nan.columns]

            # Combine simulation & observability data
            sim_data_new = sim_data.copy()
            sim_data_all = pd.concat([sim_data_new, EV_aligned_rename], axis=1)

            # Append data to dictionary
            data_dict['states'].append(o_states)
            data_dict['sensors'].append(o_sensors)
            data_dict['time_steps'].append(time_window)
            data_dict['error_variance'].append(EV_aligned_no_nan)
            data_dict['sim_data'].append(sim_data_all)

        return data_dict


def norm_function(error_covariance, avoid_zero=2):
    norm = 1 / np.log(error_covariance + avoid_zero)
    return norm


class SpiderArcGridPlot:
    def __init__(self, data, sensors_list, plot_vars=None, cmap=None, norm_min_max=None, states=0, time_steps=0,
                 time_window=None):
        """ Plot.

            Inputs:
                data: data structure with sensors, states, time-steps
                sensors_list: list of lists of sensor combinations
                plot_vars: dictionary where keys are variables to plot & values are colors

        """

        self.data = data
        self.sensors_list = sensors_list

        if plot_vars is None:
            self.plot_vars = {'zeta': np.array([170, 245, 87]) / 255,
                              'w': np.array([18, 139, 127]) / 255}
        else:
            self.plot_vars = plot_vars

        if time_window is None:
            self.time_window = 1
        else:
            self.time_window = time_window

        if states is None:
            self.states = [['v_para', 'v_perp', 'phi', 'phidot', 'w', 'zeta', 'd',
                            'I', 'm', 'C_para', 'C_perp', 'C_phi', 'km1', 'km2', 'km3', 'km4']]
            # unique_data = [list(x) for x in set(tuple(x) for x in self.data['states'])]
        else:
            self.states = states

        if time_steps is None:
            self.time_steps = [4]
        else:
            self.time_steps = time_steps

        if norm_min_max is None:
            self.norm_min_max = None
        else:
            self.norm_min_max = norm_min_max

        # Make plot labels
        self.LatexConverter = ff.LatexStates()
        self.sensor_label = []
        for v in self.sensors_list:
            lat_label = ', '.join(self.LatexConverter.convert_to_latex(v))
            full_label = '[' + lat_label + ']'
            self.sensor_label.append(full_label)

        # Get indices to plot
        self.index_map, self.n_sensors, self.n_states, self.n_time_steps = (
            get_fisher_indices(data,
                               states_list=self.states,
                               sensors_list=self.sensors_list,
                               time_steps_list=self.time_steps))

        # Angles for arc plot
        self.theta = np.linspace(0, 2 * np.pi, len(self.sensors_list), endpoint=False)

        # Get error covariance
        self.sim_data = []
        self.error_variance = []
        self.error_variance_min = {}
        self.error_variance_raw = []
        self.error_variance_norm = []
        self.get_error_covariance()

        # Plotting
        self.ax_grid = None
        self.cmap_var = {}

        if cmap is None:
            self.cmap = Colormaps(power=2, color_dict='red_blue')
        else:
            self.cmap = cmap

    def get_error_covariance(self):
        self.sim_data = []
        self.error_variance = []
        self.error_variance_raw = []
        self.error_variance_norm = []
        self.error_variance_min = {}

        for v in self.plot_vars:
            self.error_variance_min[v] = np.nan * np.zeros(self.n_sensors)

        for s in range(self.n_sensors):
            # Get sim data & error covariance
            j = self.index_map[s, 0, 0]
            sim_data = self.data['sim_data'][j]
            error_variance = self.data['error_variance'][j]

            # if not np.all(sim_data.time.values == error_variance.time.values):
            #     print(sim_data.time.values)
            #     print(error_variance.time.values)
            #     raise Exception('wot')

            # Set time window
            tsim = error_variance['time'].values

            if self.time_window is not None:
                t_start = np.where((tsim >= self.time_window[0]))[0][0]
                t_end = np.where((tsim <= self.time_window[1]))[0][-1]
                sim_data = sim_data[t_start:t_end]

            self.sim_data.append(sim_data)

            # Get minimum error variance
            self.error_variance.append(error_variance)
            self.error_variance_raw.append({})
            self.error_variance_norm.append({})
            for v in self.plot_vars:
                error_var = error_variance[v].values
                if self.time_window is not None:
                    t_start = np.where((tsim >= self.time_window[0]))[0][0]
                    t_end = np.where((tsim <= self.time_window[1]))[0][-1]
                    error_var = error_var[t_start:t_end]

                error_var_norm = norm_function(error_var)

                if self.norm_min_max is not None:
                    pass
                    error_var_norm = ((error_var_norm - self.norm_min_max[0])
                                      / (self.norm_min_max[1] - self.norm_min_max[0]))

                self.error_variance_raw[s][v] = error_var
                self.error_variance_norm[s][v] = error_var_norm
                self.error_variance_min[v][s] = np.max(error_var_norm)

            self.error_variance_norm[s] = pd.DataFrame(self.error_variance_norm[s])

    def plot_spider(self, ax=None, sub_plot_index=(1, 1, 1), var_list=None,
                    arc_color_list=None, show_labels=True, space=0.0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(projection='polar'), dpi=100)
        else:
            fig = plt.gcf()
            ax = ax

        # Spider plot
        for s, v in enumerate(self.plot_vars):
            ax.fill(self.theta, self.error_variance_min[v], facecolor=self.cmap.color_list[v], alpha=0.4)

            ax.plot(np.hstack((self.theta, self.theta[0])),
                    np.hstack((self.error_variance_min[v], self.error_variance_min[v][0])),
                    color=self.cmap.color_list[v],
                    linewidth=1.0)

        a = ax
        a.grid(linewidth=1, color='lightgray')
        a.set_rgrids([0.0, 1.0, 2.0])
        a.set_yticklabels([])
        a.set_rlim(0.0, 2.0)
        a.set_theta_zero_location("N")
        a.tick_params(pad=13)
        a.spines['polar'].set_color('lightgray')
        a.spines['polar'].set_linewidth(1)
        # a.spines['polar'].set_visible(False)
        plt.setp(a.xaxis.get_majorticklabels(), rotation=0, ha='center')
        a.set_thetagrids(np.degrees(self.theta), self.sensor_label, fontsize=6)

        if show_labels:
            pass
        else:
            a.set_xticklabels([])
            # fifi.mpl_functions.adjust_spines(a, [])

        ax_grid = fig.add_subplot(*sub_plot_index)
        fifi.mpl_functions.adjust_spines(ax_grid, [])
        ax_grid.patch.set_alpha(0.0)
        ax_grid.set_aspect(1.0)

        if var_list is None:
            var_list = ['phi', 'gamma', 'psi', 'alpha', 'of', 'a', 'g', 'q']

        if arc_color_list is None:
            arc_color_list = (np.array([120, 120, 120]) / 255,  # gray
                              np.array([255, 204, 102]) / 255,  # dark orange
                              np.array([0, 102, 255]) / 255,  # dark blue
                              np.array([255, 153, 255]) / 255,  # dark pink
                              np.array([10, 205, 245]) / 255,  # medium blue
                              np.array([230, 138, 0]) / 255,  # orange
                              np.array([0, 0, 200]) / 255,  # blue
                              np.array([204, 51, 153]) / 255)  # pink

            # arc_color_list = (np.array([148, 148, 184]) / 255,  # gray
            #                   np.array([230, 138, 0]) / 255,  # dark orange
            #                   np.array([0, 0, 200]) / 255,  # dark blue
            #                   np.array([204, 51, 153]) / 255,  # dark pink
            #                   np.array([102, 255, 51]) / 255,  # medium blue
            #                   np.array([255, 204, 102]) / 255,  # orange
            #                   np.array([0, 102, 255]) / 255,  # blue
            #                   np.array([255, 153, 255]) / 255)  # pink

        ff.plot_arc_grid_map(labels=self.sensors_list,
                             var_list=var_list,
                             color_list=arc_color_list, arrow=False, space=space,
                             ax=ax_grid, r_start=1.05, r_space=0.095, w=0.08)

        lim = 1.8
        ax_grid.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax_grid.set_aspect(1)

        self.ax_grid = ax_grid

    def plot_trajectory(self, ax=None, sensor_index=0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

        # Spider plot
        for s, v in enumerate(self.plot_vars):
            # Set position
            sim_data = self.sim_data[sensor_index]

            xpos = sim_data['xpos'].values
            ypos = sim_data['ypos'].values

            xpos = xpos - np.mean(xpos)
            ypos = ypos - np.mean(ypos)

            phi = sim_data['phi'].values

            # cvar = self.error_variance_norm[sensor_index][v].values
            cvar = self.error_variance_raw[sensor_index][v]

            ff.plot_trajectory(xpos + 0.02 * s - 0.01, ypos, phi,
                               color=cvar,
                               ax=ax,
                               size_radius=0.008,
                               nskip=1,
                               colormap=self.cmap.log[v],
                               colornorm=mpl.colors.LogNorm(1e-2, 1e6))

            fifi.mpl_functions.adjust_spines(ax, [])
            ax.set_xlim(-0.05, 0.05)
            ax.set_ylim(-0.05, 0.05)
