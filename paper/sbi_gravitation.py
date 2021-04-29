import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.collections import LineCollection

def get_observation_points(traj):
    return traj[19:101:20]

# colorline code from matplotlib examples https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, linestyle=None, label=None):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, linestyle=linestyle, label=label)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

class GravitySimulator:
    def __init__(self, m1=45, m2=33, m3=60, v0=[50., 0.], G=1.90809e5, dt=1e-3, nsteps=100, random_noise=False, noise_size=3.):
        # always start at origin
        self.m0 = 1.
        self.m1, self.m2, self.m3, self.v0, self.G, self.dt, self.nsteps = np.array(m1), np.array(m2), np.array(m3), np.array(v0), G, dt, nsteps
        self.masses = [self.m1, self.m2, self.m3]
        self.positions = np.zeros([self.nsteps, 2])
        self.attractor_positions = np.array([[20., 20.], [50., -15.], [80., 25.]])
        # first step special case
        self.positions[1] = self.positions[0] + self.v0 * self.dt + 0.5 * self.dt**2 * self.A(self.positions[0])
        self.iter_idx = 2
        self.random_noise = random_noise
        self.noise_size = noise_size
        

    def rsquare(self, x1, x2):
        # square of distance between two points --> only square dist matters for gravity
        return np.linalg.norm(x1 - x2)

    def A(self, x):
        '''Take the position of the small particle, x, and return 
           the sum of forces on it from the three attractors.'''
        # acceleration = Force/mass
        # F = G * m1 * m2 / r^2 + R(t) --> add random noise to force
        forces = np.zeros([3, 2])
        for i, mass in enumerate(self.masses):
            # since the small particle has unit mass, just G * m
            dist = self.rsquare(x, self.attractor_positions[i])
            force = self.G * mass / dist
            unit_vec = (self.attractor_positions[i] - x) / dist
            # point the force in the correct direction (attractive)
            force *= unit_vec
            forces[i] = force
        # sum up the three force vectors
        return np.sum(forces, axis=0)
    
    def run(self):
        np.random.seed(12656)
        while(self.iter_idx < self.nsteps):
            self.step()
        if self.random_noise:
            self.positions = np.random.normal(self.positions, self.noise_size)
        return self.positions

    def step(self):
        # single step of integration with velocity verlet
        last_last_x = self.positions[self.iter_idx-2] 
        last_x = self.positions[self.iter_idx-1]
        self.positions[self.iter_idx] = 2 * last_x - last_last_x + self.A(last_x) * self.dt**2
        self.iter_idx += 1

    def plot_traj(self,
                  name='trajectory.png',
                  fig=None,
                  axes=None,
                  save=True,
                  make_colorbar=False,
                  alpha=0.5,
                  cmap=plt.get_cmap('Blues').reversed(),
                  color='blue',
                  fade_lines=True,
                  linestyle='-',
                  linewidth=2,
                  label=None,
                  label_attractors=False):
        if fig is None and axes is None:
            fig, axes = plt.subplots()
        x, y =self.positions[:,0], self.positions[:,1]
        if fade_lines:
            lc = colorline(x, y, alpha=alpha, cmap=cmap, linestyle=linestyle, linewidth=linewidth, label=label)
        else:
            axes.plot(x, y, alpha=alpha, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        if make_colorbar:
            fig.colorbar(lc)
        xmin = min(x.min(), np.min(self.attractor_positions[0,:]) - 0.1 * abs(np.min(self.attractor_positions[0,:])))
        xmax = max(x.max(), np.max(self.attractor_positions[0,:]) + 0.1 * np.max(self.attractor_positions[0,:]))
        plt.xlim(xmin, xmax)
        ymin = min(y.min(), np.min(self.attractor_positions[1,:]) - 0.1 * abs(np.min(self.attractor_positions[1,:])))
        ymax = max(y.max(), np.max(self.attractor_positions[1,:]) + 0.1 * np.max(self.attractor_positions[1,:]))
        plt.ylim(ymin, ymax)
        axes.scatter(self.attractor_positions[:,0],
                    self.attractor_positions[:,1],
                    color='black',
                    label=('Attractors' if label_attractors else None))
        if save:
            plt.savefig(name)

    def set_traj(self, trajectory):
        self.positions = trajectory

def sim_wrapper(params_list):
        '''params_list should be: m1, m2, m3, v0[0], v0[1] in that order'''
        m1, m2, m3 = float(params_list[0]), float(params_list[1]), float(params_list[2])
        v0 = np.array([params_list[3], params_list[4]], dtype=np.float64)
        this_sim = GravitySimulator(m1, m2, m3, v0, random_noise=True)
        this_traj = this_sim.run()
        summary_stats = torch.as_tensor(get_observation_points(this_traj).flatten())
        return summary_stats