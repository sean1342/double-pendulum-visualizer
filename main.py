import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
import random

g = 9.81     # acceleration due to gravity
L = 1.0      # length of pendulum
m = 1.0      # mass of pendulum
b = 0.2      # damping coefficient
tmax = 200.0  # maximum simulation time
dt = 0.01    # time step
theta0 = random.uniform(-np.pi * 3, np.pi * 3)
omega0 = random.uniform(-8, 8)

def domega_dt(t, y):
    theta, omega = y
    return [omega, -g/L * np.sin(theta) - b/m * omega]

# solve differential equation
sol = solve_ivp(domega_dt, [0, tmax], [theta0, omega0], t_eval=np.arange(0, tmax, dt))

# extract theta and omega values from solution
theta_vals = sol.y[0]
omega_vals = sol.y[1]

# create grid of starting values for phase portrait
theta0_vals = np.linspace(-np.pi * 4, np.pi * 4, 32)
omega0_vals = np.linspace(-20.0, 20.0, 32)
theta0_grid, omega0_grid = np.meshgrid(theta0_vals, omega0_vals)

# calculate values of omega and theta for each starting value
dx = omega0_grid
dy = domega_dt(0, [theta0_grid, omega0_grid])[1]
color = np.sqrt(dx**2 + dy**2)
dx_norm = dx / color
dy_norm = dy / color

# plot phase portrait with arrows and trace of pendulum
# magnitude is represented with color instead of length
fig, ax = plt.subplots(figsize=(8,6))
plt.quiver(theta0_grid, omega0_grid, dx_norm, dy_norm, color, cmap='viridis', alpha=0.8)
plt.xlabel('Starting angle (rad)')
plt.ylabel('Starting angular velocity (rad/s)')
plt.title('Phase Portrait for Damped Pendulum')

line, = ax.plot([], [], 'b')
def animate(i):
    line.set_data(theta_vals[:i], omega_vals[:i])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(theta_vals), interval=dt*1000, blit=True)

fig2, ax2 = plt.subplots()
ax2.set_xlim(-1.1*L, 1.1*L)
ax2.set_ylim(-1.1*L, 1.1*L)
ax2.set_aspect('equal')
pendulum, = ax2.plot([], [], 'o-', lw=2)

def update_pendulum(i):
    x = L*np.sin(theta_vals[i])
    y = -L*np.cos(theta_vals[i])
    pendulum.set_data([0, x], [0, y])
    return pendulum,

ani2 = animation.FuncAnimation(fig2, update_pendulum, frames=len(theta_vals), interval=dt*1000, blit=True)

plt.show()
