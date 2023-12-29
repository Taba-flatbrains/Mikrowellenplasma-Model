import macromax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

m = 1
cm = m*0.01
mm = m*0.001
mym = mm*0.001
nm = mym*0.001

class Microwave:
    height = 15*cm
    width = 25*cm
    length = 15*cm
    wall_thickness = 1*mm
    shape = (length, height, width)

class Antenna:
    height = 2.5 * cm
    antenna_radius = 3*mm/2
    base_radius = 6.5*cm/2
    base_height = 0.5*cm
    x_pos = 10*cm
    y_pos = 0.5*cm
    z_pos = 10*cm

source_polarization = np.array([1, 1, 0])[:, np.newaxis, np.newaxis, np.newaxis]
data_shape = (120,120,180)
grid = macromax.Grid(data_shape, extent=Microwave.shape)

"""
epsilon = permittivity
relative permittivity:
air: 1,00059 also 1
copper: 10^6 - 10^8

mu = permeability
relative permeability:
air: 1+4*10^-7 also 1
copper: 1-6*10^7 also 1
"""

permittivity = np.ones(data_shape) * 10 ** 5  # permittivity of steel
permittivity[1:-1, 1:-1, 1:-1] = 1  # set air to 1

X, Y, Z = np.ogrid[:data_shape[0], :data_shape[1], :data_shape[2]]  # a
distance_to_center = np.sqrt((((data_shape[0]/2) - X) * grid.step[0]) ** 2 + (((data_shape[2]/2) - Z) * grid.step[2]) ** 2) + (Y*0)
is_copper = np.zeros(data_shape)
is_copper[:,int(Antenna.y_pos/grid.step[1]):int((Antenna.y_pos+Antenna.base_height)/grid.step[1]),:] = distance_to_center[:,int(Antenna.y_pos/grid.step[1]):int((Antenna.y_pos+Antenna.base_height)/grid.step[1]),:] < Antenna.base_radius # base
is_copper[:,int((Antenna.y_pos+Antenna.base_height)/grid.step[1]):int((Antenna.y_pos+Antenna.base_height+Antenna.height)/grid.step[1]),:] = distance_to_center[:,int((Antenna.y_pos+Antenna.base_height)/grid.step[1]):int((Antenna.y_pos+Antenna.base_height+Antenna.height)/grid.step[1]),:] < Antenna.antenna_radius

permittivity += is_copper * (10 ** 6)
conductivity = permittivity - 1 + 10e-12  # not actually true but more simple


current_density = np.zeros(data_shape)
current_density[round(data_shape[0]/5*2):round(data_shape[0]/5*3), round(data_shape[1]/5*2):round(data_shape[1]/5*3), -2] = 1  # waveguide exit shape: 1/5 * 1/5 microwave xy edge
current_density = current_density*source_polarization

callback = lambda s: s.iteration < 3e3 and s.residue > 1e-4
solution = macromax.solve(grid, vacuum_wavelength=12.2*cm, epsilon=permittivity, current_density=current_density, callback=callback)
J = solution.E * conductivity
for i in range(20):
    print(i)
    J = (J + solution.E * conductivity) / 2
    J[[0, 1], round(data_shape[0]/5*2):round(data_shape[0]/5*3), round(data_shape[1]/5*2):round(data_shape[1]/5*3), -2] += 1
    solution = macromax.solve(grid, vacuum_wavelength=12.2*cm, epsilon=permittivity, current_density=current_density, callback=callback, initial_field=solution.E)


J = solution.E * conductivity
electric_field = solution.E  # split in x,y,z field
electric_field = np.sum(electric_field, axis=0)


# J_direction_layer_to_display = np.squeeze(J[[1, 2], int(data_shape[0]/2), 2:int(data_shape[1]/4*2), int(data_shape[2]/4):int(data_shape[2]/4*3)])  # dont care about x direction
J_direction_layer_to_display = J[[1,2], int(data_shape[0]/2)]
J_direction_layer_to_display = np.real(J_direction_layer_to_display) + np.imag(J_direction_layer_to_display)
J_direction_layer_to_display -= J_direction_layer_to_display * (np.abs(J_direction_layer_to_display) < 1e-7)  # remove unwanted arrows

temp = np.zeros(J_direction_layer_to_display.shape)
temp[:, 2:int(data_shape[1]/4*2), int(data_shape[2]/4):int(data_shape[2]/4*3)] = 1
J_direction_layer_to_display = J_direction_layer_to_display*temp  # remove cage

J_direction_layer_to_display /= max(np.max(J_direction_layer_to_display), np.abs(np.min(J_direction_layer_to_display)))  # normalize highest value to one

E_layer_to_display = electric_field[int(data_shape[0]/2)]
E_layer_to_display = np.abs(np.real(E_layer_to_display) + np.imag(E_layer_to_display))

J = np.sum(J, axis=0)
J_layer_to_display = J[int(data_shape[0]/2)]  # J = current_density
J_layer_to_display = np.abs(np.real(J_layer_to_display) + np.abs(np.imag(J_layer_to_display)))

E_layer_to_display += 10e-9  # because scale is logarythmic, log 0 = inf
J_layer_to_display += 10e-9  # because scale is logarythmic, log 0 = inf

print("plotting")
fig = plt.figure(figsize=(15,10))

fig.add_subplot(2, 1, 1)
plt.imshow(E_layer_to_display, cmap='jet', interpolation='nearest', norm=LogNorm(vmin=1e-10, vmax=1))
plt.title("E")

fig.add_subplot(2, 1, 2)
plt.imshow(J_layer_to_display, cmap='jet', interpolation='nearest', norm=LogNorm(vmin=1e-10, vmax=1))
plt.quiver(np.arange(J_direction_layer_to_display.shape[2]), np.arange(J_direction_layer_to_display.shape[1]), J_direction_layer_to_display[1], J_direction_layer_to_display[0], minshaft=0.001, minlength=0, scale=5)  # z is x direction y stays y
plt.title("J")

plt.show()
