import numpy as np
import matplotlib.pyplot as plt
import sys

# physical potential: symmetric double well
def potential( x ):
	return ( x**2 - 1 )**2
#------------------------------------------------------------------------------------------

# physical force corresponding to the symmetric double well
def force( x ):
	return - 4. * x * ( x**2 - 1 )
#------------------------------------------------------------------------------------------

# evolution of a single Labgevin step
def langevin_metad_step( x, dt, beta, centers = [], h = 0., sigma2 = 0. ):
	return x + dt * ( force(x) + metad_force( x, centers, h, sigma2) ) + np.random.normal() * np.sqrt( (2./beta) * dt )
#------------------------------------------------------------------------------------------

# metad potential
def metad_potential( x, centers, h, sigma2 ):

	potential = []

	if centers == []:
		return np.zeros( len(x) )
	else:
		for i in x:
			potential.append( np.sum( np.exp( -(i - centers)**2 / sigma2 ) ) )

	return h * np.array(potential)
#------------------------------------------------------------------------------------------

# metad force: - gradient of metad potential
def metad_force( x, centers, h, sigma2 ):

	if centers == []:
		return 0
	else:
		tmp = x - centers
		force = np.sum( tmp * np.exp( - tmp**2 / sigma2 ) )

	return h * force / (0.5 * sigma2)
#------------------------------------------------------------------------------------------

# animate the dynamics by plotting the physical potential, the metad bias and the motion of the walkers
def animate_pbmetad( x, positions, centers = [], h = 0., sigma2 = 0. ):

	color = plt.cm.rainbow(np.linspace(0, 1, len(positions)))
	plt.clf()
	plt.xlim(-1.5, 1.5)
	plt.ylim(0, 1.6)
	plt.plot( x, potential(x), 'k-', linewidth = 3, zorder = 0 )
	plt.fill_between(x, potential(x), potential(x) + metad_potential(x, centers, h, sigma2), color='tab:blue', zorder=-1)

	for n,p in enumerate(positions):
		plt.scatter( p, potential(p) + metad_potential([p], centers, h, sigma2), color = color[n], s = 150, edgecolors = 'k', zorder = 10 )

	plt.pause(0.00001)
	plt.draw()
#------------------------------------------------------------------------------------------

def main():

	#PARAMETERS
	nwalkers  = 4                                    # number of walkers
	positions = np.random.uniform( -1, 1, nwalkers ) # sample uniformly the initial walker positions
	dt        = 1e-2                                 # integration timestep
	beta      = 3                                    # inverse temperature
	tot_steps = 1e3                                  # total number of timesteps
	interact  = True                                 # enables the interactive display of the simulation

	#METAD PARAMETERS
	pace      = 100                                  # how many steps to wait before depositing a Gaussian
	height    = 0.1                                  # heights of the Gaussian
	sigma     = 0.1                                  # sigma of the gaussian

	# CALCULATION
	x         = np.arange( -2., 2., 0.01 ) #only for potential plotting
	trajs     = []
	centers   = [] #shared by all the walkers
	sigma2    = 2. * sigma**2
	t         = 0

	while t <= tot_steps:

		# update the position of every walker
		for n in range(nwalkers):
			positions[n] = langevin_metad_step( positions[n], dt, beta, centers, height, sigma2 )

		# if the the time is equal to pace, deposit a Gaussian by saving the current coordinate as a center
		# this is done by every walker in the same list, so all of them will feel the same bias at any time
		if t%pace == 0:
			for n in range(nwalkers):
				centers.append( positions[n] )

		# save the trajectory of every walker
		trajs.append( positions )

		# if the calculation is interactive, plot the dynamics interactively
		if interact:
			animate_pbmetad(x, positions, centers, height, sigma2)

		# increase the time
		t += 1

if( __name__ == "__main__" ):
	main()
