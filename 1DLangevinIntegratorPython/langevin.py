import numpy as np
import matplotlib.pyplot as plt
import sys

def potential( x ):
	return ( x**2 - 1 )**2

def force( x ):
	return - 4. * x * ( x**2 - 1 )

#evolution of a single step
def langevin_step( x, dt, beta ):
	return x + dt * force(x) + np.random.normal() * np.sqrt( (2./beta) * dt )

#normalized Boltzmann distribution
def boltzmann( x, beta, d ):
	return np.exp( - beta * potential(x) ) / np.trapz( np.exp( - beta * potential(x)), dx = d )

#animates the particle motion on the potential
def animate( x, xt ):

	plt.clf()
	plt.xlim(-1.5, 1.5)
	plt.ylim(0, 1.6)
	plt.plot( x, potential(x), 'k-', linewidth = 2 )
	plt.scatter( xt, potential(xt), c = 'r', s = 100 )
	plt.pause(0.00001)
	plt.draw()

#animates the evolution of the probability distribution
def animate_distribution( x, beta, trajectory ):

	plt.clf()
	plt.xlim(-1.5, 1.5)
	plt.ylim(0, 2.5)
	plt.plot( x, boltzmann(x, beta, 0.01), 'r-', linewidth = 2 )
	plt.plot( x, potential(x), 'k--' )
	plt.hist( trajectory, bins = 50, density = True )
	plt.pause(0.00001)
	plt.draw()

#runs some simple analyses to check the consistency of results
def analyze( trajectory, K, beta, x ):

	print(f"Simulated Temperature:  {2. * np.mean(K)}")
	print(f"Expected Temperature:   {1./beta}")
	print(f"Simulated Fluctuations: {np.std(K)}")
	print(f"Expected Fluctuations:  {np.sqrt( 1./ (2. * beta**2))}")

	plt.figure(10)
	plt.plot( trajectory, 'k-' )

	plt.figure(20)
	plt.hist( trajectory, bins = 100, density = True, label = 'Sampled Distribution' )
	plt.plot( x, boltzmann(x, beta, 0.01), 'r-', label = 'Exact Distribution' )

	plt.legend()
	plt.show()

def main():

	#PARAMETERS
	x0 = -1
	dt = 1e-2
	beta = 0.1
	tot_steps = 1e4
	interactive = False #enables the interactive display of the simulation
	equilibrium = True #enables the interactive display of the transition towards equilibrium
	analysis = True #enables the analyses

	#AUXILIARY QUANTITIES
	time = np.arange( 0, tot_steps, dt )
	x = np.arange( -2., 2., 0.01 )
	trajectory = []
	K = []

	print("\nOVERDAMPED LANGEVIN DYNAMICS")
	print("----------------------------")

	#EXECUTION
	for t in time:

		#evolve in time
		xt = langevin_step( x0, dt, beta )

		# stochastic velocity is not defined:
		# we use x^2/2t to define the kinetic energy
		K.append( (xt-x0)**2/(4. * dt) )

		#reset position
		x0 = xt

		#save frames
		trajectory.append(xt)

		#enables interactive trajectory
		if interactive and not equilibrium:
			animate( x, xt )

		if equilibrium and not interactive:
			animate_distribution( x, beta, trajectory )

		if equilibrium and interactive:
			print("Please disable 'interactive' or 'equilibrium' first")
			print("----------------------------\n")
			sys.exit(0)

	#enables analysis of results
	if analysis:
		analyze( trajectory, K, beta, x )

	print("----------------------------\n")

if( __name__ == "__main__" ):
	main()
