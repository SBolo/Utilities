import numpy as np

def detect_folding( arr, high_cutoff, low_cutoff, eps ):

	dim = len(arr)

	time_low = 0
	paths = [ ]
	time = 0

	while( time < dim ):

		item = arr[time]

		if( item < low_cutoff + eps and item > low_cutoff - eps ):

			time_low = time
			print("# FOUND! Low cutoff reached at time ", time_low)

			counter = 1
			for jtem in arr[time_low+1:]:

				if( jtem < low_cutoff + eps and jtem > low_cutoff - eps ):
					print("# SIGH! etected backward transition at time", time_low + counter )
					print()
					time = time_low + counter - 1
					break

				elif( jtem < high_cutoff + eps and jtem > high_cutoff - eps ):
					print("# GREAT! Folded state reached at ", time_low + counter )
					print()
					paths.append( [ time_low, time_low + counter ] )
					time = time_low + counter - 1
					break

				counter += 1

		time += 1

	return paths

def detect_unfolding( arr, high_cutoff, low_cutoff, eps ):

	dim = len(arr)

	time_high = 0
	paths = [ ]
	time = 0

	while( time < dim ):

		item = arr[time]

		if( item < high_cutoff + eps and item > high_cutoff - eps ):

			time_high = time
			print("# FOUND! High cutoff reached at time ", time_high)

			counter = 1
			for jtem in arr[time_high+1:]:

				if( jtem < high_cutoff + eps and jtem > high_cutoff - eps ):
					print("# SIGH! Upward transition detected at time", time_high + counter )
					print()
					time = time_high + counter - 1
					break

				elif( jtem < low_cutoff + eps and jtem > low_cutoff - eps ):
					print("# GREAT! Folded state reached at ", time_high + counter )
					print()
					paths.append( [ time_high, time_high + counter ] )
					time = time_high + counter - 1
					break

				counter += 1
		time += 1

	return paths

if( __name__ == '__main__' ):

	high_cutoff = 0.9
	low_cutoff = 0.1
	eps = 0.03

	print("# WAIT! Loading array ... ")
	arr = np.loadtxt("Qs.txt")[:,1]

	##### FOLDING PATHWAYS
	fpaths = detect_folding( arr, high_cutoff, low_cutoff, eps )
	upaths = detect_unfolding( arr, high_cutoff, low_cutoff, eps )

	print()
	print("# SUMMARY")
	print("---------")

	npaths = len(fpaths)
	print("# Number of folding events detected: ", npaths)
	print( "# Transition paths times: ", fpaths )

	##### UNFOLDING PATHWAYS


	npaths = len(upaths)
	print("# Number of unfolding events detected: ", npaths)
	print( "# Transition paths times: ", upaths )



"""
dim = len(arr)

time_low = 0
paths = [ ]
time = 0

while( time < dim ):

	item = arr[time]

	if( item < low_cutoff + eps and item > low_cutoff - eps ):

		time_low = time
		print("# Low cutoff reached at time ", time_low)

		counter = 1
		for jtem in arr[time_low+1:]:

			if( jtem < low_cutoff + eps and jtem > low_cutoff - eps ):
				print("# Detected backward transition at time", time_low + counter )
				print()
				time = time_low + counter - 1
				break

			elif( jtem < high_cutoff + eps and jtem > high_cutoff - eps ):
				print("# GREAT! Folded state reached at ", time_low + counter )
				print()
				paths.append( [ time_low, time_low + counter ] )
				time = time_low + counter - 1
				break

			counter += 1
	#else:
	time += 1
"""


