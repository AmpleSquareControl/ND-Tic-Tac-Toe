import numpy as np
import sys

def makeLineDict():
	lines = {}

	axisOrder = ()
	for i in range(d):
		axisOrder = axisOrder + (i,)

	#rows and columns
	coords = np.empty(shape, dtype=object)
	for x,_ in np.ndenumerate(board):
		coords[x] = x

	for i in range(d):
		axisOrder = axisOrder[1:] + (axisOrder[0],)
		transposedCoords = coords.transpose(axisOrder)
		transposedCoords = transposedCoords.reshape(n**(d-1),n)
		for j in range(len(transposedCoords)):
			lines[tuple(transposedCoords[j])] = 0

	return

def createCornerList(d):
	if d == 1:
		return ((0,),)
	subList = createCornerList(d-1)
	list0 = tuple(map(lambda x: x+(0,),subList))
	list1 = tuple(map(lambda x: x+(n-1,),subList))
	return list0+list1

if __name__ == '__main__':
	n = int(sys.argv[1])
	d = int(sys.argv[2])
	shape = ()
	for i in range(d):
		shape = shape + (n,)
	board = np.zeros(shape, dtype=int)
	makeLineDict()