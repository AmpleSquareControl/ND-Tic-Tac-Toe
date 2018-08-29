import numpy as np
import sys
from operator import add

"""
Produces and returns all valid tic-tac-toe lines for a board size

@return: List of sets. Each set contains the coordinates of a line
"""
def getAllLines():
	lines = []

	#straight lines along a dimension
	coords = np.empty(shape, dtype=object)
	for x,_ in np.ndenumerate(board):
		coords[x] = x

	axisOrder = ()
	for i in range(d):
		axisOrder = axisOrder + (i,)

	for i in range(d):
		axisOrder = axisOrder[1:] + (axisOrder[0],)
		transposedCoords = coords.transpose(axisOrder)
		transposedCoords = transposedCoords.reshape(n**(d-1),n)
		for j in range(len(transposedCoords)):
			lines.append(set(transposedCoords[j]))

	"""
	Diagonals for subspace of dimension d. coordIndList is a list of indices to add
	coordinates from the original hypercube space. coordList contains the values to add
	at these indices
	"""
	def getDiagonals(subD, coordIndList, coordList):
		if subD == 1:
			return

		cornerList = createCornerList(n, subD) #Only corners with first coord = 0 to avoid double counting
		for i in range(len(cornerList)):
			subSquare = cornerList[i]
			increments = () #direction of hypercube diagonal associated with this corner
			for j in range(subD):
				if subSquare[j] == 0:
					increments += (1,)
				else:
					increments += (-1,)

			subDiagonal = set()
			for _ in range(n):
				subDiagonal.add(subSquare)
				subSquare = tuple(map(add, subSquare, increments))

			diagonal = set()
			for subSquare in subDiagonal:
				ss = list(subSquare)
				for j in range(len(coordList)):
					ss.insert(coordIndList[j], coordList[j])
				diagonal.add(tuple(ss))

			lines.append(diagonal)

		#Get diagonals in subspaces
		lastInd = -1
		if len(coordIndList) > 0:
			lastInd = coordIndList[-1]

		if subD > 2:
			for i in range(lastInd+1,d):
				for j in range(n):
					getDiagonals(subD-1,coordIndList + [i],coordList + [j])

	getDiagonals(d, [], [])
	return lines

"""
Creates a mapping from each square on the board to the indices of the
tic-tac-toe lines in global 'lines' that it lies on

@return: Dictionary with n^d keys corresponding to squares and values of lists
		 of indices in global 'lines'
"""
def createSquareToLineMapping():
	linesForSquare = {}
	for x,_ in np.ndenumerate(board):
		lineIndices = []
		for i in range(len(lines)):
			if x in lines[i]:
				lineIndices.append(i)
		linesForSquare[x] = lineIndices

	return linesForSquare

"""
For a d-dimensional hypercube of edge length n, return coordinates of corners with
first coordinate equal to 0.

@param n: hypercube edge length
@param d: hypercube dimension
@return: tuple of corner coordinates
"""
def createCornerList(n, d):
	if d == 1:
		return ((0,),)
	subList = createCornerList(n, d-1)
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
	lines = getAllLines()
	linesForSquare = createSquareToLineMapping()
	lineControllers = np.zeros(len(lines)) #Player number that controls a line, 0 if no markers
	movesOnLine = np.zeros(len(lines)) #Number of pieces a player has on a line