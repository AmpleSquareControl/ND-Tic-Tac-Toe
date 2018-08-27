import sys
import numpy as np
from operator import add
import pdb
import time
from scipy.stats import pearsonr

"""
Run with Python 3.6.4. Command line arguments are: n d player1 player2 #games printout(0 or 1)
"""

global board #The current board position
global turn #Boolean representing player to move. 0 = player1, 1 = player2 
global squareMap #For analytics. Counts the frequency a square is played on by player 2

#Checks to see if the last move by player_number is a winning one
def checkWin(move, player_number):
	#check rows and cols: hold (d-1) dimensions constant and vary the last for each dimension
	n = board.shape[0]
	d = len(board.shape)
	for axis in range(d):
		lines = np.all(board == player_number, axis=axis)
		if np.any(lines):
			return True

	#diagonals, check the main diagonals: 2^(d-1). keep a sum along those diagonals, check for n and -n.
	return checkDiagonals(board, player_number, d, move)

#Helper function for checkWin. Checks to see if move creates a valid tic tac toe on a diagonal
def checkDiagonals(brd, player_number, d, move):
	#Fix the first coordinate so we don't check each diagonal twice
	if len(brd.shape) == 1:
		return False

	didntFindWin = False
	cornerList = createCornerList(d)
	for i in range(len(cornerList)):
		square = cornerList[i]
		increments = ()
		for j in range(d):
			if square[j] == 0:
				increments += (1,)
			else:
				increments += (-1,)

		didntFindWin = False
		for k in range(n):
			if brd[square] != player_number:
				didntFindWin = True
				break
			square = tuple(map(add, square, increments))
		if not didntFindWin:
			return True

	#recursively check smaller subspaces
	for i in range(d):
		foundSubspaceWin = checkDiagonals(np.rollaxis(brd,i)[move[i]], player_number, d-1, move)
		if foundSubspaceWin:
			return True

	return False

def createCornerList(d):
	if d == 1:
		return ((0,),)
	subList = createCornerList(d-1)
	list0 = tuple(map(lambda x: x+(0,),subList))
	list1 = tuple(map(lambda x: x+(n-1,),subList))
	return list0+list1

#assumes already checked for win
def checkTie():
	return np.any(board == 0)

#Returns the move that extends the longest unopposed line
def maxInRow():
	d = len(board.shape)
	maxFound = 0
	bestMove = None
	for i in range(d):
		opposedAxes = np.any(board == (not turn) + 1, axis=i)
		axisSums = np.sum(board,axis=i)
		if turn:
			axisSums = axisSums / 2

		for axis, _ in np.ndenumerate(opposedAxes):
			if not opposedAxes[axis]:
				#the jth vector along axis i is unopposed by the other player
				if axisSums[axis] > maxFound:
					maxFound = axisSums[axis]

					move = list(axis)
					move.insert(i, np.random.randint(0,n))
					move = tuple(move)

					while checkInvalid(move):
						move = list(axis)
						move.insert(i, np.random.randint(0,n))
						move = tuple(move)
					bestMove = move

	diagonalMove, diagonalMax = maxInDiagonals(board, turn + 1, d)

	if diagonalMax > maxFound:
		bestMove = diagonalMove

	if bestMove:
		return bestMove

	move = randomBot()
	while(checkInvalid(move)):
		move = randomBot()

	return move

#Returns the move that lengthens the longest unopposed diagonal and the new unopposed length.
def maxInDiagonals(brd, player_number, d):
	if len(brd.shape) == 1:
		return None, -1

	cornerList = createCornerList(d)

	isOpposed = False
	move = None
	maxLength = -1
	for i in range(len(cornerList)):
		square = cornerList[i]
		increments = ()
		for j in range(d):
			if square[j] == 0:
				increments += (1,)
			else:
				increments += (-1,)

		isOpposed = False
		length = 0
		for k in range(n):
			if brd[square] == (not (player_number - 1)) + 1:
				isOpposed = True
				break
			length = length + brd[square]
			square = tuple(map(add, square, increments))

		if player_number == 2:
			length = length / 2

		if not isOpposed:
			if length > maxLength:
				rand = np.random.randint(0, n)
				move = cornerList[i]
				for h in range(rand):
					move = tuple(map(add, move, increments))
				
				while brd[move] != 0:
					rand = np.random.randint(0, n)
					move = cornerList[i]
					for h in range(rand):
						move = tuple(map(add, move, increments))
				maxLength = length

	for i in range(d):
		for j in range(n):
			subMove, subMaxLength = maxInDiagonals(np.rollaxis(brd,i)[j], player_number, d-1)
			if subMaxLength > maxLength:
				move = list(subMove)
				move.insert(i, j)
				move = tuple(move)
				maxLength = subMaxLength

	return move, maxLength

#Calculates the value of the board according to the objective function.
def boardValue(brd):
	d = len(brd.shape)
	n = board.shape[0]

	totalValue = 0
	for player_number in [1,2]:
		for i in range(d):
			opposedAxes = np.any(brd == (not (player_number - 1)) + 1, axis=i)
			axisSums = np.sum(brd,axis=i)
			if player_number == 2:
				axisSums = axisSums / 2

			for axis, _ in np.ndenumerate(opposedAxes):
				if (not opposedAxes[axis]) and axisSums[axis] > 0:
					#the jth vector along axis i is unopposed by one of the players
					if player_number == 1:
						if axisSums[axis] == n:
							totalValue = totalValue + 10**10
						else:
							totalValue = totalValue + 10**axisSums[axis]
					else:
						if axisSums[axis] == n:
							totalValue = totalValue - 10**10
						else:
							totalValue = totalValue - 10**axisSums[axis]

	totalValue = totalValue + diagonalValue(brd, d)

	return totalValue

#Helper function for boardValue to calculate the value along diagonals.
def diagonalValue(brd, d):
	if len(brd.shape) == 1:
		return 0

	n = board.shape[0]

	cornerList = createCornerList(d)

	isOpposed = False
	totalValue = 0
	for player_number in [1, 2]:
		for i in range(len(cornerList)):
			square = cornerList[i]
			increments = ()
			for j in range(d):
				if square[j] == 0:
					increments += (1,)
				else:
					increments += (-1,)

			isOpposed = False
			length = 0
			for k in range(n):
				if brd[square] == (not (player_number - 1)) + 1:
					isOpposed = True
					break
				length = length + brd[square]
				square = tuple(map(add, square, increments))

			if player_number == 2:
				length = length / 2

			if (not isOpposed) and length > 0:
				if player_number == 1:
					if length == n:
						totalValue = totalValue + 10**10
					else:
						totalValue = totalValue + 10**length

				else:
					if length == n:
						totalValue = totalValue - 10**10
					else:
						totalValue = totalValue - 10**length


	for i in range(d):
		for j in range(n):
			totalValue = totalValue + diagonalValue(np.rollaxis(brd,i)[j], d-1)

	return totalValue

#Plays a random legal move.
def randomBot():
	n = board.shape[0]
	d = len(board.shape)

	move = np.random.randint(0, n, size=d)

	return tuple(move)

def humanGame():
	move = input("P{} coords: ".format(turn+1))
	if sys.version_info[0] == 3:
		move = move.split(',')
		for i in range(len(move)):
			move[i] = int(move[i])
		move = tuple(move)

		return move
	else:
		return move

#Bot always blocks the opponents longest unopposed line. Prioritizes blocking diagonals.
def blockingBot():
	d = len(board.shape)
	maxFound = 0
	bestMove = None
	for i in range(d):
		opposedAxes = np.any(board == turn + 1, axis=i)
		axisSums = np.sum(board,axis=i)
		if not turn:
			axisSums = axisSums / 2

		for axis, _ in np.ndenumerate(opposedAxes):
			if not opposedAxes[axis]:
				#the jth vector along axis i is unopposed by the current player
				if axisSums[axis] > maxFound:
					maxFound = axisSums[axis]

					move = list(axis)
					move.insert(i, np.random.randint(0,n))
					move = tuple(move)

					while checkInvalid(move):
						move = list(axis)
						move.insert(i, np.random.randint(0,n))
						move = tuple(move)
					bestMove = move

	diagonalMove, diagonalMax = maxInDiagonals(board, (not turn) + 1, d)

	if diagonalMax >= maxFound:
		bestMove = diagonalMove

	if bestMove:
		return bestMove

	move = randomBot()
	while(checkInvalid(move)):
		move = randomBot()

	return move

def boardValueBot():
	bestMove = None
	bestValue = None

	modifiedBoard = np.copy(board)

	for move, _ in np.ndenumerate(board):

		if board[move] == 0:
		
			if turn:
				modifiedBoard[move] = 2

				value = boardValue(modifiedBoard)
				if bestValue == None or value < bestValue:
					bestValue = value
					bestMove = move

			else:
				modifiedBoard[move] = 1

				value = boardValue(modifiedBoard)
				if bestValue == None or value > bestValue:
					bestValue = value
					bestMove = move

			modifiedBoard[move] = 0

	return bestMove

#Used by minimax to recurse and evaluate moves. Returns the best move and its value.
def bestValueMove(brd, depth, player_number):
	bestMove = None
	bestValue = None

	modifiedBoard = np.copy(brd)

	for move, _ in np.ndenumerate(brd):

		if brd[move] == 0:
		
			if player_number == 2:
				modifiedBoard[move] = 2

				currentValue = boardValue(modifiedBoard)

				if currentValue < -10**9:
					return move, currentValue

				if depth == 1:
					value = currentValue
				else:
					_, value = bestValueMove(modifiedBoard, depth-1, 1)

				if bestValue == None or value < bestValue:
					bestValue = value
					bestMove = move
				elif value == bestValue:
					coin = np.random.randint(0,2)
					if coin:
						bestMove = move


			else:
				modifiedBoard[move] = 1

				currentValue = boardValue(modifiedBoard)

				if currentValue > 10**9:
					return move, currentValue

				if depth == 1:
					value = currentValue
				else:
					_, value = bestValueMove(modifiedBoard, depth-1, 2)

				if bestValue == None or value > bestValue:
					bestValue = value
					bestMove = move
				elif value == bestValue:
					coin = np.random.randint(0,2)
					if coin:
						bestMove = move

			modifiedBoard[move] = 0

	if bestValue == None:
		bestValue = boardValue(brd)

	return bestMove, bestValue

#Performs game tree search to depth=depth. Returns best move.
def miniMaxBot(depth):
	move, value = bestValueMove(board, depth, turn + 1)

	return move


def checkInvalid(move):
	return move == None or board[move] != 0

def play1Game(p1, p2, depth1=None, depth2=None):
	global board
	global turn
	global squareMap
	bots = {'human':humanGame, 'random':randomBot, 'max':maxInRow, 'block':blockingBot, 'value':boardValueBot, 'minimax':miniMaxBot}

	turn = 0
	shape = ()
	for i in range(d):
		shape = shape + (n,)

	board = np.zeros(shape,dtype=int)

	if p1=='human' or p2=='human':
		print(board)

	move = None
	num_turns = 0
	avgp1_time = 0
	avgp2_time = 0
	while checkTie():
		if p1=='human' or p2=='human':
			print("\n------------------------\n")
			if turn:
				print(p2 + " move:\n")
			else:
				print(p1 + " move:\n")

		while checkInvalid(move):
			if turn:
				start = time.time()
				if p2 == "minimax":
					move = bots[p2](depth2)
				else:
					move = bots[p2]()
				if p2 == "human" and board[move] != 0:
					print("Invalid Move")

				end = time.time()
				avgp2_time += (end-start)
				squareMap[move] += 1
			else:
				start = time.time()
				if p1 == "minimax":
					move = bots[p1](depth1)
				else:
					move = bots[p1]()
				if p1 == "human" and board[move] != 0:
					print("Invalid Move")
				end = time.time()
				avgp1_time += (end-start)

		if turn:
			board[move] = 2
		else:
			board[move] = 1

		if p1=='human' or p2=='human':
			print(board)

		num_turns += 1
		if checkWin(move, turn+1):

			if p1=='human' or p2=='human':
				print(board)

			avgp1_time /= num_turns
			avgp2_time /= num_turns
			print("winner: player {}".format(turn+1))
			if print_out:
				print("p1_time: {},p2_time: {}, #turns: {}".format(avgp1_time, avgp2_time,num_turns))
			return turn+1, num_turns, avgp1_time, avgp2_time

		turn = not turn

	if p1=='human' or p2=='human':
		print(board)

	print('Tie Game')
	avgp1_time /= num_turns
	avgp2_time /= num_turns
	if print_out:
		print("p1_time: {},p2_time: {}, #turns: {}".format(avgp1_time, avgp2_time,num_turns))
	return 0,num_turns, avgp1_time, avgp2_time

def main():
	global board
	global turn
	global squareMap

	shape = ()
	for i in range(d):
		shape = shape + (n,)
	squareMap = np.zeros(shape,dtype=int)
	p1 = sys.argv[3]
	p2 = sys.argv[4]
	num_games = int(sys.argv[5])

	p1_wins = 0
	p2_wins = 0
	ties = 0

	total_turns = 0 #average number of turns in a game

	depth1 = None
	depth2 = None
	if p1 == 'minimax':
		depth1 = int(input("Player 1 Minimax Depth: "))

	if p2 == 'minimax':
		depth2 = int(input("Player 2 Minimax Depth: "))

	avg_p1_times = []
	avg_p2_times = []
	num_turns = []
	for _ in range(num_games):
		winner, turns, p1_time, p2_time = play1Game(p1,p2,depth1,depth2) #0 is tie, 1 is p1, 2 is p2
		avg_p1_times.append(p1_time+np.random.uniform(1e-9,1e-10))
		avg_p2_times.append(p2_time+np.random.uniform(1e-9,1e-10))
		num_turns.append(turns+np.random.uniform(1e-9,1e-10))
		total_turns += turns
		if winner == 0:
			ties+=1
		elif winner == 1:
			p1_wins+=1
		else:
			p2_wins+=1
		

	print("{}: {},{}: {},Ties: {}".format(p1, p1_wins, p2, p2_wins, ties))
	if print_out:
		print(pearsonr(avg_p1_times,num_turns))
		print(pearsonr(avg_p2_times,num_turns))

		avg_num_turns = total_turns / num_games
		print("Avg #turns: {}".format(avg_num_turns))

		print("p1 avg time: {}, p2 avg time: {}".format(np.mean(avg_p1_times),np.mean(avg_p2_times)))
		print(squareMap)

if __name__ == '__main__':
	#args: n d player1 player2 num_games print_out
	n = int(sys.argv[1])
	d = int(sys.argv[2])
	print_out = int(sys.argv[6])
	main()
	
