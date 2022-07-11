from sigmarhotau import *
import os

path = 'game/'

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")

### 102 puzzles in total
for puzzle_n, puzzle in enumerate(puzzlesDictionary['puzzles'][:]): # [:34], [34:68], [68:]

    myFEN = puzzle['fen']
    simplifiedFEN = "".join(myFEN.replace("/",".").split())
    print("########### STARTING FEN: ", myFEN)

    myBoard = ChessBoard(myFEN)

    myBoard.setupRoutine()
    myBoard.calculationRoutine()

    myAnalysis = Analysis(myBoard.board, myBoard.pieces, myBoard.pvMoveList)

    myAnalysis.setupPiecesMinMaxValues()

    myAnalysis.representPerturbationValues(puzzle_n)

    myAnalysis.plotAllGraphsFAN2(myAnalysis.saliencyLists, myAnalysis.specificityLists,
                                 myAnalysis.relevanceLists, myAnalysis.perspectiveLists,
                                 puzzle_n)

    engine.close()
