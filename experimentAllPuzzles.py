from sigmarhotau import *
import os

path = 'game/'

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")

### 102 puzzles in total
for puzzle_n, puzzle in enumerate(puzzlesDictionary['puzzles'][:1]): # [:34], [34:68], [68:]

    myFEN = puzzle['fen']
    simplifiedFEN = "".join(myFEN.replace("/",".").split())


    myBoard = ChessBoard(myFEN)

    myBoard.executionRoutine()

    myAnalysis = Analysis(myBoard.board, myBoard.pieces, myBoard.pvMoveList)
    # print("BLE")
    myAnalysis.setupPiecesMinMaxValues()
    # print("CIAO")
    myAnalysis.representPerturbationValues(puzzle_n)

    # svg_board = chess.svg.board(myBoard.board)
    # board_png = cairosvg.svg2png(svg_board)
    # board_image = Image.open(BytesIO(board_png))
    #
    # board_png.paste(sigmarhotau_overlay_image, (0, 0), sigmarhotau_overlay_image).save(path+simplifiedFEN+".png")

    # print("BELLA")
    # base_board_png = returnConvertedBoardToPNG(myAnalysis.board, path)

    # FEN_name = simplifiedFEN
    # join2PNG(FEN_name, base_board_png, sigmarhotau_overlay_png, path)

    engine.close()
