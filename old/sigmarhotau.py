from TTDef import *
import chess
import chess.engine
import chess.svg
from cairosvg import svg2png
import drawSvg
from concurrent.futures import ThreadPoolExecutor
import math
import copy
import cairosvg
import matplotlib.pyplot as plt

engine_path = r"C:\Users\molin\PycharmProjects\thesisBeta\engine\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)


def returnSimplifiedFEN(FEN):
    return "".join(FEN.replace("/", ".").split())


def getScoreType(pov_score):
    if SCORE__TYPE == 'cp':
        return pov_score.pov(POINT__OF__VIEW).score(mate_score=2000) / (1000)
    elif SCORE__TYPE == 'wdl':
        return pov_score.wdl().pov(POINT__OF__VIEW).expectation()

# Function necessary to handle pieces out of the board state (position = -1)
def removePieceAt(board_state, square_to_remove):
    if square_to_remove == -1:
        return False
    else:
        return board_state.remove_piece_at(square_to_remove)


class ChessPiece:

    def __init__(self, p_type, p_color, p_initial_position, pv_move_list):
        self.pType = p_type
        self.pColor = p_color
        self.pInitialPosition = p_initial_position
        self.pvMoveList = pv_move_list

        self.moveListAfterPerturbation = []

        self.pvBoardStateList = []
        self.pPositions = [p_initial_position]
        self.perturbedStateValuesList = []
        self.gammaValuesList = []
        self.specificityList = []
        self.relevanceList = []
        self.perspectiveList = []
        self.saliencyList = []

        self.anomalies = []
        self.pEngine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def manageAnomalies(self, anormal_state):
        self.anomalies.append({
            'is_king': True if self.pType == "King" else False,
            'is_check_to_opponent': anormal_state.is_check(),
            'is_checkmate_to_self': anormal_state.is_checkmate(),
            'is_checkmate_to_opponent': anormal_state.is_checkmate(),
            'is_stalemate': anormal_state.is_stalemate(),
            'is_insufficient_material': anormal_state.is_insufficient_material(),
            'is_valid': anormal_state.is_valid(),
            'board_status': anormal_state.status()
        })

    def calculatePerturbedQValues(self, pv_board_states_list, pv_state_values_list):
        print("starting NEW PIECE: ", self.pColor, self.pType, self.pInitialPosition)
        # print(len(pv_board_states_list), len(pv_state_values_list))
        for step, (pv_board_state, pv_state_value) in enumerate(zip(pv_board_states_list, pv_state_values_list)):
            # print("STEP:", step, "STATE VALUE: ", pv_state_value, "WHOLE LIST: ", pv_state_values_list)
            current_square = self.pPositions[step]
            # print("YES0", pv_board_state.fen())
            ### Can't manage removing -1 because it removes 63!
            # if pv_board_state.remove_piece_at(current_square):
            if removePieceAt(pv_board_state, current_square):
                # print("YES1", pv_board_state.fen())
                # if pv_board_state.is_valid(): # DEPRECATED
                if pv_board_state.status() not in boardIrregularStatus:
                    # print("YES2", pv_board_state.fen())
                    # print("PRE")
                    reference_analysis = self.pEngine.analyse(pv_board_state, DEPTH__BOUND, multipv=MULTI__PV)
                    # print("AFTER")
                    try:
                        # print("TRY")
                        self.moveListAfterPerturbation.append(reference_analysis[PV__SELECTED]['pv'][0])
                        perturbed_state_value = getScoreType(reference_analysis[PV__SELECTED]['score'])
                        self.perturbedStateValuesList.append(perturbed_state_value)
                    except:
                        # print("EXCEPTION: ")
                        print(pv_board_state.fen())
                        # print("REFERENCE ANALYSIS: ", reference_analysis)
                else:
                    # print("NO2", pv_board_state.fen())
                    self.manageAnomalies(pv_board_state)
                    self.moveListAfterPerturbation.append("Invalid state")
                    if pv_board_state.status() == chess.Status.OPPOSITE_CHECK:
                        self.perturbedStateValuesList.append(100)
                    else:
                        self.perturbedStateValuesList.append(999)

            else:
                # print("NO1", pv_board_state.fen())
                # print("PIECE CAN'T BE REMOVED. STEP: ", step, self.pType, self.pColor, current_square)
                self.manageAnomalies(pv_board_state)
                self.moveListAfterPerturbation.append("The piece is not on the board")
                self.perturbedStateValuesList.append(pv_state_value)

    def calculateGammaValues(self, pv_state_values_list):
        for base_q_value, perturbed_value in zip(pv_state_values_list, self.perturbedStateValuesList):
            self.gammaValuesList.append(base_q_value - perturbed_value)

    def calculateSpecificity(self):
        for gamma in self.gammaValuesList:
            delta_sigma = 0
            delta_sigma = math.tanh(alfa * (math.exp(abs(gamma)) - 1))
            self.specificityList.append(delta_sigma)

    def calculateRelevance(self):
        # print(len(self.pvMoveList), len(self.moveListAfterPerturbation), len(self.gammaValuesList))
        for move_original, move_post_perturbation, gamma in zip(self.pvMoveList,
                                                                self.moveListAfterPerturbation,
                                                                self.gammaValuesList):
            delta_rho = 0
            # print("ORIGINAL MOVE: ", move_original)
            # print("MOVE POST PERT: ", move_post_perturbation)
            if move_original == move_post_perturbation:
                if gamma > beta:
                    delta_rho = 1 / 2
                elif gamma < beta and gamma > -beta:
                    delta_rho = 0
                elif gamma < -beta:
                    delta_rho = -1 / 2
                else:
                    print("Impossible?")
            else:
                if gamma > beta:
                    delta_rho = 1
                elif gamma < beta and gamma > -beta:
                    delta_rho = 0
                elif gamma < -beta:
                    delta_rho = -1
                else:
                    print("Impossible?")
            # print("SELECTED RELEVANCE VALUE: ", delta_rho)

            self.relevanceList.append(delta_rho)

    def calculatePerspective(self):
        self.perspectiveList = PERSPECTIVE__VALUES[:DEPTH__PV__LIMIT + 1]

    def calculateSaliency(self):
        for delta_sigma, delta_rho, perspective_filter in zip(self.specificityList, self.relevanceList,
                                                              self.perspectiveList):
            self.saliencyList.append(delta_sigma * delta_rho * perspective_filter)


############################################################################################
class ChessBoard:
    strictFEN = 0

    def __init__(self, FEN):
        self.FEN = FEN
        self.board = chess.Board(self.FEN)
        self.pieces = {}
        self.baseReferenceAnalysis = None
        self.pvMoveList = []
        self.pvMoveSquaresList = []
        self.pvBoardStatesList = []
        self.pvStateValuesList = []

        self.piecesValues = {}

        self.baseReferenceValues = []

        self.simplePiecesValuesDictionary = {}

        self.cbEngine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # A1: GENERATE BASE BOARD ANALYSIS
    # Take board,
    # append board state in pvBoardStatesList,
    # make engine analyse board,
    # append board state Q-value to pvStateValuesList
    def generateBaseBoardAnalysis(self):
        self.pvBoardStatesList.append(self.board)
        self.baseReferenceAnalysis = self.cbEngine.analyse(self.board, DEPTH__BOUND, multipv=MULTI__PV)
        self.pvStateValuesList.append(getScoreType(self.baseReferenceAnalysis[PV__SELECTED]['score']))

    # A2: GENERATE PRINCIPAL VARIATION MOVES LIST
    # Take engine analysis,
    # save Principal Variation Moves List to pvMoveList
    def selectMovesLineToAnalyse(self):  ### 3
        self.pvMoveList = self.baseReferenceAnalysis[PV__SELECTED]['pv'][:DEPTH__PV__LIMIT]

    # A3: GENERATE STATE FOR EACH MOVE IN PRINCIPAL VARIATION AND ITS Q-VALUE
    # Make copy of board,
    # FOR EACH move IN pvMoveList:
    # append the starting square and the ending square of the move to pvMoveSquaresList
    # push the move to the board copy,
    # make engine analyse board copy,
    # append board state Q-value to pvStateValuesList
    def generateBoardStatesAndValuesFromMovesLine(self):
        current_board = self.board.copy()

        for move in self.pvMoveList:
            self.pvMoveSquaresList.append((move.from_square, move.to_square))
            current_board.push(move)
            self.pvBoardStatesList.append(current_board.copy())
            reference_analysis = self.cbEngine.analyse(current_board, DEPTH__BOUND, multipv=MULTI__PV)
            self.pvStateValuesList.append(getScoreType(reference_analysis[PV__SELECTED]['score']))

    # A4: CREATE CHESSPIECES AS AGENTS
    # Create empty pieces dictionary,
    # FOR EACH square of the piece, piece IN pieces on the board:
    # generate ChessPiece object with type, color, square, Principal Variation Move List,
    # save the piece in pieces dictionary with key "square"+"piece type"
    def generateAndInitializeAllPieces(self):  ### 1
        self.pieces = {}
        # pieces_on_board_dict = self.board.piece_map().items()

        pieces_on_board_dict = [(square, piece) for (square, piece) in self.board.piece_map().items()
                                if piece.symbol().lower() != 'k']
        # Clean list from kings

        for square, piece in pieces_on_board_dict:
            generated_key = str(square) + str(piecesNumberDict[piece.piece_type])
            generated_piece_args = piecesNumberDict[piece.piece_type], colorDict[piece.color], square, (
                        self.pvMoveList + ['nan'])
            # generated_piece_args = piecesNumberDict[piece.piece_type], colorDict[piece.color], square, self.pvMoveList
            self.pieces[generated_key] = ChessPiece(*generated_piece_args)

    # A5: UPDATE SQUARE LIST FOR EACH CHESSPIECE
    # List ChessPiece objects of pieces dictionary,
    # FOR EACH step number, (move from square, move to square) IN enumerate(List of moves (from, to) format):
    # FOR EACH piece IN ChessPiece objects list:
    # if the piece is involved in the move:
    # update its squares list
    def establishSquareSequenceForAllPieces(self):
        pieces_list = list(self.pieces.values())
        for step, (move_from_square, move_to_square) in enumerate(self.pvMoveSquaresList):
            for piece in pieces_list:
                if piece.pPositions[-1] == move_from_square:
                    piece.pPositions.append(move_to_square)
                elif piece.pPositions[-1] == move_to_square:
                    # piece.pPositions.append(-1) doesn't work --> it goes to the last element
                    # of the chessboard, removing the piece in 63!
                    piece.pPositions.append(-1)
                else:
                    piece.pPositions.append(piece.pPositions[-1])

    # B1: GENERATE SALIENCY FOR EACH CHESSPIECE
    # Generate perturbation Q-values,
    # generate gamma values,
    # calculate specificity,
    # calculate relevance,
    # calculate saliency
    def generateSaliencyForPiece(self, piece):
        # print("###### STARTING: ", piece.pType, piece.pColor, piece.pInitialPosition)
        piece.calculatePerturbedQValues(copy.deepcopy(self.pvBoardStatesList), self.pvStateValuesList)
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Q values done")
        piece.calculateGammaValues(self.pvStateValuesList)
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Gamma values done")
        piece.calculateSpecificity()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Specificity values done")
        piece.calculateRelevance()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Relevance values done")
        piece.calculatePerspective()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Perspective values done")
        piece.calculateSaliency()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Saliency values done")

    # B2: KILL ALL ENGINES WHEN ENGINES' WORK DONE
    def killAllEngines(self):
        self.cbEngine.close()
        pieces_list = list(self.pieces.values())

        for piece in pieces_list:
            piece.pEngine.close()

    # ROTUINE A: (A1, A2, A3, A4, A5)
    # Setup the chessboard for the calcuation
    def setupRoutine(self):
        self.generateBaseBoardAnalysis()
        print("A1")
        self.selectMovesLineToAnalyse()
        print("A2")
        self.generateBoardStatesAndValuesFromMovesLine()
        print("A3")
        self.generateAndInitializeAllPieces()
        print("A4")
        self.establishSquareSequenceForAllPieces()
        print("A5")

    # ROUTINE B: (B1, B2)
    # Calculate saliency for each ChessPiece
    def calculationRoutine(self):
        pieces_list = list(self.pieces.values())

        calculatePerspectiveValues()
        if PARALLELIZED:
            with ThreadPoolExecutor(max_workers=4) as pool:
                pool.map(self.generateSaliencyForPiece, pieces_list)
        else:
            for piece in pieces_list:
                self.generateSaliencyForPiece(piece)

        print("B1")
        self.killAllEngines()
        print("B2")


class Analysis:
    def __init__(self, board, pieces, pv_move_list):
        self.board = board
        self.pieces = pieces
        self.pvMoveList = [str(move) for move in pv_move_list] + [" "]
        # self.pvMoveList = [" "] + [str(move) for move in pv_move_list]
        self.pvMoveStringsList = ".".join([str(move) for move in pv_move_list])
        self.QValuesLists = [piece.perturbedStateValuesList for piece in list(pieces.values())]
        self.gammaLists = [piece.gammaValuesList for piece in list(pieces.values())]
        self.specificityLists = [piece.specificityList for piece in list(pieces.values())]
        self.relevanceLists = [piece.relevanceList for piece in list(pieces.values())]
        self.perspectiveLists = [piece.perspectiveList for piece in list(pieces.values())]
        self.saliencyLists = [piece.saliencyList for piece in list(pieces.values())]
        self.pPositionsLists = [piece.pPositions for piece in list(pieces.values())]

        self.piecesMinMaxValues = {}

    def plotAllGraphs(self, input):
        plt.rcParams["figure.figsize"] = (60, 60)
        plt.rcParams["font.size"] = 15
        plt.rcParams['lines.linewidth'] = 8
        plt.rcParams['lines.markeredgecolor'] = 'blue'
        plt.rcParams['lines.markerfacecolor'] = 'red'
        plt.rcParams['lines.markersize'] = 12

        for values, piece in zip(input, self.pieces):
            plt.plot(values, label=str(piece), linestyle='-', marker='o')

        plt.legend()

    def plotAllGraphsFAN2(self, input1, input2, input3, input4, puzzle_n):
        my_labels = ["y = 0", "Specificity", "Relevance", "Perspective", "Saliency"]
        plt.rcParams["figure.figsize"] = (11, 35)
        plt.rcParams["font.size"] = 10
        # plt.rcParams['lines.linewidth'] = 4
        # plt.rcParams['lines.markeredgecolor'] = 'white'
        # plt.rcParams['lines.markerfacecolor'] = 'teal'
        # plt.rcParams['lines.markersize'] = 11
        # plt.rcParams['lines.markeredgewidth'] = 2

        number_of_subplots = len(input1)
        number_of_subplots_x = 2
        number_of_subplots_y = math.ceil(number_of_subplots / number_of_subplots_x)
        fig, axs = plt.subplots(number_of_subplots_y, number_of_subplots_x)
        # fig, axs = plt.subplots(number_of_subplots, figsize=(20, 7.5))

        for idx, (ax, values1, values2, values3, values4, piece) in enumerate(
                zip(axs.flat, input1, input2, input3, input4, self.pieces.values())):
            # GREY BORDERS
            for spine in ax.spines.values():
                # spine.set_visible(False)
                spine.set_edgecolor('lightgrey')

            # ZERO LINE REFERENCE
            ax.hlines(y=0, linewidth=1, xmin=0, xmax=len(self.pvMoveList) - 1, linestyle='dotted', color='black')

            # GENERIC
            ax.set(ylim=(-1.2, 1.2))
            current_title = str(idx) + ": " + str(piece.pColor) + " " + str(piece.pType) + " in " + positionConversion[
                piece.pInitialPosition]
            ax.title.set_text(current_title)
            ax.grid(color='lightgrey', linewidth=1)

            # PLOT VALUE2
            ax.plot(self.pvMoveList, values2, color='goldenrod', linestyle='--', linewidth=2, label='Specificity')

            # PLOT VALUE3
            ax.plot(self.pvMoveList, values3, color='salmon', linestyle='--', linewidth=2, label='Relevance')

            # PLOT VALUE4
            ax.plot(self.pvMoveList, values4, color='blue', linestyle='--', linewidth=1, label='Perspective')

            # PLOT VALUE1
            ax.plot(self.pvMoveList, values1, color='teal', linestyle='-', marker='o', linewidth=4,
                    markeredgecolor='white', markerfacecolor='teal', markersize=11,
                    markeredgewidth=2, label="Saliency")

            for idy, value in zip(self.pvMoveList, values1):
                ax.annotate(
                    round(value, 2),  # this is the text
                    (idy, value),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 15),  # distance from text to points (x,y)
                    ha='center',
                    fontweight="bold"
                )

        fig.legend(labels=my_labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout()

        dir_string = str(puzzle_n) + "." + str(returnSimplifiedFEN(self.board.fen())) + "_GRAPHS" + ".png"
        fig.savefig('game/' + dir_string)

    def setupPiecesMinMaxValues(self):
        for key, piece in self.pieces.items():
            curr_max = max([x for x in piece.saliencyList if x > 0], default=0)
            curr_min = min([x for x in piece.saliencyList if x < 0], default=0)
            self.piecesMinMaxValues[piece] = (curr_max, curr_min)

    def representPerturbationValues(self, puzzle_n):
        das = drawSvg.Drawing(390, 390)
        cell_dim = 45
        # print("11111")
        svg_board = chess.svg.board(self.board)
        board_png = cairosvg.svg2png(svg_board)
        das.append(drawSvg.Image(0, 0, 390, 390, data=board_png))
        # print("11112")
        for piece, (curr_max, curr_min) in self.piecesMinMaxValues.items():
            posStartPoint = (posnegValueRepresentationLocation[positionConversion[piece.pInitialPosition]][0],
                             posnegValueRepresentationLocation[positionConversion[piece.pInitialPosition]][1])
            posEndPoint = (posStartPoint[0], round(posStartPoint[1] + abs(curr_max * cell_dim), 3))
            negStartPoint = (posnegValueRepresentationLocation[positionConversion[piece.pInitialPosition]][2],
                             posnegValueRepresentationLocation[positionConversion[piece.pInitialPosition]][3])
            negEndPoint = (negStartPoint[0], round(negStartPoint[1] + abs(curr_min * cell_dim), 3))

            das.append(drawSvg.Line(posStartPoint[0], posStartPoint[1], posEndPoint[0], posEndPoint[1], stroke='green',
                                    stroke_width=3))
            das.append(drawSvg.Line(negStartPoint[0], negStartPoint[1], negEndPoint[0], negEndPoint[1], stroke='red',
                                    stroke_width=3))

        dir_string = str(puzzle_n) + "." + str(returnSimplifiedFEN(self.board.fen())) + self.pvMoveStringsList + ".png"
        das.savePng('game/' + dir_string)

        return das
