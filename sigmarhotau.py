from TTDef import *
import chess
import chess.engine
import chess.svg
from cairosvg import svg2png
from PIL import Image
import drawSvg
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from concurrent.futures import ThreadPoolExecutor
import math
import copy
import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os

engine_path = r"C:\Users\molin\PycharmProjects\thesisBeta\engine\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)


def returnSimplifiedFEN(FEN):
    return "".join(FEN.replace("/", ".").split())


def returnConvertedBoardToPNG(board, path):
    svg_board = chess.svg.board(board=board)
    svg2png(file_obj=svg_board)
    svg2png(bytestring=svg_board, write_to=path + 'output.png')


def join2PNG(FEN_name, background, foreground, path):
    # background = Image.open(background)
    # foreground = Image.open(foreground)
    background.paste(foreground, (0, 0), foreground)
    background.save(path + FEN_name)
    return background


def getScoreType(pov_score):
    if SCORE__TYPE == 'cp':
        return pov_score.pov(POINT__OF__VIEW).score(mate_score=5000) / (1000)
    elif SCORE__TYPE == 'wdl':
        return pov_score.wdl().pov(POINT__OF__VIEW).expectation()


class ChessPiece:

    def __init__(self, p_type, p_color, p_initial_position):
        self.pType = p_type
        self.pColor = p_color
        self.pInitialPosition = p_initial_position

        self.pvBoardStateList = []
        self.pPositions = [p_initial_position]
        self.perturbedStateValuesList = []
        self.gammaValuesList = []
        self.specificityList = []
        self.relevanceList = []
        self.saliencyList = []

        self.anomalies = []
        self.pEngine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # def initializeEngine(self):
    #     self.pEngine = chess.engine.SimpleEngine.popen_uci(engine_path)

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

        fens = [i.fen() for i in pv_board_states_list]
        # print(fens)
        for step, (pv_board_state, pv_state_value) in enumerate(zip(pv_board_states_list, pv_state_values_list)):
            # print(step, pv_state_value, pv_state_values_list)
            current_square = self.pPositions[step]

            if pv_board_state.remove_piece_at(current_square):
                # print("YES1", pv_board_state.fen())
                if pv_board_state.is_valid():
                    # print("YES2", pv_board_state.fen())
                    reference_analysis = self.pEngine.analyse(pv_board_state, DEPTH__BOUND, multipv=MULTI__PV)
                    try:
                        perturbed_state_value = getScoreType(reference_analysis[PV__SELECTED]['score'])
                        self.perturbedStateValuesList.append(perturbed_state_value)
                    except:
                        print(pv_board_state.fen())
                        print("REFERENCE ANALYSIS: ", reference_analysis)
                else:
                    # print("NO2", pv_board_state.fen())
                    self.manageAnomalies(pv_board_state)
                    self.perturbedStateValuesList.append(pv_state_value)
            else:
                # print("NO1", pv_board_state.fen())
                self.manageAnomalies(pv_board_state)
                self.perturbedStateValuesList.append(pv_state_value)

    def calculateGammaValues(self, pv_state_values_list):
        for base_q_value, perturbed_value in zip(pv_state_values_list, self.perturbedStateValuesList):
            self.gammaValuesList.append(base_q_value - perturbed_value)

    def calculateSpecificity(self):
        for gamma in self.gammaValuesList:
            delta_sigma = math.tanh((math.exp(abs(gamma)) - 1))
            self.specificityList.append(delta_sigma)

    def calculateRelevance(self):
        for gamma in self.gammaValuesList:
            delta_rho = math.tanh(alfa * gamma)
            self.relevanceList.append(delta_rho)

    def calculateSaliency(self):
        for delta_sigma, delta_rho in zip(self.specificityList, self.relevanceList):
            self.saliencyList.append(delta_sigma * delta_rho)


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

    def generateBaseBoardAnalysis(self):
        self.pvBoardStatesList.append(self.board)
        self.baseReferenceAnalysis = self.cbEngine.analyse(self.board, DEPTH__BOUND, multipv=MULTI__PV)
        self.pvStateValuesList.append(getScoreType(self.baseReferenceAnalysis[PV__SELECTED]['score']))

    def selectMovesLineToAnalyse(self):  ### 3
        self.pvMoveList = self.baseReferenceAnalysis[PV__SELECTED]['pv'][:DEPTH__PV__LIMIT]
        # print(type(self.pvMoveList), self.pvMoveList)
        # print(str(self.pvMoveList[0]))

    def generateBoardStatesAndValuesFromMovesLine(self):
        current_board = self.board.copy()

        for move in self.pvMoveList:
            # print(self.board.fen())
            # print("MOVE: ", move)
            self.pvMoveSquaresList.append((move.from_square, move.to_square))
            current_board.push(move)
            self.pvBoardStatesList.append(current_board.copy())
            # print(self.pvBoardStatesList)
            reference_analysis = self.cbEngine.analyse(current_board, DEPTH__BOUND, multipv=MULTI__PV)
            self.pvStateValuesList.append(getScoreType(reference_analysis[PV__SELECTED]['score']))

        # print(self.pvMoveSquaresList)
        fens = [i.fen() for i in self.pvBoardStatesList]
        # print(fens)

    def generateAndInitializeAllPieces(self):  ### 1
        self.pieces = {}
        pieces_on_board_dict = self.board.piece_map().items()
        for square, piece in pieces_on_board_dict:
            generated_key = str(square) + str(piecesNumberDict[piece.piece_type])
            generated_piece_args = piecesNumberDict[piece.piece_type], colorDict[piece.color], square
            self.pieces[generated_key] = ChessPiece(*generated_piece_args)

    def establishSquareSequenceForAllPieces(self):
        pieces_list = list(self.pieces.values())
        for step, (move_from_square, move_to_square) in enumerate(self.pvMoveSquaresList):
            for piece in pieces_list:
                if piece.pPositions[-1] == move_from_square:
                    piece.pPositions.append(move_to_square)
                elif piece.pPositions[-1] == move_to_square:
                    piece.pPositions.append(-1)
                else:
                    piece.pPositions.append(piece.pPositions[-1])

    def killAllEngines(self):  ### LAST
        self.cbEngine.close()
        pieces_list = list(self.pieces.values())

        for piece in pieces_list:
            piece.pEngine.close()

    def generateSaliencyForAllPieces(self, piece):
        print("###### STARTING: ", piece.pType, piece.pColor, piece.pInitialPosition)
        piece.calculatePerturbedQValues(copy.deepcopy(self.pvBoardStatesList), self.pvStateValuesList)
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Q values done")
        piece.calculateGammaValues(self.pvStateValuesList)
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Gamma values done")
        piece.calculateSpecificity()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Specificity values done")
        piece.calculateRelevance()
        # print(piece.pType, piece.pColor, piece.pInitialPosition, "Relevance values done")
        piece.calculateSaliency()
        print(piece.pType, piece.pColor, piece.pInitialPosition, "Saliency values done")

    def executionRoutine(self):
        print("1")
        self.generateBaseBoardAnalysis()
        print("2")
        self.selectMovesLineToAnalyse()
        print("3")
        self.generateBoardStatesAndValuesFromMovesLine()
        print("4")
        self.generateAndInitializeAllPieces()
        print("5")
        self.establishSquareSequenceForAllPieces()
        print("6")

        pieces_list = list(self.pieces.values())

        # for piece in pieces_list:
        #     self.generateSaliencyForAllPieces(piece)

        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.map(self.generateSaliencyForAllPieces, pieces_list)

        print("7")
        self.killAllEngines()

        print("8")


class Analysis:
    def __init__(self, board, pieces, pv_move_list):
        self.board = board
        self.pieces = pieces
        self.pvMoveList = [" "] + [str(move) for move in pv_move_list]
        self.pvMoveStringsList = ".".join([str(move) for move in pv_move_list])
        self.QValuesLists = [piece.perturbedStateValuesList for piece in list(pieces.values())]
        self.gammaLists = [piece.gammaValuesList for piece in list(pieces.values())]
        self.specificityLists = [piece.specificityList for piece in list(pieces.values())]
        self.relevanceLists = [piece.relevanceList for piece in list(pieces.values())]
        self.saliencyLists = [piece.saliencyList for piece in list(pieces.values())]
        self.pPositionsLists = [piece.pPositions for piece in list(pieces.values())]

        self.piecesMinMaxValues = {}

    def plotAllGraphs(self, input):
        plt.rcParams["figure.figsize"] = (20, 15)
        plt.rcParams["font.size"] = 15
        plt.rcParams['lines.linewidth'] = 8
        plt.rcParams['lines.markeredgecolor'] = 'blue'
        plt.rcParams['lines.markerfacecolor'] = 'red'
        plt.rcParams['lines.markersize'] = 12

        for values, piece in zip(input, self.pieces):
            plt.plot(values, label=str(piece), linestyle='-', marker='o')

        plt.legend()

    def plotAllGraphsFAN(self, input):
        plt.rcParams["figure.figsize"] = (7.5, 80)
        plt.rcParams["font.size"] = 12
        plt.rcParams['lines.linewidth'] = 4
        plt.rcParams['lines.markeredgecolor'] = 'white'
        plt.rcParams['lines.markerfacecolor'] = 'teal'
        plt.rcParams['lines.markersize'] = 11
        plt.rcParams['lines.markeredgewidth'] = 2

        number_of_subplots = len(input)
        fig, axs = plt.subplots(number_of_subplots)
        plot_counter = 0

        for idx, values in enumerate(input):
            for spine in axs[plot_counter].spines.values():
                # spine.set_visible(False)
                spine.set_edgecolor('lightgrey')

            axs[plot_counter].hlines(y=0, linewidth=2, xmin=0, xmax=len(self.pvMoveList)-1, color='coral')

            axs[plot_counter].set(ylim=(-1.2, 1.2))
            axs[plot_counter].title.set_text(idx)
            axs[plot_counter].plot(self.pvMoveList, values, color='teal', linestyle='-', marker='o')
            axs[plot_counter].grid(color='whitesmoke', linewidth=1)

            for idy, value in zip(self.pvMoveList, values):
                axs[plot_counter].annotate(
                    round(value, 2),  # this is the text
                    (idy, value),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 15),  # distance from text to points (x,y)
                    ha='center'
                )

            plot_counter += 1

        fig.tight_layout()

        fig.savefig('22222GRAPHS_OUTPUT2222.png')

    def plotAllGraphs1(self, input):
        plt.rcParams["figure.figsize"] = (20, 80)
        plt.rcParams["font.size"] = 15
        plt.rcParams['lines.linewidth'] = 8
        plt.rcParams['lines.markeredgecolor'] = 'blue'
        plt.rcParams['lines.markerfacecolor'] = 'red'
        plt.rcParams['lines.markersize'] = 12

        number_of_subplots = len(input)
        fig, axs = plt.subplots(number_of_subplots, sharex=True, sharey=True)
        plot_counter = 0

        for values, piece in zip(input, self.pieces):
            axs[plot_counter].title.set_text(piece)
            axs[plot_counter].plot(range(DEPTH__PV__LIMIT + 1), values, label="1", linestyle='-', marker='o')
            axs[plot_counter].legend()
            axs[plot_counter].grid(linewidth=2)
            plot_counter += 1

        fig.tight_layout()

        fig.savefig('GRAPHS_OUTPUT.png')

    def plotAllGraphs2(self, input1, input2):
        plt.rcParams["figure.figsize"] = (20, 80)
        plt.rcParams["font.size"] = 15
        plt.rcParams['lines.linewidth'] = 8
        plt.rcParams['lines.markeredgecolor'] = 'blue'
        plt.rcParams['lines.markerfacecolor'] = 'red'
        plt.rcParams['lines.markersize'] = 12

        number_of_subplots = len(input1)
        fig, axs = plt.subplots(number_of_subplots)
        plot_counter = 0

        for values1, values2, piece in zip(input1, input2, self.pieces):
            axs[plot_counter].title.set_text(piece)
            axs[plot_counter].plot(range(DEPTH__PV__LIMIT + 1), values1, label="1", linestyle='-', marker='o')
            axs[plot_counter].plot(range(DEPTH__PV__LIMIT + 1), values2, label="2", linestyle='-', marker='o')
            axs[plot_counter].legend()
            axs[plot_counter].grid()
            plot_counter += 1

        fig.tight_layout()

    def setupPiecesMinMaxValues(self):
        # self.piecesMinMaxValues = {}
        for key, piece in self.pieces.items():
            curr_max = max([x for x in piece.saliencyList if x > 0], default=0)
            curr_min = min([x for x in piece.saliencyList if x < 0], default=0)
            self.piecesMinMaxValues[piece] = (curr_max, curr_min)

    def representPerturbationValues(self, puzzle_n):
        das = drawSvg.Drawing(390, 390)
        cell_dim = 45
        print("11111")
        svg_board = chess.svg.board(self.board)
        board_png = cairosvg.svg2png(svg_board)
        das.append(drawSvg.Image(0, 0, 390, 390, data=board_png))
        print("11112")
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

        # img_png = cairosvg.svg2png(das.rasterize())
        # img = Image.open(BytesIO(img_png))
        das.savePng('game/' + "RESULT.png")
        das.savePng('game/' + str(puzzle_n) + "." + str(
            returnSimplifiedFEN(self.board.fen())) + self.pvMoveStringsList + ".png")
        print("222222")
        return das
    #   return das
