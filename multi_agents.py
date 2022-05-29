import numpy as np
import abc
import util
from game import Agent, Action

# CONSTANTS #
possible_tiles = 2 ** np.linspace(1, 11, 11)
CORNER = 1
SIDE = 2
ELSE = 3


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile

        score = successor_game_state.score
        num_zeros = count_zeros(current_game_state)
        closest_neigbors = check_neighbors(current_game_state)
        num_two = count_two(current_game_state)
        corner = is_biggest_in_corner(current_game_state)
        biggestest = dist_between_biggestest(current_game_state)
        return score + max_tile + 2 * corner + biggestest + 2 * check_monotonito(current_game_state) + num_zeros - num_two


def check_monotonito(state):
    board = state.board
    best = 0
    for i in range(4):
        board_der_x = board - np.roll(board, -1)
        board_der_x = board_der_x[:, :-1]
        board_der_y = board - np.roll(board, -1, axis=1)
        board_der_y = board_der_y[:-1, :]
        count_y = np.count_nonzero(board_der_y >= 0)
        count_x = np.count_nonzero(board_der_x >= 0)
        count = count_x + count_y
        if best < count:
            best = count
        board = np.rot90(board)
    return best


def check_neighbors(state):
    board = state.board
    score = 0
    for y in range(len(board)):
        for x in range(len(board)):
            if board[y, x] == 0:
                continue
            neighbor_status = get_neighbors(board, x, y)
            if neighbor_status == 0:
                continue
            score += 1 / get_neighbors(board, x, y)
    return score


def count_zeros(state):
    board = state.board
    return np.count_nonzero(board == 0)


def get_neighbors(board, x, y):
    val = board[y, x]
    board_w = len(board[0])
    board_h = len(board)
    neighbors = []
    if x + 1 < board_w and board[y, x + 1] != 0:
        neighbors.append(board[y, x + 1])
    if x - 1 >= 0 and board[y, x - 1] != 0:
        neighbors.append(board[y, x - 1])
    if y + 1 < board_h and board[y + 1, x] != 0:
        neighbors.append(board[y + 1, x])
    if y - 1 >= 0 and board[y - 1, x] != 0:
        neighbors.append(board[y - 1, x])
    if not neighbors:
        return 1
    neighbors = np.array(neighbors)
    neighbors = neighbors / val
    neighbors = np.log(neighbors)
    neighbors.sort()
    res = neighbors[:2]
    return np.sum(res)


def count_two(state):
    return np.count_nonzero(state.board == 2)


def check_double_pairs(state):
    board = state.board
    board_der_x = board - np.roll(board, -1)
    board_der_x = board_der_x[:, :-1]
    board_der_y = board - np.roll(board, -1, axis=1)
    board_der_y = board_der_y[:-1, :]
    count_x = np.count_nonzero(board_der_x == 0)
    count_y = np.count_nonzero(board_der_y == 0)
    count = count_x + count_y
    return count


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def recursive_helper(self, game_state, depth, agent_index):
        if depth == 0:
            return self.evaluation_function(game_state), None
        kidos = game_state.get_legal_actions(agent_index)
        if agent_index == 0:
            max_eval = -np.inf
            max_move = None
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, 1)
                if eval[0] > max_eval:
                    max_eval = eval[0]
                    max_move = kido
            return max_eval, max_move
        else:
            min_eval = np.inf
            min_move = None
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, 0)
                if eval[0] < min_eval:
                    min_eval = eval[0]
                    min_move = kido
            return min_eval, min_move

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        action = self.recursive_helper(game_state, 2 * self.depth, 0)

        if action[1]:
            return action[1]
        return Action.STOP


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def recursive_helper(self, game_state, depth, alpha, beta, agent_index):
        if depth == 0:
            return self.evaluation_function(game_state), None
        kidos = game_state.get_legal_actions(agent_index)
        if agent_index == 0:
            max_eval = -np.inf
            max_move = None
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, alpha, beta,
                                             1)
                if eval[0] > max_eval:
                    max_eval = eval[0]
                    max_move = kido
                alpha = max(alpha, eval[0])
                if beta <= alpha:
                    break
            return max_eval, max_move
        else:
            min_eval = np.inf
            min_move = None
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, alpha, beta,
                                             0)
                if eval[0] < min_eval:
                    min_eval = eval[0]
                    min_move = kido
                beta = min(beta, eval[0])
                if beta <= alpha:
                    break
            return min_eval, min_move

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action = self.recursive_helper(game_state, 2 * self.depth, -np.inf, np.inf, 0)
        if action[1]:
            return action[1]
        return Action.STOP


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def recursive_helper(self, game_state, depth, agent_index):
        if depth == 0:
            return self.evaluation_function(game_state), None
        kidos = game_state.get_legal_actions(agent_index)
        if agent_index == 0:
            max_eval = -np.inf
            max_move = None
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, 1)
                if eval[0] > max_eval:
                    max_eval = eval[0]
                    max_move = kido
            return max_eval, max_move
        else:
            min_move = None
            count = 0
            for kido in kidos:
                eval = self.recursive_helper(game_state.generate_successor(agent_index, kido), depth - 1, 0)
                count += eval[0]
            expectedMin = count / len(kidos)
            return expectedMin, min_move

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """

        action = self.recursive_helper(game_state, 2 * self.depth, 0)
        if action[1]:
            return action[1]
        return Action.STOP


def find_closest_to_biggest(state):
    board = state.board
    index = np.argmax(board)
    board_w = len(board[0])
    board_h = len(board)
    y = index // board_w
    x = index % board_w
    corners = [(0, 0), (0, board_w), (board_h, board_w), (board_h, 0)]
    best_corner = 0
    min_dist = np.inf
    for corner in corners:
        dist = np.sqrt((y - corner[0]) ** 2 + (x - corner[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_corner = corner
    return best_corner, dist * np.log(board[y, x])


def get_dist_matrix(corner):
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    final = np.sqrt((x - corner[1]) ** 2 + (y - corner[0]) ** 2)
    return final


def close_to_corner(state):
    board = state.board
    best_corner, _ = find_closest_to_biggest(state)
    dist_matrix = get_dist_matrix(best_corner)
    result = np.log(np.sum(dist_matrix * board))
    return result


def is_biggest_in_corner(state):
    board = state.board
    index = np.argmax(board)
    board_w = len(board[0])
    board_h = len(board)
    y = index // board_w
    x = index % board_w
    # corners = [(0, 0), (0, board_w), (board_h, board_w), (board_h, 0)]
    if (y, x) == (0, 0):  # or (y, x) == (0, board_w - 1) or (y, x) == (board_h - 1, board_w - 1) or (y, x) == (
        # board_h - 1, 0):
        return board[y, x]
    return -100 * board[y, x]


def dist_between_biggestest(state):
    board = state.board
    index = np.argmax(board)
    board_w = len(board[0])
    board_h = len(board)
    y_0 = index // board_w
    x_0 = index % board_w
    val_0 = board[y_0, x_0]
    board[y_0, x_0] = 0
    index = np.argmax(board)
    y_1 = index // board_w
    x_1 = index % board_w
    dist = np.sqrt((y_1 - y_0) ** 2 + (x_1 - x_0) ** 2)
    board[y_0, x_0] = val_0
    if dist == 1:
        return val_0
    return val_0 / (2 * dist)


def dist_between_biggestest_helper(board):
    index = np.argmax(board)
    board_w = len(board[0])
    board_h = len(board)
    y_0 = index // board_w
    x_0 = index % board_w
    val_0 = board[y_0, x_0]
    board[y_0, x_0] = 0
    index = np.argmax(board)
    y_1 = index // board_w
    x_1 = index % board_w
    dist = np.sqrt((y_1 - y_0) ** 2 + (x_1 - x_0) ** 2)
    if dist == 1:
        return val_0, board
    return 0, board


def snake_structure(state):
    board = state.board
    if np.max(board) < 128 or np.argmax(board) != 0:
        return 0
    check = 1
    result = 0
    while check != 0:
        if np.max(board) <= 32:
            break
        val, board = dist_between_biggestest_helper(board)
        result += val
    return np.log(result + 1)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: we used a few heuristics in our evaluation function:
    1. the score of the state
    2. sticking the largest tile to the (0,0) corner
    3. sticking the second largest tile to the largest tile
    4. the value of the maximum tile
    5. the monotonity of the board - that is, not creating a situation of 256, 1024, 512 (in this order),
    because it will be difficult to merge them into 2048
    6. the number of zeros on the board - we and it to be as empty as possible
    7. the number of twos on the board - we want to have as little small tiles as possible
    8. the number of equal neighbors on the board - we want to have as many as possible because we
    will be able to merge them in the next move.
    it is important to say that in our way of weighting the heuristics, we preferred heuristics 2 and 3,
    while the last 4 heuristics had a small impact, but leaving them in like that did not ruin the algorithm.

    """
    board = current_game_state.board
    max_tile = current_game_state.max_tile

    score = current_game_state.score
    # doubles = check_double_tiles(current_game_state)
    num_zeros = count_zeros(current_game_state)
    closest_neigbors = check_neighbors(current_game_state)
    num_two = count_two(current_game_state)
    # structure = close_to_corner(current_game_state)
    _, dist_to_biggest = find_closest_to_biggest(current_game_state)
    place_of_biggest = is_biggest_in_corner(current_game_state)
    couple_highest = dist_between_biggestest(current_game_state)
    double_pairs = check_double_pairs(current_game_state)
    # structure = snake_structure(current_game_state)
    return score // 2 + 2 * place_of_biggest + couple_highest + 1.2 * max_tile + \
    3 * check_monotonito(current_game_state) + 2 * num_zeros - num_two + double_pairs
    # return ((score ** 0.5) * (place_of_biggest ** 2) * couple_highest * max_tile * \
    #         (check_monotonito(current_game_state) ** 3) * (num_zeros ** 2) / (num_two + 1) * double_pairs)**(1 / 8)


# Abbreviation
better = better_evaluation_function
