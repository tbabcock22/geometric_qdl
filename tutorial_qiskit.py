import warnings
import os
import random
import torch
from torch.utils.data import DataLoader
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from torch import optim
import numpy as np

# Suppress Intel MKL warnings
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
warnings.filterwarnings('ignore', message='.*MKL*')

# Fix seeds for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(16)
random.seed(16)



# Create an empty board
def create_board():
    """
    Creates a new 3x3 Tic-Tac-Toe board represented as a PyTorch tensor with all elements initialized to 0.

    Returns:
        torch.Tensor: A 3x3 tensor representing the initial state of the Tic-Tac-Toe board.
    """
    return torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

# Check for empty places on board
def possibilities(board):
    l = [(i, j) for i in range(len(board)) for j in range(3) if board[i, j] == 0]
    return l

# Select a random place for the player
def random_place(board, player):
    """
    Randomly places the given player's move on the game board.

    Args:
        board (torch.Tensor): The current state of the game board.
        player (int): The player whose move is being placed (1 or -1).

    Returns:
        torch.Tensor: The updated game board with the player's move placed.
    """
    selection = possibilities(board)
    current_loc = random.choice(selection)
    board[current_loc] = player
    return board

# Check if there is a winner by having 3 in a row
def row_win(board, player):
    """
    Checks if a player has won a row in the game board.

    Args:
        board (numpy.ndarray): The current state of the game board.
        player (int): The player number (1 or 2) to check for a row win.

    Returns:
        bool: True if the player has won a row, False otherwise.
    """
    for x in range(len(board)):
        if all([board[x, y] == player for y in range(3)]):
            return True
    return False

# Check if there is a winner by having 3 in a column
def col_win(board, player):
    """
    Checks if the given player has won the game by having all their pieces in a single column.

    Args:
        board (dict): A dictionary representing the game board, where the keys are (x, y) tuples and the values are the player occupying that square.
        player (str): The player to check for a column win.

    Returns:
        bool: True if the given player has won the game by having all their pieces in a single column, False otherwise.
    """
    for y in range(3):
        if all([board[x, y] == player for x in range(len(board))]):
            return True
    return False

# Check if there is a winner by having 3 in a diagonal
def diag_win(board, player):
    """
    Checks if the given board configuration represents a diagonal win for the specified player.

    Args:
        board (numpy.ndarray): The current state of the game board.
        player (int): The player whose diagonal win is being checked.

    Returns:
        bool: True if the specified player has a diagonal win, False otherwise.
    """
    if all([board[i, i] == player for i in range(len(board))]) or all(
        [board[i, len(board) - i - 1] == player for i in range(len(board))]
    ):
        return True
    return False


# Check if the win conditions have been met or if a draw has occurred
def evaluate_game(board):
    """
    Evaluates the state of the game board and determines the winner.

    Args:
        board (torch.Tensor): The current state of the game board, represented as a 3x3 tensor with values -1, 0, or 1.

    Returns:
        int: The winner of the game, where -1 represents player 1, 0 represents a tie, and 1 represents player 2. If the game is not yet finished, returns `None`.
    """
    winner = None
    for player in [1, -1]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player

    if torch.all(board != 0) and winner is None:
        winner = 0

    return winner

# Main function to start the game
def play_game():
    """
    Plays a single game of Tic-Tac-Toe and returns the final board state and the winner.

    Returns:
        list: The final board state as a flattened list, and the winner (-1, 0, or 1).
    """
    board, winner, counter = create_board(), None, 1
    while winner is None:
        for player in [1, -1]:
            board = random_place(board, player)
            counter += 1
            winner = evaluate_game(board)
            if winner is not None:
                break
    return [board.flatten(), winner]


def create_dataset(size_for_each_winner):
    """
    Creates a dataset of game boards and their corresponding winners.

    Args:
        size_for_each_winner (int): The number of game boards to include for each winner.

    Returns:
        list: A list of tuples, where each tuple contains a game board and its corresponding winner (-1, 0, or 1).
    """
    game_d = {-1: [], 0: [], 1: []}
    while min([len(v) for v in game_d.values()]) < size_for_each_winner:
        board, winner = play_game()
        if len(game_d[winner]) < size_for_each_winner:
            game_d[winner].append(board)
    res = []
    for winner, boards in game_d.items():
        res += [(board, winner) for board in boards]
    return res


NUM_TRAINING = 450
NUM_VALIDATION = 600

# Create datasets with even numbers of each outcome
with torch.no_grad():
    dataset = create_dataset(NUM_TRAINING // 3)
    dataset_val = create_dataset(NUM_VALIDATION // 3)

# Set up the device and the backend
backend = AerSimulator()


def create_observables():
    """
    Creates three observables for a 9-qubit quantum system.

    The `create_observables()` function returns three observables:
    - `ob_center`: The observable for the center qubit.
    - `ob_corner`: The observable for the four corner qubits.
    - `ob_edge`: The observable for the four edge qubits.

    These observables are constructed using tensor products of Pauli Z and identity matrices.
    """
    # Define observables using SparsePauliOp
    ob_center = SparsePauliOp.from_list([("IIIIZIIII", 1)])
    ob_corner = SparsePauliOp.from_list([("IIIIIIIIZ", 1/4), ("IIIIIIZII", 1/4), ("IIZIIIIII", 1/4), ("ZIIIIIIII", 1/4)])
    ob_edge = SparsePauliOp.from_list([("IIIIIIIZI", 1/4), ("IIIIIZIII", 1/4), ("IIIZIIIII", 1/4), ("IZIIIIIII", 1/4)])

    return ob_center, ob_corner, ob_edge

ob_center, ob_corner, ob_edge = create_observables()



def build_circuit(x, p, sym=True):
    """
    Builds a quantum circuit with single-qubit rotations and two-qubit controlled-Y gates.

    The circuit is constructed based on the input parameters `x` and `p`. If `sym` is True, the circuit has a symmetric structure with single-qubit rotations on the center, corners, and edges of the circuit, as well as two-qubit controlled-Y gates circling the edge of the circuit, connecting the corners to the center, and connecting the edges to the center. If `sym` is False, the circuit has a more general structure with single-qubit rotations and two-qubit controlled-Y gates.

    Args:
        x (torch.Tensor): A tensor of 9 parameters for the single-qubit rotations.
        p (torch.Tensor): A tensor of parameters for the single-qubit rotations and two-qubit controlled-Y gates.
        sym (bool): If True, the circuit has a symmetric structure. If False, the circuit has a more general structure.

    Returns:
        qiskit.QuantumCircuit: The constructed quantum circuit.
    """
    qc = QuantumCircuit(9)  # Add 9 classical bits for measurement
    for i in range(9):
        qc.rx(theta=x[i].item(), qubit=i)
    if sym:
        # Centre single-qubit rotation
        qc.rx(theta=p[0].item(), qubit=4)
        qc.ry(theta=p[1].item(), qubit=4)

        # Corner single-qubit rotation
        qc.rx(theta=p[2].item(), qubit=0)
        qc.rx(theta=p[2].item(), qubit=2)
        qc.rx(theta=p[2].item(), qubit=6)
        qc.rx(theta=p[2].item(), qubit=8)

        qc.ry(theta=p[3].item(), qubit=0)
        qc.ry(theta=p[3].item(), qubit=2)
        qc.ry(theta=p[3].item(), qubit=6)
        qc.ry(theta=p[3].item(), qubit=8)

        # Edge single-qubit rotation
        qc.rx(theta=p[4].item(), qubit=1)
        qc.rx(theta=p[4].item(), qubit=3)
        qc.rx(theta=p[4].item(), qubit=5)
        qc.rx(theta=p[4].item(), qubit=7)

        qc.ry(theta=p[5].item(), qubit=1)
        qc.ry(theta=p[5].item(), qubit=3)
        qc.ry(theta=p[5].item(), qubit=5)
        qc.ry(theta=p[5].item(), qubit=7)

        # Entagling two-qubit gates
        # circling the edge of the board
        qc.cry(theta=p[6].item(), control_qubit=0, target_qubit=1)
        qc.cry(theta=p[6].item(), control_qubit=2, target_qubit=1)
        qc.cry(theta=p[6].item(), control_qubit=2, target_qubit=5)
        qc.cry(theta=p[6].item(), control_qubit=8, target_qubit=5)
        qc.cry(theta=p[6].item(), control_qubit=8, target_qubit=7)
        qc.cry(theta=p[6].item(), control_qubit=6, target_qubit=7)
        qc.cry(theta=p[6].item(), control_qubit=6, target_qubit=3)
        qc.cry(theta=p[6].item(), control_qubit=0, target_qubit=3)

        # To the corners from the centre
        qc.cry(theta=p[7].item(), control_qubit=4, target_qubit=0)
        qc.cry(theta=p[7].item(), control_qubit=4, target_qubit=2)
        qc.cry(theta=p[7].item(), control_qubit=4, target_qubit=6)
        qc.cry(theta=p[7].item(), control_qubit=4, target_qubit=8)

        # To the centre from the edges
        # print("p[8] = ", p[8])
        qc.cry(theta=p[8].item(), control_qubit=1, target_qubit=4)
        qc.cry(theta=p[8].item(), control_qubit=3, target_qubit=4)
        qc.cry(theta=p[8].item(), control_qubit=5, target_qubit=4)
        qc.cry(theta=p[8].item(), control_qubit=7, target_qubit=4)
    else:
        qc.rx(theta=p[0].item(), qubit=4)
        qc.ry(theta=p[1].item(), qubit=4)
        for i in range(8):
            qc.rx(theta=p[2 + i].item(), qubit=i)
            qc.ry(theta=p[10 + i].item(), qubit=i)
        for i in [(0, 1), (2, 1), (2, 5), (8, 5), (8, 7), (6, 7), (6, 3), (0, 3)]:
            qc.cry(theta=p[18].item(), control_qubit=i[0], target_qubit=i[1])

        qc.cry(theta=p[26].item(), control_qubit=4, target_qubit=[0, 2, 6, 8])

        qc.cry(theta=p[30].item(), control_qubit=[1, 3, 5, 7], target_qubit=4)

    return qc



def execute_circuit(qc, backend):
    """
    Executes a quantum circuit on the specified backend, using a preset pass manager for optimization.

    Args:
        qc (QuantumCircuit): The quantum circuit to be executed.
        backend (Backend): The backend on which to execute the circuit.

    Returns:
        QuantumCircuit: The optimized circuit ready for execution.
    """
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_circuit = pm.run(circuits=qc)
    # t_qc = transpile(circuits=qc, backend=backend, pass_manager=pm)
    # t_qc.measure_all()

    return isa_circuit
    # return t_qcdd




# Calculate expectation value using Estimator
def expval(circuit, obs):
    """
Calculates the expectation value of an observable `obs` for a given quantum circuit `circuit`.

Args:
    circuit (QuantumCircuit): The quantum circuit to measure the expectation value for.
    obs (Operator): The observable to measure the expectation value of.

Returns:
    float: The expectation value of the observable for the given circuit.
"""
    observable = obs.apply_layout(circuit.layout)
    estimator = Estimator(backend=backend)
    job = estimator.run([(circuit, observable)])
    # print(f"job: {job.job_id()}\n")
    # print(f"job status: {job.status()}\n")

    result = job.result()
    # print(f"result: {result}\n")
    output  = torch.tensor([float(result[0].data.evs)], requires_grad=True)

    return output


def circuit(x, p):
    """
Constructs a quantum circuit based on the input parameters `x` and `p`.

Args:
    x (numpy.ndarray): The input data.
    p (torch.Tensor): The model parameters.

Returns:
    list: A list containing the expectation values for the center, corner, and edge observables.
"""
    qc = build_circuit(x, p, sym=True)
    # qc.draw(output='mpl')
    # # print(qc)
    # plt.show()

    circuit = execute_circuit(qc, backend)
    center = expval(circuit=circuit, obs=ob_center)
    corner = expval(circuit=circuit, obs=ob_corner)
    edge = expval(circuit=circuit, obs=ob_edge)

    return torch.cat([center, corner, edge])




def circuit_no_sym(x, p):
    """
Executes the quantum circuit and returns the expectation values for the center, corner, and edge observables.

Args:
    x (torch.Tensor): The input tensor representing the game board.
    p (torch.Tensor): The parameter tensor for the quantum circuit.

Returns:
    list: A list of three floats representing the expectation values for the center, corner, and edge observables.
"""
    qc = build_circuit(x, p, sym=False)
    circuit = execute_circuit(qc, backend)
    center = expval(circuit=circuit, obs=ob_center)
    corner = expval(circuit=circuit, obs=ob_corner)
    edge = expval(circuit=circuit, obs=ob_edge)

    return torch.cat([center, corner, edge])


fig, ax = plt.subplots()
ax.set_title("Quantum Circuit")
ax.plot()


def encode_game(game):
    """
    Encodes a game state into a feature vector.

    Args:
        game (tuple): A tuple containing the game board and the result of the game.

    Returns:
        tuple: A tuple containing the encoded board and the encoded result.
    """
    board, res = game
    x = board * (2 * np.pi) / 3
    if res == 1:
        y = [-1, -1, 1]
    elif res == -1:
        y = [1, -1, -1]
    else:
        y = [-1, 1, -1]
    return x, y


def cost_function(params, input, target):
    """
    Computes the cost function for the quantum circuit model.

    Args:
        params (torch.Tensor): The parameters of the quantum circuit.
        input (torch.Tensor): The input data to the quantum circuit.
        target (torch.Tensor): The target output for the input data.

    Returns:
        torch.Tensor: The mean squared error between the circuit output and the target.
    """
    # print(f"this is an example input: {circuit(input[0], params)}\n")
    output = torch.stack([circuit(x, params) for x in input])
    vec = output - target
    sum_sqr = torch.sum(vec * vec, dim=1)
    return torch.mean(sum_sqr)

params = 0.01 * torch.randn(9)
params.requires_grad = True
opt = optim.Adam([params], lr=1e-2)

max_epoch = 15
max_step = 30
batch_size = 10

encoded_dataset = list(zip(*[encode_game(game) for game in dataset]))
encoded_dataset_val = list(zip(*[encode_game(game) for game in dataset_val]))

def accuracy(p, x_val, y_val):
    """
    Computes the accuracy of the model on the given validation dataset.

    Args:
        p (torch.Tensor): The model parameters.
        x_val (list): The input validation data.
        y_val (list): The target validation data.

    Returns:
        float: The accuracy of the model on the validation dataset.
    """
    with torch.no_grad():
        y_val = torch.tensor(y_val)
        y_out = torch.stack([circuit(x, p) for x in x_val])
        acc = torch.sum(torch.argmax(y_out, axis=1) == torch.argmax(y_val, axis=1))
        return acc / len(x_val)

print(f"symmetric accuracy without training = {accuracy(params, *encoded_dataset_val)}")

x_dataset = torch.stack(encoded_dataset[0])
y_dataset = torch.tensor(encoded_dataset[1], requires_grad=False)

# train_loader = DataLoader(list(zip(x_dataset, y_dataset)), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

saved_costs_sym = []
saved_accs_sym = []
for epoch in range(max_epoch):
    rand_idx = torch.randperm(len(x_dataset))
    # Shuffled dataset
    x_dataset = x_dataset[rand_idx]
    y_dataset = y_dataset[rand_idx]
    costs = []
    for step in range(max_step):
        x_batch = x_dataset[step * batch_size : (step + 1) * batch_size]
        y_batch = y_dataset[step * batch_size : (step + 1) * batch_size]
        def opt_func():
            opt.zero_grad()
            loss = cost_function(params, x_batch, y_batch)
            costs.append(loss.item())
            loss.backward()
            return loss
        opt.step(opt_func)

    cost = np.mean(costs)
    saved_costs_sym.append(cost)

    if (epoch + 1) % 1 == 0:
        acc_val = accuracy(params, *encoded_dataset_val)
        saved_accs_sym.append(acc_val)
        res = [epoch + 1, cost, acc_val]
        print("Epoch: {:2d} | Loss: {:3f} | Validation accuracy: {:3f}".format(*res))

params = 0.01 * torch.randn(34)
params.requires_grad = True
opt = optim.Adam([params], lr=1e-2)


def cost_function_no_sym(params, input, target):
    """
    Calculates the cost function for the model without using symmetric encoding.

    Args:
        params (torch.Tensor): The model parameters.
        input (torch.Tensor): The input data.
        target (torch.Tensor): The target data.

    Returns:
        torch.Tensor: The mean squared error between the model output and the target.
    """
    output = torch.stack([circuit_no_sym(x, params) for x in input])
    vec = output - target
    sum_sqr = torch.sum(vec * vec, dim=1)
    return torch.mean(sum_sqr)

max_epoch = 15
max_step = 30
batch_size = 15

encoded_dataset = list(zip(*[encode_game(game) for game in dataset]))
encoded_dataset_val = list(zip(*[encode_game(game) for game in dataset_val]))

def accuracy_no_sym(p, x_val, y_val):
    """
    Calculates the accuracy of a model without using symmetric encoding.

    Args:
        p (torch.Tensor): The model parameters.
        x_val (torch.Tensor): The validation input data.
        y_val (torch.Tensor): The validation target data.

    Returns:
        float: The accuracy of the model on the validation data.
    """
    with torch.no_grad():
        y_val = torch.tensor(y_val)
        y_out = torch.stack([circuit_no_sym(x, p) for x in x_val])
        acc = torch.sum(torch.argmax(y_out, axis=1) == torch.argmax(y_val, axis=1))
        return acc / len(x_val)

print(f"non-symmetric accuracy without training = {accuracy_no_sym(params, *encoded_dataset_val)}")

x_dataset = torch.stack(encoded_dataset[0])
y_dataset = torch.tensor(encoded_dataset[1], requires_grad=False)

# train_loader = DataLoader(list(zip(x_dataset, y_dataset)), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

saved_costs = []
saved_accs = []
for epoch in range(max_epoch):
    costs = []
    rand_idx = torch.randperm(len(x_dataset))
    # Shuffled dataset
    x_dataset = x_dataset[rand_idx]
    y_dataset = y_dataset[rand_idx]
    costs = []
    for step in range(max_step):
        x_batch = x_dataset[step * batch_size : (step + 1) * batch_size]
        y_batch = y_dataset[step * batch_size : (step + 1) * batch_size]
        def opt_func():
            opt.zero_grad()
            loss = cost_function_no_sym(params, x_batch, y_batch)
            costs.append(loss.item())
            loss.backward()
            return loss
        opt.step(opt_func)
    cost = np.mean(costs)
    saved_costs.append(cost)
    if (epoch + 1) % 1 == 0:
        acc_val = accuracy_no_sym(params, *encoded_dataset_val)
        saved_accs.append(acc_val)
        res = [epoch + 1, cost, acc_val]
        print("Epoch: {:2d} | Loss: {:3f} | Validation accuracy: {:3f}".format(*res))

plt.title("Validation accuracies")
plt.plot(saved_accs_sym, "b", label="Symmetric")
plt.plot(saved_accs, "g", label="Standard")
plt.ylabel("Validation accuracy (%)")
plt.xlabel("Optimization steps")
plt.legend()
plt.show()
