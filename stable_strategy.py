import evolutionary_model as me
import scipy.optimize as opt
import numpy as np


# Function to minimize in check of Nash Equilibrium
def y_min(x, x_opt, args):
    y_x = []
    x_aux = x_opt - x
    for i in range(len(x)):
        y_x.append(me.y(i, x, args))
    return np.dot(x_aux, y_x)


# Function to maximize in check of Evolutionary Stable Strategies
def y_max(x, x_opt, args):
    y_x = []
    x_aux = x - x_opt
    for i in range(len(x)):
        y_x.append(me.y(i, x_opt, args))
    return - np.dot(x_aux, y_x)


# Check of Nash Equilibrium
# Starting condition x_initial; strategy to check x_opt; network and miner args
def equilibrium_Nash(x_initial, x_opt, args):
    cero = np.zeros([len(x_opt)])
    y_x = np.zeros([len(x_opt)])
    for i in range(len(x_opt)):
        y_x[i] = round(me.y(i, x_opt, args), 6) #Rounded to 6 digits to follow fiat restrictions

    #If all expected payoffs are 0 we find a Nash Equilibria by definition
    if np.array_equal(y_x, cero):
        result = np.array([sum(y_x), x_opt])
    #If the minimun possible payoff is below zero we infer that all possible payoffs are also below zero
    else:
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        bnds = [(0, 1) for i in range(len(x_opt))]
        test = opt.minimize(y_max, x_initial, args=(x_opt, args), bounds=bnds, constraints=cons)
        result = [- y_max(test.x, x_opt, args), test.x]

    return result


# Check of Evolutionary Stable Strategy
# Starting condition x_initial, strategy to check x_opt; network and miner args
def minimization(x_initial, x_opt, args):
    y_x = np.zeros(len(x_opt))
    for i in range(len(x_opt)):
        y_x[i] = me.y(i, x_opt, args)
    # If all expected payoffs are 0 we can skip the extra restriction
    if np.all(np.round(y_x, 6)) == 0:
        cons = [{'type': 'eq', 'fun': lambda x: 1 - sum(x)}] # All population states must add up to 1
    else:
        cons = [{'type': 'eq', 'fun': lambda x: 1 - sum(x)},
                {'type': 'eq', 'fun': lambda x: np.dot((x - x_opt), y_x) }]

    bnds = [(0, 1) for i in range(len(x_opt))]
    test = opt.minimize(y_min, x_initial, args=(x_opt, args), bounds=bnds, constraints=cons)
    result = [y_min(test.x, x_opt, args), test.x]

    # If the only population state feasible is our strategy x_opt we can conclude it is not a ESS
    if np.array_equal(result[1], x_opt):
        result[0] = -1
        return result

    if result[0] >= 0:
        return result

    else:
        # The condition for stability need only be meet in a neighborhood of the strategy, we find the smalest posible
        # neighborhood that meets our conditions
        while result[0] < 0 and np.linalg.norm(x_opt - result[1]) > 1 / args[1]:

            cons2 = [{'type': 'eq', 'fun': lambda x: 1 - sum(x)},
                     {'type': 'ineq', 'fun': lambda x: np.linalg.norm(x_opt - result[1]) - np.linalg.norm(x_opt - x) - 0.00001},
                     {'type': 'eq', 'fun': lambda x: np.dot((x - x_opt), y_x)}]
            test2 = opt.minimize(y_min, x_initial, args=(x_opt, args), bounds = bnds, constraints = cons2)
            result = [y_min(test2.x, x_opt, args), test2.x]

        return result
