import matplotlib.pyplot as plt
import evolutionary_model as me
import stable_strategy as ss
import numpy as np
import time

if __name__ == '__main__':
    # Miner parameters
    m = 8
    n = m * 2500
    x = np.array([1 / m for i in range(m)])
    # Network parameters
    R = 12.5
    r = 30 * (1000000 / 100000000)  # 30 satoshis per byte
    g = 0.005
    T = 600
    p = 0.079 * 0.37 / 72335.89  # Electricity cost in BTC/CNY
    s = 1
    # Strategy profile
    w_total = 114.190 * T  # PentaHashes (PH) per cycle T
    w = np.array([np.float64(.30625), np.float64(.25125), np.float64(.1975), np.float64(.13125),  np.float64(.07375),
                  np.float64(.0675), np.float64(.0475), np.float64(.035)]) * np.float(w_total)
    # args list
    args = (w, n, s, R, r, g, T, p)
    # Max time for model
    t_max = 20000

    # Time start
    start = time.time()
    # Evolutionary model
    x_t = me.evol(x, t_max, args)
    x_final = x_t[t_max - 1]
    y_final = []
    for i in range(len(x_final)):
        y_final.append(me.y(i, x_final, args))

    print('The population state: ' + str(x_final) + ' with payoff : ' + str(y_final) + 'is candidate to be an ESS.')

    # Plot
    start_plt = time.time()  # timer pause
    for i in range(0, m):
        plt.plot(x_t[:, i], label='Gruop ' + str(i + 1) + ' w = ' + str(w[i]))
    plt.title('Evolutionary Model with N = ' + str(n) + ' M = ' + str(m))
    plt.xlabel('Time t')
    plt.ylabel('Population State')
    plt.legend()
    plt.show()

    # End timer pause
    end_plt = time.time()
    start += end_plt - start_plt

    # Check Nash Equilibrium
    EN = False
    test_nash = ss.equilibrium_Nash(x, x_final, args)

    if np.array_equal(test_nash[1], x_final):
        EN = True
        print('The expected payoff of the population state: ' + str(x_final) + ' is equal to the 0 vector, therefore it'
                                                                               ' is a NE')

    elif test_nash[0] <= 0:
        EN = True
        print('The mutant strategy profile that maximizes crossed payoffs is : ' + str(test_nash[1]) +
              ' the maximum payoff: ' + str(test_nash[0]) + ' is less than 0. \nTherefore ' + str(x_final) + ' is a NE.')
    else:
        print('The mutant strategy profile that maximizes crossed payoffs is : ' + str(test_nash[1]) +
              ' the maximum payoff: ' + str(test_nash[0]) + ' is bigger than 0. \nTherefore ' + str(x_final) +
              ' is NOT a NE.')

    # Evolutionary Stable Strategy check
    if EN:
        test = ss.minimization(x, x_final, args)

        if test[0] > 0:
            print('The mutant strategy profile that maximizes crossed payoffs is : ' + str(test[1]) +
                  ' the minimum payoff: ' + str(test[0]) + ' is bigger than 0. \nTherefore ' + str(x_final) +
                  ' is an ESS.')
        else:
            print('The mutant strategy profile that maximizes crossed payoffs is : ' + str(test[1]) +
                  ' the minimum payoff: ' + str(test[0]) + ' is less than 0. \nTherefore ' + str(x_final) +
                  ' is NOT an ESS')

    # End timer
    end = time.time()
    print('END - - - ' + str(end - start) + ' Run time (in seconds)')
