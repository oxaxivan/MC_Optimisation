# from scipy.optimize import minimize
import numpy as np
# from scipy.optimize import LinearConstraint
import os
import nevergrad as ng
import subprocess


class Step:
    step_number = 0

    def __init__(self, start_point):
        self.exp_data, self.weight_factors = exp_data()
        self.abs0 = start_point[0]
        self.scat0 = start_point[1]
        self.anis0 = start_point[2]


def exe_runner():
    os.chdir(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo')
    os.system(r'"LightTissueIntercation.exe"')


def parameters_writer(param):
    abskwrd = 'ABSKWRD'
    scatkwrd = 'SCATKWRD'
    aniskwrd = 'ANISKWRD'

    f0 = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\settings0.txt', 'r')
    f0.seek(0, 0)
    text = f0.read()
    f0.close()

    text = text.replace(abskwrd, str(step.abs0 * param[0]))                   # !!!!!!!!!!!!!!!!!!!!!!
    text = text.replace(scatkwrd, str(step.scat0 * param[1]))
    text = text.replace(aniskwrd, str(step.anis0 * param[2]))

    f = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\settings.txt', 'w')
    f.write(text)
    f.close()


def simul_result():
    s = np.zeros([2, 200])
    file = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\output.txt')
    kwrd1 = 'For distance '
    len1 = len(kwrd1)
    kwrd2 = 'Collimated transmittance    : '
    len2 = len(kwrd2)
    kwrd3 = 'Diffusive  transmittance    : '
    len3 = len(kwrd3)
    while 1:
        # print('Seeking' + line)
        tmp = file.readline()  # troubles expected
        if 'Absorbed' in tmp:
            break
    i = 0
    # print('Start encoding \n')
    for line in file:
        if kwrd1 in line:
            s[0, i] = float(line[len1:])
            # print('Found Distance')
        if kwrd2 in line:
            s[1, i] = float(line[len2:-2])
            # print('Found CT')
        if kwrd3 in line:
            s[1, i] += float(line[len3:-2])
            # print('Found DT')
            i += 1
    file.close()
    return s


def interpolation(fd):
    xp = fd[0, :]
    fp = fd[1, :]
    inter = [np.interp(step.exp_data[0], xp, fp)]
    return np.concatenate((step.exp_data, inter), axis=0)


def eval_nevyazka(fd):
    arr = interpolation(fd)
    return np.sum((arr[1]-arr[2]) ** 2 * step.weight_factors)


def exp_data():
    file = open(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo\ExpData.txt')
    fulltext = file.read()
    npoints = len(fulltext.split('\n'))
    arr = np.zeros((2, npoints))
    wfarr = np.zeros(npoints)
    file.seek(0, 0)
    i = 0
    for line in file:
        tmp = line.split()
        arr[0, i] = float(tmp[0])
        arr[1, i] = float(tmp[1])
        wfarr[i] = float(tmp[2])
        i += 1
    return arr, wfarr


# exp_data = exp_data();

def nevyazka(param):
    parameters_writer(param)
    os.chdir(r'C:\Users\user\Documents\Python_Scripts\New_Jopa\Monte-Carlo')
    subprocess.run(r'"LightTissueIntercation.exe"', capture_output=True)
    # exe_runner()
    fd = simul_result()
    # (ed, wf) = exp_data()
    tmp = eval_nevyazka(fd)
    print('Step:', step.step_number, "   Parameters: %.2f %.2f %.2f" % (step.abs0 * param[0], step.scat0 * param[1],
          step.anis0 * param[2]), ' \t Nevyazka=', tmp)  # !!!!!!!!!
    step.step_number += 1
    return tmp


def find_solution():
    # start_point = np.array([3., 30., 7.])
    optimizer = ng.optimizers.CMA(parametrization=3, budget=300)
    optimizer.parametrization.register_cheap_constraint(lambda x:
                            (0.1 < x[0] < 10) & (0.1 < x[1] < 10) & (0 < x[2] < 1/step.anis0))
    # optimizer.parametrization.register_cheap_constraint(lambda x: x[1] > 0)
    # optimizer.parametrization.register_cheap_constraint(lambda x: x[2] > 0)
    # optimizer.parametrization.register_cheap_constraint(lambda x: x[2] < 1)
    recommendation = optimizer.minimize(nevyazka)
    print(recommendation.value)
    # constraint = LinearConstraint([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], [np.inf, np.inf, 1])
    # return minimize(nevyazka, start_point, method='COBYLA',
    #                 constraints=constraint, options={'disp': True, 'rhobeg': 2, 'maxiter': 200})


print('Enter start point: Absorption, Scattering, Anisotropy:')
a_tmp = float(input())
s_tmp = float(input())
an_tmp = float(input())
step = Step((a_tmp, s_tmp, an_tmp))
print(step.exp_data)
print(step.weight_factors)
print(find_solution())
