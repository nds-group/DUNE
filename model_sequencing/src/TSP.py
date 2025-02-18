import collections

import pyomo.environ as pyo
import numpy as np

# ToDo: if the function runs the TSP, the name should be run_tsp_mtz.
#  Function names should be lowercase and indicate what they do
def TSP_MTZ_Formulation(costMatrix_FP, cost_F1, logger):
    '''
    Function to formulate and run TSP to find the optimal sequence of the clusters
    ToDo: explain here the particulars of this TSP_MTZ formulation. In particular, that it does not return a tour.
        Explain how tourRepo should be interpreted.
    
    Arguments:
        n: int
            Number of clusters.
        costMatrix_FP: np.ndarray
            Matrix carrying the False Positive values of clusters.
        cost_F1: np.array
            An array carrying the F1-score of the clusters
    ToDo: indicate what objects are returned.
    '''
    n = costMatrix_FP.shape[0]

    #%
    # 1 | initialize sets and notations
    # N = [i for i in range(1,n+1)]
    # arc_IJ = [(i,j) for i in N for j in N if i!=j]
    # cost_FP = {(i,j) : costMatrix_FP[i-1][j-1] for (i,j) in arc_IJ}
    # cost_F1 = {(i) : cost_F1[i-1]/100 for i in range(1,n+1)}
    cost_matrix = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                cost_matrix[i][j] = costMatrix_FP[j][i]*(1-cost_F1[i]/100)
            else:
                cost_matrix[i][j] = np.inf

    # 2 | initialize the model
    model = pyo.ConcreteModel()
    # Indexes for the cities
    model.M = pyo.RangeSet(n)
    model.N = pyo.RangeSet(n)
    # Index for the dummy variable u
    model.U = pyo.RangeSet(2, n)

    # 3 | initialize decision variables
    # Decision variables x_{i,j}
    model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

    # Dummy variable u_i
    model.u = pyo.Var(model.N, within=pyo.NonNegativeIntegers, bounds=(0, n - 1))

    # model.x = pyo.Var(arc_IJ, within=pyo.Binary)
    # model.u = pyo.Var(N, within=pyo.NonNegativeIntegers,bounds=(0,n-1))

    # Cost Matrix cij
    model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: cost_matrix[i - 1][j - 1])

    # 4 | define objective function
    def obj_func(model):
        return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)

    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # 5 | define constraints
    def rule_const1(model, i, j):
        if i != j:
            return model.x[i, j] + model.x[j, i] == 1
        else:
            # A rule type function must provide a Pyomo object, so that’s why we had to write a weird else condition.
            return model.x[i, j] - model.x[i, j] == 0

    model.const1 = pyo.Constraint(model.N, model.M, rule=rule_const1)

    def rule_const2(model, i, j):
        if i != j:
            return model.u[i] - model.u[j] + model.x[i, j] * n <= n - 1
        else:
            # A rule type function must provide a Pyomo object, so that’s why we had to write a weird else condition.
            return model.u[i] - model.u[i] == 0

    model.rest3 = pyo.Constraint(model.U, model.N, rule=rule_const2)

    # # 6 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    # model.pprint()
    solver = pyo.SolverFactory('gurobi')
    completeResults = solver.solve(model,tee = True)

    # # 7 | extract the results
    solutionObjective = model.objective()
    tourRepo = []
    for i in model.x:
        if model.x[i].value > 0:
            tourRepo.append((i, model.x[i].value))
            # cluster_pair = str(model.x[i])[2:-1].split(sep=',')
            # cluster_pair[0] = int(cluster_pair[0])
            # cluster_pair[1] = int(cluster_pair[1])
            # logger.info(f'{str(model.x[i]), model.x[i].value, cost_FP[(cluster_pair[1], cluster_pair[0])],cost_F1[cluster_pair[0]]}')
    solutionGap = (completeResults.Problem._list[0]['Upper bound'] - completeResults.Problem._list[0]['Lower bound']) / completeResults.Problem._list[0]['Upper bound']

    # ToDo: In general, returning many and varied objects is not a good practice.
    return solutionObjective, solutionGap, tourRepo, completeResults


def get_cluster_seq(tourRepo, n_of_clusters, logger):
    '''
    Function to get the sequence of the sub-models obtained by the TSP
    '''
    # ToDo: what does _occ mean? Ocurrences?
    store_cluster_occ = {cluster_idx:0 for cluster_idx in range(1, n_of_clusters+1)}
    # for i in range(0, n_of_clusters):
    #     store_cluster_occ[i] = 0
    for var in tourRepo:
        # ToDo: substracting 1 here obscures the logic. It is better to use the cluster index as is and later substract
        #  one from the final sequence list.
        # store_cluster_occ[var[0][0]-1] = store_cluster_occ[var[0][0]-1] + 1
        node = var[0][0]
        store_cluster_occ[node] += 1

    # store_cluster_occ = collections.defaultdict(int)
    # for var in tourRepo:
    #     store_cluster_occ[var[0][0] - 1] = store_cluster_occ[var[0][0] - 1] + 1


    # ToDo: this is a very convoluted way to get the sequence. What do you think of the below implementation?
    # clusters_seq = list(dict(sorted(store_cluster_occ.items(), reverse=True, key=lambda item: item[1])).keys())
    clusters_seq = [item[0] for item in sorted(store_cluster_occ, reverse=True, key=store_cluster_occ.get)]
    # ToDo: imho, it is a good idea to return the dictionary store_cluster_occ instead.
    #  And attach it to the Partition class---not yet implemented (see Issue #5).

    logger.info(f'The sequence is: {clusters_seq}')
    return clusters_seq