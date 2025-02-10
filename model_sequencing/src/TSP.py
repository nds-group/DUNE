import pyomo.environ as pyo
import numpy as np

def TSP_MTZ_Formulation(n, costMatrix_FP, cost_F1, logger):
    '''
    Function to formulate and run TSP to find the optimal sequence of the clusters 
    
    Arguments:
        n: int
            Number of clusters.
        costMatrix_FP: np.ndarray
            Matrix carrying the False Positive values of clusters.
        cost_F1: np.array
            An array carrying the F1-score of the clusters
    '''
    #%
    # 1 | initialize sets and notations
    N = [i for i in range(1,n+1)]
    arc_IJ = [(i,j) for i in N for j in N if i!=j]
    cost_FP = {(i,j) : costMatrix_FP[i-1][j-1] for (i,j) in arc_IJ}
    cost_F1 = {(i) : cost_F1[i-1]/100 for i in range(1,n+1)}
    
    # 2 | initialize the model
    model = pyo.ConcreteModel()

    # 3 | initialize decision variables
    model.x = pyo.Var(arc_IJ, within=pyo.Binary)
    model.u = pyo.Var(N, within=pyo.NonNegativeIntegers,bounds=(0,n-1))

    # 4 | define objective function
    model.objective = pyo.Objective(
        expr= sum(model.x[i,j]*cost_FP[j,i]*(1-cost_F1[i]) for (i,j) in arc_IJ),
        sense=pyo.minimize)
        # sense=pyo.maximize)

    # 5 | define constraints
    model.constraints = pyo.ConstraintList()

    for i in N:
        for j in N:
            if i!=j:
                model.constraints.add(model.x[i,j] + model.x[j,i] == 1)

    # c) subtour elimination constraints
    for i in N:
        for j in N:
            if i!=j:
                model.constraints.add(model.u[i] - model.u[j] + model.x[i,j] * n <= n-1)

    # # 6 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    model.pprint()
    solver = pyo.SolverFactory('gurobi')
    completeResults = solver.solve(model,tee = True)

    # # 7 | extract the results
    solutionObjective = model.objective()
    tourRepo = []
    for i in model.x:
        if model.x[i].value > 0:
            tourRepo.append((i, model.x[i].value))
            cluster_pair = str(model.x[i])[2:-1].split(sep=',')
            cluster_pair[0] = int(cluster_pair[0])
            cluster_pair[1] = int(cluster_pair[1])
            logger.info(f'{str(model.x[i]), model.x[i].value, cost_FP[(cluster_pair[1], cluster_pair[0])],cost_F1[cluster_pair[0]]}')
    solutionGap = (completeResults.Problem._list[0]['Upper bound'] - completeResults.Problem._list[0]['Lower bound']) / completeResults.Problem._list[0]['Upper bound']

    return solutionObjective, solutionGap, tourRepo, completeResults


def get_cluster_seq(tourRepo, n_of_clusters, logger):
    '''
    Function to get the sequence of the sub-models obtained by the TSP
    '''
    store_cluster_occ = {}
    for i in range(0, n_of_clusters):
        store_cluster_occ[i] = 0
    for var in tourRepo:
        store_cluster_occ[var[0][0]-1] = store_cluster_occ[var[0][0]-1] + 1
    clusters_seq = list(dict(sorted(store_cluster_occ.items(), key=lambda item: item[1])).keys())[::-1]
    logger.info(f'The sequence is: {clusters_seq}')
    return clusters_seq