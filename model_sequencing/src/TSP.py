import pyomo.environ as pyo
import numpy as np

def find_with_tsp_selected_edges(costMatrix_FP, cost_F1, print_gap=False):
    '''
    Function to formulate and run TSP to find the optimal sequence of the clusters
    ToDo: explain here the particulars of this TSP_MTZ formulation. In particular, that it does not return a tour.
        Explain how selected_edges should be interpreted.
    
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

    # Cost Matrix cij
    model.c = pyo.Param(model.N, model.M, initialize=lambda model, i, j: cost_matrix[i - 1][j - 1])

    # 4 | define objective function
    def obj_func(model):
        return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)

    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # 5 | define constraints
    def rule_const1(model, i, j):
        '''
        this contraint ensures that only one edge is selected between two nodes
        '''
        if i != j:
            return model.x[i, j] + model.x[j, i] == 1
        else:
            # A rule type function must provide a Pyomo object, so that’s why we had to write a weird else condition.
            return model.x[i, j] - model.x[i, j] == 0
    model.const1 = pyo.Constraint(model.N, model.M, rule=rule_const1)

    def rule_const2(model, i, j):
        '''
        this constraint eliminates the possibility of subtours
        '''
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
    selected_edges = []
    for i in model.x:
        if model.x[i].value > 0:
            selected_edges.append((i, model.x[i].value))

    if print_gap:
        solution_gap = (completeResults.Problem._list[0]['Upper bound'] - completeResults.Problem._list[0]['Lower bound']) / completeResults.Problem._list[0]['Upper bound']
        print(solution_gap)

    # ToDo: identify when there are multiple optimal solutions and log a warning to the user that only one
    #  solution is returned.

    return selected_edges


def get_cluster_seq(selected_edges, n_of_clusters):
    '''
    ToDo: explain what is done in the for loop to extract the order from the selected_edges.
    Function to get the sequence of the sub-models obtained by the TSP
    '''
    store_cluster_occurrences = {cluster_idx:0 for cluster_idx in range(1, n_of_clusters+1)}
    for edge in selected_edges:
        node = edge[0][0]
        store_cluster_occurrences[node] += 1

    clusters_seq = sorted(store_cluster_occurrences, reverse=True, key=store_cluster_occurrences.get)

    # our cluster indices are 0-based, so we need to subtract 1 from each index
    return list(map(lambda x: x - 1, clusters_seq))
