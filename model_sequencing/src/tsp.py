import pyomo.environ as pyo
import numpy as np

def order_blocks_with_tsp(costMatrix_FP, cost_F1, print_gap=False):
    '''
    This function implements a Mixed Integer Linear Programming (MILP) model to solve a modified Traveling Salesman Problem (TSP)
    to find the optimal sequence of the blocks.
    The model aims to order clusters such that ML sub-models with a low rate of false positives and a high accuracy are deployed
    earlier in the sequence by preventing the formation of multiedges between two blocks.

    Arguments:
        n: int
            Number of clusters.
        costMatrix_FP: np.ndarray
            Matrix carrying the False Positive values of clusters.
        cost_F1: np.array
            An array carrying the F1-score of the clusters
    Return:
        The list of block indices representing the optimal sequence.
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

    def get_cluster_seq(selected_edges, n_of_clusters):
        '''
        The TSP problem returs a directed acyclic structure rather than a simple ordered tour.
        The directionality ensures that node x comes before node y in the sequence.
        For example, the solver provides, as a solution, the following:
                (2, 1)-->1.0
                (2, 3)-->1.0
                (2, 4)-->1.0
                (3, 1)-->1.0
                (4, 1)-->1.0
                (4, 3)-->1.0
        This indicates that the x_{2,1} variable is set to 1.0, which means that cluster 2 comes before cluster 1 in
        the sequence. Cluster 2 should also come before cluster 3 and 4,
        and cluster 4 should come before cluster 1 and 3.

        This function takes the selected edges and returns the sequence of clusters.
        '''
        store_cluster_occurrences = {cluster_idx: 0 for cluster_idx in range(1, n_of_clusters + 1)}
        for edge in selected_edges:
            node = edge[0][0]
            store_cluster_occurrences[node] += 1

        clusters_seq = sorted(store_cluster_occurrences, reverse=True, key=store_cluster_occurrences.get)

        # our cluster indices are 0-based, so we need to subtract 1 from each index
        return list(map(lambda x: x - 1, clusters_seq))

    clusters_seq = get_cluster_seq(selected_edges, n)

    return clusters_seq