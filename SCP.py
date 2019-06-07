import numpy as np
import random
import time
from itertools import combinations


class SetCover():

    def load_data(self, filepath):

        lines = []

        with open(filepath, 'r') as file:
            for line in file:
                lines.append(line)

        # First line
        line_0 = lines[0].strip().split()

        # Number of atributes
        self.nr_atr = int(line_0[0])

        # Number of subsets
        self.nr_subsets = int(line_0[1])

        iter_lines = iter(lines)
        next(iter_lines)

        self.subsets_cost = []
        count_lines_cost = 0

        for line in iter_lines:
            self.subsets_cost.extend(line.strip('\n').split())
            count_lines_cost += 1

            if len(self.subsets_cost) == self.nr_subsets:
                break

        # List containing the cost of each subset
        self.subsets_cost = [int(cost) for cost in self.subsets_cost]

        count_elem = -1
        count_aux = 0
        nr_subsets_in_line = 0

        line_of_size = True

        # Computing the list of subsets
        self.subsets = [set() for _ in range(self.nr_subsets)]

        for line in iter_lines:

            if line_of_size:
                nr_subsets_in_line = int(line.strip('\n').split()[0])
                line_of_size = False
                count_elem += 1
                count_aux = 0

            else:
                line_search = line.strip('\n').split()
                count_aux += len(line_search)

                for j in line_search:
                    self.subsets[int(j) - 1].add(count_elem)

                if count_aux == nr_subsets_in_line:
                    line_of_size = True

        # Creating numpy arrays for increasing performance in some operations
        self.subsets_np = np.array(self.subsets)
        self.subsets_cost_np = np.array(self.subsets_cost)

    def evaluate_solution(self, solution):
        total_cost = self.subsets_cost_np[list(solution)].sum()

        return total_cost

    def is_complete(self, solution):

        if len(solution) == 0:
            return False

        elif len(set.union(*self.subsets_np[list(solution)])) == self.nr_atr:
            return True

        else:
            return False

    def greedy_randomized_algorithm(self, alpha):

        # Subsets in the solution
        Solution = set()

        # Atributes satisfied by the solution
        Solution_atr = set()

        # Candidate set
        C = set(range(self.nr_subsets))

        while not self.is_complete(Solution):
            ratio = dict()

            # Calculate the ratio for every subset in the candidate set
            for i in C:
                atr_added = len(self.subsets[i] - Solution_atr)

                if atr_added > 0:
                    ratio[i] = self.subsets_cost[i] / atr_added

            c_min = min(ratio.values())
            c_max = max(ratio.values())

            RCL = [i for i in C if i in ratio.keys() and ratio[i] <= c_min + alpha * (c_max - c_min)]

            s_index = random.choice(RCL)

            C -= {s_index}
            Solution.add(s_index)
            Solution_atr.update(self.subsets[s_index])

        return Solution

    def remove_redundancy(self, solution):

        for i in solution:
            sol_aux = solution.copy()
            sol_aux.remove(i)
            if self.is_complete(sol_aux):
                solution = sol_aux.copy()

        return solution

    def local_search(self, solution):

        sol_set = self.remove_redundancy(solution)

        # Subsets not in the solution
        unused_set = set([s for s in range(self.nr_subsets) if s not in sol_set])

        best_cost = self.evaluate_solution(sol_set)
        best_sol_set = sol_set.copy()

        increase_remove = True
        increase_swap = True

        while increase_remove or increase_swap:

            increase_remove = False
            increase_swap = False

            search_set = sol_set.copy()

            for i_out in sol_set:

                # Removes redundancy if present
                search_set.difference_update({i_out})

                if self.is_complete(search_set):
                    increase_remove = True
                    sol_set.difference_update({i_out})

                    # If redundancy is present, does not make sense to evaluate any swap
                    break
                else:
                    search_set.update({i_out})

                for i_in in unused_set:

                    # If the subset entering the solution has higher cost than the one exiting, improvement is not possible
                    # This local search never allows the solution to get worst
                    if self.subsets_cost[i_in] <= self.subsets_cost[i_out]:

                        search_set.update({i_in})
                        search_set.difference_update({i_out})

                        cost = self.evaluate_solution(search_set)

                        if cost < best_cost and self.is_complete(search_set):
                            increase_swap = True

                            best_cost = cost
                            best_sol_set = search_set.copy()

                            sub_in = i_in
                            sub_out = i_out

                        search_set = sol_set.copy()

            if increase_swap:

                unused_set.difference_update({sub_in})
                unused_set.update({sub_out})

                sol_set = best_sol_set.copy()

        return sol_set

    def path_relinking(self, initial_solution, final_solution):

        sym_dif = initial_solution.symmetric_difference(final_solution)

        cost_init_sol = self.evaluate_solution(initial_solution)
        cost_final_sol = self.evaluate_solution(final_solution)

        best_cost = min(cost_init_sol, cost_final_sol)

        if cost_init_sol < cost_final_sol:
            best_sol = initial_solution

        else:
            best_sol = final_solution

        current_sol = initial_solution.copy()

        best_dif_cost = 0
        best_change_cost = best_cost

        while len(sym_dif) > 0:

            new_it = True

            for i in sym_dif:

                if i in current_sol:
                    dif_cost = -self.subsets_cost[i]

                else:
                    dif_cost = self.subsets_cost[i]

                current_sol.symmetric_difference_update({i})

                if (dif_cost < best_dif_cost or new_it) and self.is_complete(current_sol):
                    new_it = False

                    best_i = i
                    best_dif_cost = dif_cost

                current_sol.symmetric_difference_update({i})

            current_sol.symmetric_difference_update({best_i})

            best_change_cost += best_dif_cost

            if best_change_cost < best_cost:
                best_cost = best_change_cost
                best_sol = current_sol.copy()

            sym_dif.remove(best_i)

        return best_sol

    def GRASP(self, MaxTime, alpha, nr_pool):
        best_cost = 0
        new_it = True
        P = []

        start = time.time()

        clock = 0
        iteration = -1

        while clock < MaxTime:
            iteration += 1
            x = self.greedy_randomized_algorithm(alpha)
            x = self.local_search(x)

            P.append(x)
            cost = self.evaluate_solution(P[-1])

            if cost < best_cost or new_it:
                best_cost = cost
                new_it = False
                best_sol = P[-1]

                time_best = round(time.time() - start, 2)
                iteration_best = iteration

            if iteration > 1:

                pool = random.sample(range(len(P)), min(iteration, nr_pool))

                for s, t in combinations(pool, 2):

                    x_p = self.path_relinking(P[s], P[t])

                    cost_x_p = self.evaluate_solution(x_p)

                    if cost_x_p < best_cost:

                        best_cost = cost_x_p
                        best_sol = x_p

                        time_best = round(time.time() - start, 2)
                        iteration_best = iteration

            clock = round(time.time() - start, 1)

        return best_cost, best_sol, time_best, iteration_best

