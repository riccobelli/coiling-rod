# Copyright (C) 2020 Davide Riccobelli
#
# This file is part of Coiling Rod library for FEniCS.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from dolfin import (
    Function,
    derivative,
    TestFunction,
    TrialFunction,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    XDMFFile)
import os
import time
from mpi4py import MPI


def log(msg, warning=False, success=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0 and warning:
        fmt = "\033[1;37;31m%s\033[0m"  # Red
    elif rank == 0 and success:
        fmt = "\033[1;37;32m%s\033[0m"  # Green
    elif rank == 0:
        fmt = "\033[1;37;34m%s\033[0m"  # Blue
    if rank == 0:
        timestamp = "[%s] " % time.strftime("%H:%M:%S")
        print(fmt % (timestamp + msg))


class ParameterContinuation(object):

    def __init__(self,
                 problem,  # Class containing the problem to be solved
                 param_name,  # Name of the parameter that we use for the continuation (string)
                 start=0,  # Starting value of the control parameter
                 end=0,  # Final value of the control parameter
                 dt=0.01,  # Step increment (% with respect to end-start)
                 min_dt=1e-6,  # The step is halved if the nonlinear solver does not converge
                 saving_file_parameters={},
                 output_folder="output",
                 remove_old_output_folder=True):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0 and remove_old_output_folder is True:
            os.system("rm -r " + output_folder)
        if rank == 0 and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.problem = problem
        self._param_name = param_name
        self._param_start = start
        self._param_end = end
        self._dt = dt
        self._min_dt = min_dt

        # Set solver parameters
        self._solver_params = {}
        self._save_file = XDMFFile(output_folder + "/results.xdmf")
        self._solver_params.update(problem.solver_parameters())
        if 'nonlinear_solver' not in self._solver_params:
            self._solver_params['nonlinear_solver'] = 'snes'
            self._solver_params['snes_solver'] = {}
        solver_type = self._solver_params['nonlinear_solver']
        self._solver_params[solver_type + '_solver']['error_on_nonconvergence'] = False

    def run(self):
        # Create the mesh and setup of the functional space
        mesh = self.problem.mesh()
        V = self.problem.function_space(mesh)
        u = Function(V)  # Unknown of the problem
        u0 = Function(V)  # Backup of the problem solution at previous step

        # Assign to u and u0 the initial guess provided by the user
        u.assign(self.problem.initial_guess(V))
        u0.assign(self.problem.initial_guess(V))

        # Extract the parameter used for the continuation algorithm
        param = self.problem.parameters()[self._param_name]
        param.assign(self._param_start)

        # Setup of the boundary conditions
        bcs = self.problem.boundary_conditions(mesh, V)
        residual = self.problem.residual(u, TestFunction(V), param)

        # Computation of the Jacobian
        J = derivative(residual, u, TrialFunction(V))

        # We start the solution of the problem
        T = 1.0  # Total simulation time
        t = 0.0  # Starting simulation time
        log("Parameter continuation started")
        goOn = True

        # We iterate the solution of the nonlinear problem until we reach T=1
        # (i.e. the parameter reaches its maximum value set when we created the
        # object belonging to this class) or if we have halved the incremental
        # step 5 times in a row or if dt reaches 10^-6, otherwise we set
        # goOn = False
        # and the while cycle ends.
        while round(t, 10) < T and self._dt > 1e-6 and goOn is True:
            t += self._dt
            round(t, 8)
            param.assign(self._param_start + (self._param_end - self._param_start) * t)
            # We print the actual value of the control parameter
            log("Percentage completed: " + str(round(t * 100, 10)) + "%" +
                " " + self._param_name + ": " + str(round(float(param), 10)))
            ok = 0
            n_halving = 0
            while ok == 0:
                # We solve the nonlinear problem
                self.problem.modify_initial_guess(u, param)
                status = self.pc_nonlinear_solver(residual, u, bcs, J)
                if status[1] is True:
                    # The nonlinear solver converged! We call the monitor.
                    self.problem.monitor(u, param, self._save_file)
                    log("Nonlinear solver converged", success=True)
                    # We save the new solution in u0
                    u0.assign(u)
                    ok = 1
                else:
                    # The nonlinear solver did not converge
                    n_halving += 1
                    log("Nonlinear solver did not converge, halving step", warning=True)
                    # We halve the step
                    self._dt = self._dt / 2.
                    t += -self._dt
                    param.assign(self._param_start + (self._param_end - self._param_start) * t)
                    # We assign to u the solution found at the previous step
                    u.assign(u0)
                    if n_halving > 5:
                        # We halved the step 5 times, we give up.
                        ok = 1
                        log("Max halving reached! Ending simulation", warning=True)
                        goOn = False

    def pc_nonlinear_solver(self, residual, u, bcs, J):
        dolfin_problem = NonlinearVariationalProblem(residual, u, bcs, J)
        solver = NonlinearVariationalSolver(dolfin_problem)
        solver.parameters.update(self._solver_params)
        return solver.solve()
