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
    DOLFIN_PI,
    IntervalMesh,
    Constant,
    Expression,
    grad,
    dx,
    Function,
    FunctionSpace,
    FiniteElement,
    MixedElement,
    assign,
    project,
    parameters,
    sqrt,
    split,
    derivative,
    assemble)
import numpy as np
import scipy.integrate
import scipy.optimize
import os

parameters["allow_extrapolation"] = True


class GeneralProblem(object):
    """
    Generates a new object describing a nonlinear problem
    """

    def mesh(self):
        return NotImplementedError

    def function_space(self, mesh):
        return NotImplementedError

    def residual(self, u):
        return NotImplementedError

    def solver_parameters(self):
        return {}

    def monitor(self, solution, param, output_file=None):
        pass

    def initial_guess(self, V):
        return Function(V)

    def modify_initial_guess(self, u, param):
        pass


class CoilingU2sFree(GeneralProblem):

    def __init__(
            self,
            beta=0.01,
            sigma=0.01,
            L=20,
            n=1,
            n_inter=5000,
            u3s=0,
            output_folder="output"):
        Pi = DOLFIN_PI
        self.beta = Constant(beta)
        self.sigma = Constant(sigma)
        self.L = Constant(L)
        self.n_inter = n_inter
        self.n = n
        self.u3s = u3s
        self._number_postbuckling_solutions_found = 0
        self.output_folder = output_folder

        self.u2s_cr = ((self.beta * self.n**2 * Pi**2) / self.L**2 + self.sigma) / 2.

    def mesh(self):
        return IntervalMesh(self.n_inter, 0, self.L)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"u2s": Constant(0)}

    def residual(self, y, v, u2s):
        dy = grad(y)[0]
        # Definition of the nonlinear energy and of the weak problem
        W = ((u2s - y ** 2) ** 2 - (self.beta * dy ** 2) / (-1 + y ** 2) + self.sigma *
             (self.u3s - y * sqrt(1 - y ** 2)) ** 2) / 2.
        self.psi = W * dx
        F = derivative(self.psi, y, v)
        return F

    def solver_parameters(self):
        parameters = {
            'nonlinear_solver': 'snes',
            'snes_solver': {
                'error_on_nonconvergence': False,
                'linear_solver': 'mumps',
                'absolute_tolerance': 1e-10,
                'relative_tolerance': 1e-10,
                'maximum_iterations': 20,
            }
        }

        return parameters

    def modify_initial_guess(self, y, u2s):
        if float(u2s) > float(self.u2s_cr) and self._number_postbuckling_solutions_found < 3:
            if self._number_postbuckling_solutions_found == 0:
                # Amplitude predicted by the WNL analysis
                Pi = DOLFIN_PI
                if self.n == 0:
                    self.xi = sqrt(2 * u2s - self.sigma) / (sqrt(2) * sqrt(1 - self.sigma))
                    self.domega_1 = Constant(1)
                else:
                    self.xi = (2 * sqrt(-(self.beta * self.n ** 2 * Pi ** 2) + self.L ** 2 *
                                        (-self.sigma + 2 * u2s)))\
                        / sqrt((2 * self.beta * self.n ** 4 * Pi ** 4) / self.L ** 2
                               + 6 * self.n ** 2 * Pi ** 2 * (1 - self.sigma) -
                               (3 * self.L ** 2 * self.sigma ** 2 * self.u3s ** 2)
                               / (2. * self.beta))
                    self.domega_1 = self.n * Pi / self.L * \
                        Expression("cos(n*Pi*x[0]/L)", n=self.n, Pi=DOLFIN_PI, L=self.L, degree=4)

            initial_guess = project(self.xi * self.domega_1, y.function_space())
            y.assign(initial_guess)
            self._number_postbuckling_solutions_found += 1

    def boundary_conditions(self, mesh, V):
        return []

    def monitor(self, y, u2s, output_file):
        s = np.linspace(0, float(self.L), self.n_inter)
        domega = np.array([y((s_,)) for s_ in s])
        omega = scipy.integrate.cumtrapz(domega, s, initial=0)

        delta_omega = max(abs(omega))
        output_file = open(self.output_folder + "/bif_diag_data.txt", "a")
        output_file.write("{%.10f, %.10f}," % (u2s, delta_omega))
        output_file.close()

        # Energy
        theoretical_energy = self.L / 2 * (u2s**2 + self.u3s**2)
        actual_energy = assemble(self.psi)
        diff = actual_energy / theoretical_energy

        output_file = open(self.output_folder + "/energy_diff.txt", "a")
        output_file.write("{%.10f, %.10f}," % (u2s, diff))
        output_file.close()

        if not os.path.exists(self.output_folder + "/mathematica"):
            os.makedirs(self.output_folder + "/mathematica")
        mathematica_file = open(
            self.output_folder +
            "/mathematica/mathematica_u2s_%.8f.txt" %
            (float(u2s)),
            "w")
        with np.printoptions(threshold=np.inf):
            mathematica_file.write(np.array2string(
                np.array([s, omega]).transpose(), separator=', '))


class CoilingU2sPinned(CoilingU2sFree):
    def function_space(self, mesh):
        Velem = FiniteElement("CG", mesh.ufl_cell(), 1)
        Pelem = FiniteElement("R", mesh.ufl_cell(), 0)
        VPelem = MixedElement([Velem, Pelem])
        W = FunctionSpace(mesh, VPelem)
        return W

    def residual(self, yp, vq, u2s):
        y, p = split(yp)

        dy = grad(y)[0]
        W = ((u2s - y ** 2) ** 2 - (self.beta * dy ** 2) / (-1 + y ** 2) + self.sigma *
             (self.u3s - y * sqrt(1 - y ** 2)) ** 2) / 2.

        self.psi = W * dx + p * y * dx
        F = derivative(self.psi, yp, vq)
        return F

    def modify_initial_guess(self, yp, u2s):
        if float(u2s) > float(self.u2s_cr) and self._number_postbuckling_solutions_found < 3:
            if self._number_postbuckling_solutions_found == 0:
                # Amplitude predicted by the WNL analysis
                Pi = DOLFIN_PI
                self.xi = (2 * sqrt(-(self.beta * self.n ** 2 * Pi ** 2) + self.L ** 2 *
                                    (-self.sigma + 2 * u2s)))\
                    / sqrt((2 * self.beta * self.n ** 4 * Pi ** 4) / self.L ** 2
                           + 6 * self.n ** 2 * Pi ** 2 * (1 - self.sigma) -
                           (3 * self.L ** 2 * self.sigma ** 2 * self.u3s ** 2)
                           / (2. * self.beta))
                self.domega_1 = self.n * Pi / self.L * \
                    Expression("cos(n*Pi*x[0]/L)", n=self.n, Pi=DOLFIN_PI, L=self.L, degree=4)

            initial_guess = project(
                self.xi * self.domega_1,
                FunctionSpace(
                    yp.function_space().mesh(),
                    "CG",
                    1))
            assign(yp.sub(0), initial_guess)
            self._number_postbuckling_solutions_found += 1

    def monitor(self, y, u2s, output_file):
        s = np.linspace(0, float(self.L), self.n_inter)
        domega = np.array([y((s_,))[0] for s_ in s])
        omega = scipy.integrate.cumtrapz(domega, s, initial=0)

        delta_omega = max(abs(omega))
        output_file = open(self.output_folder + "/bif_diag_data.txt", "a")
        output_file.write("{%.10f, %.10f}," % (u2s, delta_omega))
        output_file.close()

        # Energy
        theoretical_energy = self.L / 2 * (u2s**2 + self.u3s**2)
        actual_energy = assemble(self.psi)
        diff = actual_energy - theoretical_energy

        output_file = open(self.output_folder + "/energy_diff.txt", "a")
        output_file.write("{%.10f, %.10f}," % (u2s, diff))
        output_file.close()

        if not os.path.exists(self.output_folder + "/mathematica"):
            os.makedirs(self.output_folder + "/mathematica")
        mathematica_file = open(
            self.output_folder +
            "/mathematica/mathematica_u2s_%.8f.txt" %
            (float(u2s)),
            "w")
        with np.printoptions(threshold=np.inf):
            mathematica_file.write(np.array2string(
                np.array([s, omega]).transpose(), separator=', '))


class CoilingFFree(GeneralProblem):

    def __init__(
            self,
            beta=0.01,
            sigma=0.01,
            L=10,
            n=1,
            u2s=0,
            u3s=0,
            n_inter=5000,
            output_folder="output"):
        Pi = DOLFIN_PI
        self.beta = Constant(beta)
        self.sigma = Constant(sigma)
        self.L = Constant(L)
        self.n_inter = n_inter
        self.n = n
        self.u2s = u2s
        self.u3s = u3s
        self._number_postbuckling_solutions_found = 0
        self.output_folder = output_folder

        self.F_cr = 2 * self.u2s - (self.n**2 * Pi**2 * self.beta) / self.L**2 - self.sigma

    def mesh(self):
        return IntervalMesh(self.n_inter, 0, self.L)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        return {"F": Constant(0)}

    def residual(self, y, v, F):
        dy = grad(y)[0]
        # Definition of the nonlinear energy and of the weak problem
        W = ((self.u2s - y ** 2) ** 2 +
             self.sigma * (self.u3s - y * sqrt(1 - y ** 2)) ** 2 -
             (self.beta * dy ** 2) / (-1 + y ** 2)) / 2.
        ext_forces = (F * sqrt(1 - y ** 2)) * dx
        self.psi = W * dx
        energy = self.psi - ext_forces
        FF = derivative(energy, y, v)
        return FF

    def solver_parameters(self):
        parameters = {
            'nonlinear_solver': 'snes',
            'snes_solver': {
                'error_on_nonconvergence': False,
                'linear_solver': 'mumps',
                'absolute_tolerance': 1e-10,
                'relative_tolerance': 1e-10,
                'maximum_iterations': 20,
            }
        }

        return parameters

    def modify_initial_guess(self, y, F):
        if float(F) < float(self.F_cr) and self._number_postbuckling_solutions_found < 3:
            # Amplitude predicted by the WNL analysis
            Pi = DOLFIN_PI
            if self.n == 0:
                # self.xi = sqrt(-F - self.sigma + 2 * self.u2s) / (sqrt(2)
                #                                                  * sqrt(1 - self.sigma))
                F_float = float(F)
                sigma_float = float(self.sigma)

                def eq(xi):
                    return (-F_float + np.sqrt(1 - xi**2) *
                            (- sigma_float - 2 * xi**2 + 2 * sigma_float * xi**2))
                self.xi = scipy.optimize.bisect(eq, 0.001, 0.3, xtol=1e-13)
                print(self.xi)
                self.domega_1 = Constant(1)
            else:
                self.xi = 2 * sqrt(2) * \
                    sqrt(-((F + (self.beta * self.n**2 * Pi**2) /
                            self.L**2 + self.sigma - 2 * self.u2s) /
                           ((self.beta * self.n**4 * Pi ** 4) / self.L**4 +
                            (3 * self.n**2 * Pi**2 * (4 - 5 * self.sigma + 2 * self.u2s)) /
                            self.L**2 - (3 * self.sigma**2 * self.u3s**2) / self.beta)))
                self.domega_1 = self.n * Pi / self.L * \
                    Expression("cos(n*Pi*x[0]/L)", n=self.n, Pi=DOLFIN_PI, L=self.L, degree=4)

            initial_guess = project(self.xi * self.domega_1, y.function_space())
            y.assign(initial_guess)
            self._number_postbuckling_solutions_found += 1

    def boundary_conditions(self, mesh, V):
        return []

    def monitor(self, y, F, output_file):
        s = np.linspace(0, float(self.L), self.n_inter)
        domega = np.array([y((s_,)) for s_ in s])
        omega = scipy.integrate.cumtrapz(domega, s, initial=0)

        delta_omega = max(abs(omega))
        output_file = open(self.output_folder + "/bif_diag_data.txt", "a")
        output_file.write("{%.10f, %.10f}," % (F, delta_omega))
        output_file.close()

        # Energy
        theoretical_energy = self.L / 2 * (self.u2s**2 + self.u3s**2)
        actual_energy = assemble(self.psi)
        diff = actual_energy - theoretical_energy

        output_file = open(self.output_folder + "/energy_diff.txt", "a")
        output_file.write("{%.10f, %.10f}," % (F, diff))
        output_file.close()

        # Force-displacement
        spos = assemble(sqrt(1 - y ** 2) * dx)
        output_file = open(self.output_folder + "/f_spos.txt", "a")
        output_file.write("{%.10f, %.10f}," % (spos, F))
        output_file.close()

        if not os.path.exists(self.output_folder + "/mathematica"):
            os.makedirs(self.output_folder + "/mathematica")
        mathematica_file = open(
            self.output_folder +
            "/mathematica/mathematica_F_%.8f.txt" %
            (float(F)),
            "w")
        with np.printoptions(threshold=np.inf):
            mathematica_file.write(np.array2string(
                np.array([s, omega]).transpose(), separator=', '))


class CoilingFPinned(CoilingFFree):
    def function_space(self, mesh):
        Velem = FiniteElement("CG", mesh.ufl_cell(), 1)
        Pelem = FiniteElement("R", mesh.ufl_cell(), 0)
        VPelem = MixedElement([Velem, Pelem])
        W = FunctionSpace(mesh, VPelem)
        return W

    def residual(self, yp, vq, F):
        y, p = split(yp)

        dy = grad(y)[0]
        # Definition of the nonlinear energy and of the weak problem
        W = -(F * sqrt(1 - y ** 2)) + \
            ((self.u2s - y ** 2) ** 2 +
             self.sigma * (self.u3s - y * sqrt(1 - y ** 2)) ** 2 -
             (self.beta * dy ** 2) / (-1 + y ** 2)) / 2.

        self.psi = W * dx + p * y * dx
        F = derivative(self.psi, yp, vq)
        return F

    def modify_initial_guess(self, yp, F):
        if float(F) < float(self.F_cr) and self._number_postbuckling_solutions_found < 3:
            if self._number_postbuckling_solutions_found == 0:
                # Amplitude predicted by the WNL analysis
                Pi = DOLFIN_PI
                self.xi = 2 * sqrt(2) * \
                    sqrt(-((F + (self.beta * self.n**2 * Pi**2) /
                            self.L**2 + self.sigma - 2 * self.u2s) /
                           ((self.beta * self.n**4 * Pi ** 4) / self.L**4 +
                            (3 * self.n**2 * Pi**2 * (4 - 5 * self.sigma + 2 * self.u2s)) /
                            self.L**2 - (3 * self.sigma**2 * self.u3s**2) / self.beta)))
                self.domega_1 = self.n * Pi / self.L * \
                    Expression("cos(n*Pi*x[0]/L)", n=self.n, Pi=DOLFIN_PI, L=self.L, degree=4)

            initial_guess = project(
                self.xi * self.domega_1,
                FunctionSpace(
                    yp.function_space().mesh(),
                    "CG",
                    1))
            assign(yp.sub(0), initial_guess)
            self._number_postbuckling_solutions_found += 1

    def monitor(self, y, F, output_file):
        s = np.linspace(0, float(self.L), self.n_inter)
        domega = np.array([y((s_,))[0] for s_ in s])
        omega = scipy.integrate.cumtrapz(domega, s, initial=0)

        delta_omega = max(abs(omega))
        output_file = open(self.output_folder + "/bif_diag_data.txt", "a")
        output_file.write("{%.10f, %.10f}," % (F, delta_omega))
        output_file.close()

        # Energy
        theoretical_energy = self.L / 2 * (self.u2s**2 + self.u3s**2)
        actual_energy = assemble(self.psi)
        diff = actual_energy - theoretical_energy

        output_file = open(self.output_folder + "/energy_diff.txt", "a")
        output_file.write("{%.10f, %.10f}," % (F, diff))
        output_file.close()

        # Force-displacement
        spos = assemble(sqrt(1 - y ** 2) * dx)
        output_file = open(self.output_folder + "/f_spos.txt", "a")
        output_file.write("{%.10f, %.10f}," % (spos, F))
        output_file.close()

        if not os.path.exists(self.output_folder + "/mathematica"):
            os.makedirs(self.output_folder + "/mathematica")
        mathematica_file = open(
            self.output_folder +
            "/mathematica/mathematica_F_%.8f.txt" %
            (float(F)),
            "w")
        with np.printoptions(threshold=np.inf):
            mathematica_file.write(np.array2string(
                np.array([s, omega]).transpose(), separator=', '))
