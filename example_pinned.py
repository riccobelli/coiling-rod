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


from continuation import ParameterContinuation
from problems import CoilingU2sPinned, CoilingFPinned

L = 40
nu = 0.35

problem = CoilingU2sPinned(
    beta=0.01,  # t/h = 0.1
    sigma=0.01 * 2 / (1 + nu),
    L=L,
    n=1,  # Single perversion mode
    output_folder="output_u2s_pinned")

XDMF_options = {"flush_output": True,
                "functions_share_mesh": True,
                "rewrite_function_mesh": False}
analysis = ParameterContinuation(
    problem,
    "u2s",
    start=0,
    end=0.4,
    dt=0.0025,
    saving_file_parameters=XDMF_options,
    output_folder="output_u2s_pinned")
analysis.run()

problem = CoilingFPinned(
    beta=0.01,
    sigma=0.01 * 2 / (1 + nu),
    L=L,
    n=1,
    output_folder="output_F_pinned")

XDMF_options = {"flush_output": True,
                "functions_share_mesh": True,
                "rewrite_function_mesh": False}
analysis = ParameterContinuation(
    problem,
    "F",
    start=0,
    end=-1,
    dt=.001,
    saving_file_parameters=XDMF_options,
    output_folder="output_F_pinned")
analysis.run()
