# %% [markdown]
# # quri-parts-itensorモジュールのベンチマーク
#

# %% [markdown]
# 代表的な量子回路のMPSシミュレータのひとつであるITensorをchemistry-qulacsで利用できるようにするため、quri-parts-itensorモジュールを実装した。ここでは主に計算時間についてのqulacsとの比較を通してベンチマークを行う。
#

# %% [markdown]
# 基底関数系は 6-31g を用い、必要 qubit 数を 4,6,…,18 とするため活性空間は(4e, 2o)から(4e, 9o)まで 8 種用意した。直後の`molecule`の生成セルのみ必ず事前に実行が必要だが、それ以外の各種ベンチマークセルはすべて独立して動作するようになっている。

# %% [markdown]
# ## molecule の生成
#

# %% [markdown]
# 本ベンチマークではすべてのケースでLiH分子のハミルトニアンを用いる。ここではOpenFermionの`MolecularData`オブジェクトを生成する。

# %%

import json
import juliacall

jl = juliacall.Main
jl.Pkg.activate(".")
jl.Pkg.instantiate()
import json

print(f"{jl.Base.active_project()=}")
print(f"{jl.Base.current_project()=}")

import time
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from tqdm import tqdm

basis = "6-31g"
multiplicity = 1
charge = 0
distance = 1.6
geometry = [("Li", (0, 0, 0)), ("H", (0, 0, distance))]

pyscf_molecule = MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(pyscf_molecule)
n_orbs = molecule.n_orbitals
n_elecs = molecule.n_electrons


# 関数実行時間を返す関数
def measure_time(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start


# %% [markdown]
# ## ダミー計算
# quri-parts-itensorモジュールでは、JuliaCallライブラリを通して内部でJuliaプログラムを呼び出している。そのためITensorの初回利用時にJuliaのインストール等が行われ、待ち時間が発生する。ここでは以後のベンチマークの結果に影響が出ないよう、一度ダミーの計算を行い初回動作時のオーバーヘッドを消費しておく。

# %%
from quri_parts.algo.ansatz import HardwareEfficient
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.itensor.estimator import create_itensor_mps_parametric_estimator

it_estimator = create_itensor_mps_parametric_estimator()

ansatz = HardwareEfficient(4, 2)
state = ParametricCircuitQuantumState(4, ansatz)

operator = Operator({pauli_label("Z0"): 1.0})

it_estimator(operator, state, [0.0] * ansatz.parameter_count)


# %% [markdown]
# ## Estimator
# オブザーバブルと量子状態から期待値を計算する`Estimator`の実行時間を比較する。AnsatzはHardwareEfficientとUCCSDの2種類。

# %% [markdown]
# ### HardwareEfficient ansatz
#
# 回路depthに関わる`reps`は1, 10, 20とした。

# %%

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random

from quri_parts.algo.ansatz import HardwareEfficient
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.itensor.estimator import create_itensor_mps_parametric_estimator
from quri_parts.openfermion.transforms import jordan_wigner
from quri_parts.qulacs.estimator import create_qulacs_vector_parametric_estimator

op_mapper = jordan_wigner.get_of_operator_mapper()
ql_estimator = create_qulacs_vector_parametric_estimator()
it_estimator = create_itensor_mps_parametric_estimator()
it_estimator_maxdim_50 = create_itensor_mps_parametric_estimator(maxdim=50)

avg_times_ql = defaultdict(list)
avg_times_it = defaultdict(list)
avg_times_it_maxdim_50 = defaultdict(list)

hwe_reps = [1, 10, 20]

for reps in hwe_reps:
    print(f"{reps=}")
    for n_active_orbs in tqdm(range(n_elecs // 2, n_orbs - 1)):
        qubit_count = 2 * n_active_orbs
        hamiltonian = molecule.get_molecular_hamiltonian(
            range(n_elecs // 2), range(n_active_orbs)
        )
        qp_h = op_mapper(hamiltonian)
        param_circuit = LinearMappedUnboundParametricQuantumCircuit(qubit_count)
        for i in range(n_elecs):
            param_circuit.add_X_gate(i)
        ansatz = HardwareEfficient(qubit_count, reps)
        param_circuit.extend(ansatz)
        state = ParametricCircuitQuantumState(qubit_count, param_circuit)

        times_ql = []
        times_it = []
        times_it_maxdim_50 = []
        for i in range(3):
            params = [
                random.random() for _ in range(state.parametric_circuit.parameter_count)
            ]
            times_ql.append(measure_time(ql_estimator, qp_h, state, params))
            times_it.append(measure_time(it_estimator, qp_h, state, params))
            times_it_maxdim_50.append(
                measure_time(it_estimator_maxdim_50, qp_h, state, params)
            )
        avg_times_ql[reps].append(np.average(times_ql))
        avg_times_it[reps].append(np.average(times_it))
        avg_times_it_maxdim_50[reps].append(np.average(times_it_maxdim_50))


with open("avg_times_ql.json", "w") as f:
    json.dump(avg_times_ql, f)
with open("avg_times_it.json", "w") as f:
    json.dump(avg_times_it, f)
with open("avg_times_it_maxdim_50.json", "w") as f:
    json.dump(avg_times_it_maxdim_50, f)

# Plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)

x_data = [i for i in range(n_elecs, (n_orbs - 1) * 2, 2)]
for reps in hwe_reps:
    ax.scatter(x_data, avg_times_ql[reps], label=f"reps: {reps}, Qulacs", marker="o")
for reps in hwe_reps:
    ax.scatter(x_data, avg_times_it[reps], label=f"reps: {reps}, ITensor", marker="s")
for reps in hwe_reps:
    ax.scatter(
        x_data,
        avg_times_it_maxdim_50[reps],
        label=f"reps: {reps}, ITensor maxdim=50",
        marker="+",
    )

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_yscale("log")
ax.set_xlabel("Qubits")
ax.set_ylabel("Duration (sec.)")
plt.savefig("hwe.png")
