import json
from matplotlib import pyplot as plt
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

hwe_reps = [1, 10, 20]


with open("avg_times_ql.json", "r") as f:
    avg_times_ql = json.load(f)
with open("avg_times_it.json", "r") as f:
    avg_times_it = json.load(f)
with open("avg_times_it_maxdim_50.json", "r") as f:
    avg_times_it_maxdim_50 = json.load(f)

# Plot
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)

x_data = [i for i in range(n_elecs, (n_orbs - 1) * 2, 2)]
for reps in hwe_reps:
    reps = str(reps)
    ax.scatter(x_data, avg_times_ql[reps], label=f"reps: {reps}, Qulacs", marker="o")
for reps in hwe_reps:
    reps = str(reps)
    ax.scatter(x_data, avg_times_it[reps], label=f"reps: {reps}, ITensor", marker="s")
for reps in hwe_reps:
    reps = str(reps)
    ax.scatter(
        x_data,
        avg_times_it_maxdim_50[reps],
        label=f"reps: {reps}, ITensor maxdim=50",
        marker="+",
    )

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Qubits")
ax.set_ylabel("Duration (sec.)")
plt.savefig("hwe_ylinear.png", bbox_inches="tight")

ax.set_yscale("log")
plt.savefig("hwe_ylog.png", bbox_inches="tight")
