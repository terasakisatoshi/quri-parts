# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import sqrt

from quri_parts.core.estimator import (
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
    QuantumEstimator,
)
from quri_parts.core.estimator.sampling.pauli import (
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
)
from quri_parts.core.estimator.utils import is_estimatable
from quri_parts.core.measurement import (
    CommutablePauliSetMeasurement,
    CommutablePauliSetMeasurementFactory,
    PauliReconstructorFactory,
)
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, Operator
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    PauliSamplingShotsAllocator,
)
from quri_parts.core.state import CircuitQuantumState

from .estimator_helpers import (
    CircuitShotPairPreparationFunction,
    distribute_shots_among_pauli_sets,
    get_sampling_circuits_and_shots,
)


class _Estimate:
    def __init__(
        self,
        op: Operator,
        const: complex,
        pauli_sets: Sequence[CommutablePauliSet],
        pauli_recs: Sequence[PauliReconstructorFactory],
        sampling_counts: Sequence[MeasurementCounts],
    ):
        self._op = op
        self._const = const
        self._pauli_sets = pauli_sets
        self._pauli_recs = pauli_recs
        self._sampling_counts = sampling_counts

    @cached_property
    def value(self) -> complex:
        val = self._const
        for pauli_set, pauli_rec, counts in zip(
            self._pauli_sets, self._pauli_recs, self._sampling_counts
        ):
            val += general_pauli_sum_expectation_estimator(
                counts, pauli_set, self._op, pauli_rec
            )
        return val

    @cached_property
    def error(self) -> float:
        square_err: float = 0.0
        for pauli_set, pauli_rec, counts in zip(
            self._pauli_sets, self._pauli_recs, self._sampling_counts
        ):
            total_counts = sum(counts.values())
            var = general_pauli_sum_sample_variance(
                counts, pauli_set, self._op, pauli_rec
            )
            square_err += var / total_counts
        return sqrt(square_err)


@dataclass
class _ConstEstimate:
    value: complex
    error: float = 0.0


def get_estimate_from_sampling_result(
    op: Operator,
    measurement_groups: Iterable[CommutablePauliSetMeasurement],
    const: complex,
    sampling_counts: Iterable[MeasurementCounts],
) -> Estimate[complex]:
    """Converts sampling counts into the estimation of the operator's
    expectation value."""
    pauli_sets = tuple(m.pauli_set for m in measurement_groups)
    pauli_recs = tuple(m.pauli_reconstructor_factory for m in measurement_groups)
    return _Estimate(op, const, pauli_sets, pauli_recs, tuple(sampling_counts))


def sampling_estimate(
    op: Estimatable,
    state: CircuitQuantumState,
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
    circuit_shot_pair_prep_fn: CircuitShotPairPreparationFunction = get_sampling_circuits_and_shots,  # noqa: E501
) -> Estimate[complex]:
    """Estimate expectation value of a given operator with a given state by
    sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        op: An operator of which expectation value is estimated.
        state: A quantum state on which the operator expectation is evaluated.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
        circuit_shot_pair_prep_fn: A :class:`~CircuitShotPairPreparationFunction` that
            prepares the set of circuits to perform measurement with. It is default to
            a function that concatenates the measurement circuits after the state
            preparation circuit.
    Returns:
        The estimated value (can be accessed with :attr:`.value`) with standard error
            of estimation (can be accessed with :attr:`.error`).
    """
    assert is_estimatable(
        op, state
    ), "Number of qubits of the operator is too large to estimate."

    if not isinstance(op, Operator):
        op = Operator({op: 1.0})

    if len(op) == 0:
        return _ConstEstimate(0.0)

    if len(op) == 1 and PAULI_IDENTITY in op:
        return _ConstEstimate(op[PAULI_IDENTITY])

    const = op.constant
    measurements = measurement_factory(op)
    measurements = [m for m in measurements if m.pauli_set != {PAULI_IDENTITY}]

    shots_map = distribute_shots_among_pauli_sets(
        op, measurements, shots_allocator, total_shots
    )
    circuit_and_shots = circuit_shot_pair_prep_fn(state, measurements, shots_map)
    sampling_counts = sampler(circuit_and_shots)
    return get_estimate_from_sampling_result(op, measurements, const, sampling_counts)


def create_sampling_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> QuantumEstimator[CircuitQuantumState]:
    """Create a :class:`QuantumEstimator` that estimates operator expectation
    value by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """

    def estimator(op: Estimatable, state: CircuitQuantumState) -> Estimate[complex]:
        return sampling_estimate(
            op, state, total_shots, sampler, measurement_factory, shots_allocator
        )

    return estimator


def concurrent_sampling_estimate(
    operators: Collection[Estimatable],
    states: Collection[CircuitQuantumState],
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
    circuit_shot_pair_prep_fn: CircuitShotPairPreparationFunction = get_sampling_circuits_and_shots,  # noqa: E501
) -> Iterable[Estimate[complex]]:
    """Estimate expectation value of given operators with given states by
    sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        operators: Operators of which expectation value is estimated.
        states: Quantum states on which the operator expectation is evaluated.
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
        circuit_shot_pair_prep_fn: A :class:`~CircuitShotPairPreparationFunction` that
            prepares the set of circuits to perform measurement with. It is default to
            a function that concatenates the measurement circuits after the state
            preparation circuit.
    Returns:
        The estimated values (can be accessed with :attr:`.value`) with standard errors
            of estimation (can be accessed with :attr:`.error`).
    """
    num_ops = len(operators)
    num_states = len(states)

    if num_ops == 0:
        raise ValueError("No operator specified.")

    if num_states == 0:
        raise ValueError("No state specified.")

    if num_ops > 1 and num_states > 1 and num_ops != num_states:
        raise ValueError(
            f"Number of operators ({num_ops}) does not match"
            f"number of states ({num_states})."
        )

    if num_states == 1:
        states = [next(iter(states))] * num_ops
    if num_ops == 1:
        operators = [next(iter(operators))] * num_states

    return [
        sampling_estimate(
            op,
            state,
            total_shots,
            sampler,
            measurement_factory,
            shots_allocator,
            circuit_shot_pair_prep_fn,
        )
        for op, state in zip(operators, states)
    ]


def create_sampling_concurrent_estimator(
    total_shots: int,
    sampler: ConcurrentSampler,
    measurement_factory: CommutablePauliSetMeasurementFactory,
    shots_allocator: PauliSamplingShotsAllocator,
) -> ConcurrentQuantumEstimator[CircuitQuantumState]:
    """Create a :class:`ConcurrentQuantumEstimator` that estimates operator
    expectation value by sampling measurement.

    The sampling measurements are configured with arguments as follows.

    Args:
        total_shots: Total number of shots available for sampling measurements.
        sampler: A Sampler that actually performs the sampling measurements.
        measurement_factory: A function that performs Pauli grouping and returns
            a measurement scheme for Pauli operators constituting the original operator.
        shots_allocator: A function that allocates the total shots to Pauli groups to
            be measured.
    """

    def estimator(
        operators: Collection[Estimatable],
        states: Collection[CircuitQuantumState],
    ) -> Iterable[Estimate[complex]]:
        return concurrent_sampling_estimate(
            operators,
            states,
            total_shots,
            sampler,
            measurement_factory,
            shots_allocator,
        )

    return estimator
