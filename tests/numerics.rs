//! Independent f64 equations and central differences, without `PyTorch` or autodiff.
use burn::tensor::{Device, Tensor, TensorData};
use burn_xlstm::functional::{MlstmInput, MlstmState, SlstmGates, SlstmState};

const TIME: usize = 6;
const DIM: usize = 3;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "test inputs deliberately round to f32"
)]
fn inputs(size: usize, phase: f64) -> Vec<f32> {
    (0..size)
        .map(|i| ((f64::from(u32::try_from(i).unwrap()) * 1.37 + phase).sin() * 1.4) as f32)
        .collect()
}

fn weights() -> Vec<f32> {
    inputs(TIME * DIM, 0.81)
}

// Direct, unstabilized paper equations are safe for these moderate test inputs.
// mLSTM deliberately uses negligible epsilon (1e-12) in the tensor path, so the
// oracle needs neither implementation's log-scale convention.
#[expect(
    clippy::needless_range_loop,
    reason = "explicit matrix indices make the independent equations auditable"
)]
fn matrix_equations(values: &[f64]) -> Vec<f64> {
    let span = TIME * DIM;
    let (query, rest) = values.split_at(span);
    let (key, rest) = rest.split_at(span);
    let (value, rest) = rest.split_at(span);
    let (input, forget) = rest.split_at(TIME);
    let mut memory = [[0.0; DIM]; DIM];
    let mut normalizer = [0.0; DIM];
    let mut outputs = Vec::with_capacity(span);
    for time in 0..TIME {
        let i = input[time].exp();
        let f = sigmoid(forget[time]);
        for row in 0..DIM {
            let k = key[time * DIM + row] / 3.0_f64.sqrt();
            normalizer[row] = f * normalizer[row] + i * k;
            for col in 0..DIM {
                memory[row][col] = f * memory[row][col] + i * k * value[time * DIM + col];
            }
        }
        let dot: f64 = (0..DIM)
            .map(|row| query[time * DIM + row] * normalizer[row])
            .sum();
        for col in 0..DIM {
            let numerator: f64 = (0..DIM)
                .map(|row| query[time * DIM + row] * memory[row][col])
                .sum();
            outputs.push(numerator / dot.abs().max(1.0));
        }
    }
    outputs
}

fn scalar_equations(gates: &[f64]) -> Vec<f64> {
    let mut memory = [0.0; DIM];
    let mut normalizer = [0.0; DIM];
    let mut outputs = Vec::with_capacity(TIME * DIM);
    for time in 0..TIME {
        for lane in 0..DIM {
            let gate = |g| gates[(time * 4 + g) * DIM + lane];
            let i = gate(0).exp();
            let f = sigmoid(gate(1));
            memory[lane] = f * memory[lane] + i * gate(2).tanh();
            normalizer[lane] = f * normalizer[lane] + i;
            outputs.push(sigmoid(gate(3)) * memory[lane] / normalizer[lane]);
        }
    }
    outputs
}

fn compare(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - e).abs();
        assert!(
            a.is_finite() && e.is_finite() && error <= tolerance * (1.0 + e.abs()),
            "coordinate {index}: {a} vs {e}, error {error}"
        );
    }
}

fn differences(values: &[f32], equations: fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    let values: Vec<f64> = values.iter().copied().map(f64::from).collect();
    let weights = weights();
    let objective = |v: &[f64]| -> f64 {
        equations(v)
            .iter()
            .zip(&weights)
            .map(|(a, w)| a * f64::from(*w))
            .sum()
    };
    let mut gradients = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[index] += 1e-5;
        minus[index] -= 1e-5;
        gradients.push((objective(&plus) - objective(&minus)) / 2e-5);
    }
    gradients
}

fn read<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec().unwrap()
}

#[test]
#[expect(
    clippy::single_range_in_vec_init,
    reason = "Burn slicing takes one range per tensor axis"
)]
fn matrix_outputs_and_all_input_derivatives_match_independent_equations() {
    let device = Device::flex().autodiff();
    for phase in [0.2, 1.7, 3.1] {
        let values = inputs(3 * TIME * DIM + 2 * TIME, phase);
        let expected = matrix_equations(&values.iter().copied().map(f64::from).collect::<Vec<_>>());
        let numerical = differences(&values, matrix_equations);
        for parallel in [false, true] {
            let all =
                Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), &device)
                    .require_grad();
            let span = TIME * DIM;
            let sequence = |start| {
                all.clone()
                    .slice([start..start + span])
                    .reshape([1, 1, TIME, DIM])
            };
            let gate = |start| {
                all.clone()
                    .slice([start..start + TIME])
                    .reshape([1, 1, TIME, 1])
            };
            let data = MlstmInput {
                query: sequence(0),
                key: sequence(span),
                value: sequence(2 * span),
                input: gate(3 * span),
                forget: gate(3 * span + TIME),
            };
            let output = if parallel {
                data.parallel(1e-12)
            } else {
                let mut state = MlstmState::zeros([1, 1, DIM], &device);
                let mut outputs = Vec::new();
                for time in 0..TIME {
                    let take =
                        |tensor: &Tensor<4>| tensor.clone().slice([0..1, 0..1, time..time + 1]);
                    let (output, next) = MlstmInput {
                        query: take(&data.query),
                        key: take(&data.key),
                        value: take(&data.value),
                        input: take(&data.input),
                        forget: take(&data.forget),
                    }
                    .step(state, 1e-12);
                    outputs.push(output);
                    state = next;
                }
                Tensor::cat(outputs, 2)
            };
            compare(&read(output.clone()), &expected, 3e-6);
            let weights =
                Tensor::<4>::from_data(TensorData::new(weights(), [1, 1, TIME, DIM]), &device);
            let gradients = (output * weights).sum().backward();
            compare(&read(all.grad(&gradients).unwrap()), &numerical, 2e-5);
        }
    }
}

#[test]
fn scalar_outputs_and_all_gate_derivatives_match_independent_equations() {
    let device = Device::flex().autodiff();
    for phase in [0.2, 1.7, 3.1] {
        let values = inputs(TIME * 4 * DIM, phase);
        let all =
            Tensor::<4>::from_data(TensorData::new(values.clone(), [TIME, 1, 4, DIM]), &device)
                .require_grad();
        let mut state = SlstmState::zeros([1, 1, DIM], &device);
        let mut outputs = Vec::new();
        for time in 0..TIME {
            let gate = |g| {
                all.clone()
                    .slice([time..time + 1, 0..1, g..g + 1])
                    .reshape([1, 1, DIM])
            };
            state = SlstmGates {
                input: gate(0),
                forget: gate(1),
                candidate: gate(2),
                output: gate(3),
            }
            .step(state);
            outputs.push(state.hidden.clone());
        }
        let output = Tensor::cat(outputs, 1);
        let expected = scalar_equations(&values.iter().copied().map(f64::from).collect::<Vec<_>>());
        compare(&read(output.clone()), &expected, 3e-6);
        let weights = Tensor::<3>::from_data(TensorData::new(weights(), [1, TIME, DIM]), &device);
        let gradients = (output * weights).sum().backward();
        compare(
            &read(all.grad(&gradients).unwrap()),
            &differences(&values, scalar_equations),
            2e-5,
        );
    }
}
