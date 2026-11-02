use burn::tensor::{Device, Tensor, TensorData};
use serde_json::Value;

pub fn fixture() -> Value {
    serde_json::from_str(include_str!("../fixtures/nxai.json")).unwrap()
}

pub fn tensor<const D: usize>(value: &Value, device: &Device) -> Tensor<D> {
    let shape: Vec<usize> = serde_json::from_value(value["shape"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(value["values"].clone()).unwrap();
    Tensor::from_data(TensorData::new(values, shape), device)
}

pub fn close<const D: usize>(actual: Tensor<D>, expected: Tensor<D>, tolerance: f32) {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.into_data().try_to_vec::<f32>().unwrap();
    let expected = expected.into_data().try_to_vec::<f32>().unwrap();
    let mut worst = 0.0_f32;
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let error = (actual - expected).abs() / (1.0 + expected.abs());
        worst = worst.max(error);
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "coordinate {index}: got {actual}, expected {expected}, scaled error {error} (worst {worst})"
        );
    }
}
