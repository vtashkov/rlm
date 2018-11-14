extern crate arrayfire;

use arrayfire::*;

#[test]
fn can_get_empty_array_dimensions() {
    let n = 3;
    let m = 5;
    let gpu_array = Array::<f64>::new_empty(Dim4::new(&[n, m, 1, 1]));
    let dim = gpu_array.dims();

    assert_eq!(n, dim[0]);
    assert_eq!(m, dim[1]);
    assert_eq!(1, dim[2]);
    assert_eq!(1, dim[3]);
}

#[test]
fn can_create_array_with_one_constant_integer_value() {
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[1, 1, 1, 1]));
    let mut host_array = &mut vec![0];
    gpu_array.host(&mut host_array);
    assert_eq!(value, host_array[0]);
}

#[test]
fn can_create_array_with_one_constant_double_value() {
    let value = 1.0;
    let gpu_array = constant(value, Dim4::new(&[1, 1, 1, 1]));
    let mut host_array = &mut vec![0.0];
    gpu_array.host(&mut host_array);
    assert_eq!(value, host_array[0]);
}

#[test]
fn can_create_array_with_one_value_from_host() {
    let data = &vec![1.0];
    let gpu_array = Array::new(data, Dim4::new(&[1, 1, 1, 1]));
    let mut host_array = &mut vec![0.0];
    gpu_array.host(&mut host_array);

    assert_eq!(data, host_array);
}

#[test]
fn can_create_array_with_n_constant_integer_values() {
    let n = 3;
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[n as u64, 1, 1, 1]));
    let mut host_array = &mut vec![0; n];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1, 1, 1];
    assert_eq!(n, host_array.len());
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_n_constant_double_values() {
    let n = 3;
    let value: f64 = 1.0;
    let gpu_array = constant(value, Dim4::new(&[n as u64, 1, 1, 1]));
    let mut host_array = &mut vec![0.0; n];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1.0, 1.0, 1.0];
    assert_eq!(n, host_array.len());
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_n_values_from_host() {
    let n = 3;
    let data = &vec![1.0, 2.0, 3.0];
    let gpu_array = Array::new(data, Dim4::new(&[n as u64, 1, 1, 1]));
    let mut host_array = &mut vec![0.0; n];
    gpu_array.host(&mut host_array);

    assert_eq!(data, host_array);
}

#[test]
fn can_create_array_with_n_x_m_constant_integer_values() {
    let n = 3;
    let m = 3;
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[n as u64, m as u64, 1, 1]));
    let mut host_array = &mut vec![0; n * m];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1, 1, 1, 1, 1, 1, 1, 1, 1];
    assert_eq!(n * m, host_array.len());
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_n_x_m_constant_double_values() {
    let n = 3;
    let m = 3;
    let value = 1.0;
    let gpu_array = constant(value, Dim4::new(&[n as u64, m as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; n * m];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(n * m, host_array.len());
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_n_x_m_values_from_host() {
    let n = 2;
    let m = 3;
    let data = &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu_array = Array::new(data, Dim4::new(&[n as u64, m as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; n * m];
    gpu_array.host(&mut host_array);

    assert_eq!(data, host_array);
}

#[test]
fn can_create_array_with_n_x_m_uniformly_distributed_values() {
    let n = 3;
    let m = 3;
    let gpu_array = randu::<f64>(Dim4::new(&[n as u64, m as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; n * m];
    gpu_array.host(&mut host_array);

    for i in 0..n * m {
        assert!(host_array[i] >= 0.0);
        assert!(host_array[i] <= 1.0);
    }
}

#[test]
fn can_create_3_x_3_identity_matrix() {
    let gpu_array = identity::<f64>(Dim4::new(&[3, 3, 1, 1]));
    let mut host_array = &mut vec![0.0; 3 * 3];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(3 * 3, host_array.len());
    assert_eq!(expected_array, host_array);
}
