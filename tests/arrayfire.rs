extern crate arrayfire;

use arrayfire::*;

#[test]
fn can_get_empty_array_dimensions() {
    let m = 3;
    let n = 5;
    let gpu_array = Array::<f64>::new_empty(Dim4::new(&[m, n, 1, 1]));
    let dim = gpu_array.dims();

    assert_eq!(m, dim[0]);
    assert_eq!(n, dim[1]);
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
fn can_create_array_with_m_x_n_constant_integer_values() {
    let m = 3;
    let n = 3;
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_array = &mut vec![0; m * n];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1, 1, 1, 1, 1, 1, 1, 1, 1];
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_m_x_n_constant_double_values() {
    let m = 3;
    let n = 3;
    let value = 1.0;
    let gpu_array = constant(value, Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; m * n];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_create_array_with_m_x_n_values_from_host() {
    let m = 2;
    let n = 3;
    let data = &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu_array = Array::new(data, Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; m * n];
    gpu_array.host(&mut host_array);

    assert_eq!(data, host_array);
}

#[test]
fn can_create_array_with_m_x_n_uniformly_distributed_values() {
    let m = 3;
    let n = 3;
    let gpu_array = randu::<f64>(Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_array = &mut vec![0.0; m * n];
    gpu_array.host(&mut host_array);

    for i in 0..m * n {
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
    assert_eq!(expected_array, host_array);
}

#[test]
fn can_add_matrices() {
    let m = 2;
    let n = 3;
    let a = Array::new(&vec![1.0, 3.0, 1.0, 1.0, 0.0, 0.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = Array::new(&vec![0.0, 0.0, 5.0, 7.0, 5.0, 0.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let c = a + b;
    let mut host_c = &mut vec![0.0; n * m];
    c.host(&mut host_c);

    let expected_array = &mut vec![1.0, 3.0, 6.0, 8.0, 5.0, 0.0];
    assert_eq!(expected_array, host_c);
}

#[test]
fn can_subtract_matrices() {
    let m = 2;
    let n = 3;
    let a = Array::new(&vec![1.0, 3.0, 1.0, 1.0, 0.0, 0.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = Array::new(&vec![0.0, 0.0, 5.0, 7.0, 5.0, 0.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let c = a - b;
    let mut host_c = &mut vec![0.0; n * m];
    c.host(&mut host_c);

    let expected_array = &mut vec![1.0, 3.0, -4.0, -6.0, -5.0, 0.0];
    assert_eq!(expected_array, host_c);
}

#[test]
fn can_multiply_matrix_and_scalar() {
    let m = 2;
    let n = 3;
    let a = Array::new(&vec![1.0, 2.0, 4.0, 5.0, 3.0, 2.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = 2.0;
    let c: Array<f64> = a * b;
    let mut host_c = &mut vec![0.0; n * m];
    c.host(&mut host_c);

    let expected_array = &mut vec![2.0, 4.0, 8.0, 10.0, 6.0, 4.0];
    assert_eq!(expected_array, host_c);
}