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
    let m = 3;
    let n = 2;
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
    let m = 3;
    let n = 2;
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
    let m = 3;
    let n = 2;
    let a = Array::new(&vec![1.0, 2.0, 4.0, 5.0, 3.0, 2.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = 2;
    let c: Array<f64> = a * b;
    let mut host_c = &mut vec![0.0; n * m];
    c.host(&mut host_c);

    let expected_array = &mut vec![2.0, 4.0, 8.0, 10.0, 6.0, 4.0];
    assert_eq!(expected_array, host_c);
}

#[test]
fn can_divide_matrix_and_scalar() {
    let m = 3;
    let n = 2;
    let a = Array::new(&vec![1.0, 2.0, 4.0, 5.0, 3.0, 2.0], Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = 2;
    let c: Array<f64> = a / b;
    let mut host_c = &mut vec![0.0; n * m];
    c.host(&mut host_c);

    let expected_array = &mut vec![0.5, 1.0, 2.0, 2.5, 1.5, 1.0];
    assert_eq!(expected_array, host_c);
}

#[test]
fn can_multiply_matrix_and_vector() {
    let m = 3;
    let n = 3;

    // 1 2 3
    // 4 5 6
    // 7 8 9
    // data is being read column by column!
    let data = &vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

    let a = Array::new(data, Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = Array::new(&vec![1.0, 1.0, 1.0], Dim4::new(&[n as u64, 1, 1, 1]));
    let c: Array<f64> = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    let mut host_c = &mut vec![0.0; m];
    c.host(&mut host_c);

    let expected_array = &mut vec![6.0, 15.0, 24.0];
    assert_eq!(expected_array, host_c);
}

#[test]
fn can_multiply_matrix_and_matrix() {
    let m = 2;
    let n = 2;
    let o = 2;

    // 1 3
    // 2 5
    // data is being read column by column!
    let data_a = &vec![1.0, 2.0, 3.0, 5.0];

    // 0 1
    // 3 2
    // data is being read column by column!
    let data_b = &vec![0.0, 3.0, 1.0, 2.0];

    let a = Array::new(data_a, Dim4::new(&[m as u64, n as u64, 1, 1]));
    let b = Array::new(data_b, Dim4::new(&[n as u64, o as u64, 1, 1]));
    let c: Array<f64> = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    let mut host_c = &mut vec![0.0; m * o];
    c.host(&mut host_c);

    //  9  7
    // 15 12
    // data is being read column by column!
    let expected_array = &mut vec![9.0, 15.0, 7.0, 12.0];

    assert_eq!(expected_array, host_c);
}

#[test]
fn can_mulitply_matrix_and_identity() {
    let m = 3;
    let n = 2;
    let a = randu::<f64>(Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_a = &mut vec![0.0; m * n];
    a.host(&mut host_a);

    let i = identity::<f64>(Dim4::new(&[n as u64, n as u64, 1, 1]));

    let c = matmul(&a, &i, MatProp::NONE, MatProp::NONE);
    let mut host_c = &mut vec![0.0; m * n];
    c.host(&mut host_c);

    assert_eq!(host_a, host_c);
}

#[test]
fn can_mulitply_identity_and_matrix() {
    let m = 3;
    let n = 2;
    let a = randu::<f64>(Dim4::new(&[m as u64, n as u64, 1, 1]));
    let mut host_a = &mut vec![0.0; m * n];
    a.host(&mut host_a);

    let i = identity::<f64>(Dim4::new(&[m as u64, m as u64, 1, 1]));

    let c = matmul(&i, &a, MatProp::NONE, MatProp::NONE);
    let mut host_c = &mut vec![0.0; m * n];
    c.host(&mut host_c);

    assert_eq!(host_a, host_c);
}
