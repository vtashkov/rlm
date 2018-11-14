extern crate arrayfire;

use arrayfire::*;

#[test]
fn can_create_array_with_one_constant_value() {
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[1, 1, 1, 1]));
    let mut host_array = &mut vec![0];
    gpu_array.host(&mut host_array);
    assert_eq!(value, host_array[0]);
}

#[test]
fn can_create_array_with_n_constant_values() {
    let n = 3;
    let value = 1;
    let gpu_array = constant(value, Dim4::new(&[n as u64, 1, 1, 1]));
    let mut host_array = &mut vec![0; n];
    gpu_array.host(&mut host_array);

    let expected_array = &mut vec![1, 1, 1];
    assert_eq!(n, host_array.len());
    assert_eq!(expected_array, host_array);
}
