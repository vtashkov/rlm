extern crate arrayfire;

use arrayfire::*;

#[test]
fn can_create_array_with_one_constant_value() {
    let value = 1;
    let a = constant(value, Dim4::new(&[1, 1, 1, 1]));
    let mut b = &mut vec![0];
    a.host(&mut b);
    assert_eq!(value, b[0]);
}

#[test]
fn can_create_array_with_n_constant_values() {
    let n = 3;
    let value = 1;
    let a = constant(value, Dim4::new(&[n as u64, 1, 1, 1]));
    let mut b = &mut vec![0; n];
    a.host(&mut b);

    let mut expected = &mut vec![1, 1, 1];
    assert_eq!(n, b.len());
    assert_eq!(expected, b);
}
