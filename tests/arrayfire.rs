extern crate arrayfire;

use arrayfire::*;

#[test]
fn can_create_array_with_one_constant_value() {
    let value = 0;
    let a = constant(value, Dim4::new(&[1, 1, 1, 1]));
    let mut b = &mut vec![0];
    a.host(&mut b);
    assert_eq!(value, b[0]);
}
