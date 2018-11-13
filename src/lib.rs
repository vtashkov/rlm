
#[cfg(test)]
mod tests {

    extern crate arrayfire as af;

    use self::af::*;

    #[test]
    fn it_works() {
        info();
        assert_eq!(2, 2);
    }
}
