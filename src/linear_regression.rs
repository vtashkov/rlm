struct LinearRegressionModel {
    theta0: f64,
    theta1: f64,
}

impl LinearRegressionModel {
    fn predict(&self, x: f64) -> f64 {
        self.theta0 + x * self.theta1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_model_always_predicts_zero() {
        let model = LinearRegressionModel {
            theta0: 0.0,
            theta1: 0.0,
        };

        assert_eq!(0.0, model.predict(0.0));
        assert_eq!(0.0, model.predict(1.0));
        assert_eq!(0.0, model.predict(-1.0));
    }

    #[test]
    fn zero_one_model_always_predicts_x() {
        let model = LinearRegressionModel {
            theta0: 0.0,
            theta1: 1.0,
        };

        assert_eq!(0.0, model.predict(0.0));
        assert_eq!(1.0, model.predict(1.0));
        assert_eq!(-1.0, model.predict(-1.0));
    }

    #[test]
    fn one_one_model_always_predicts_x_plus_one() {
        let model = LinearRegressionModel {
            theta0: 1.0,
            theta1: 1.0,
        };

        assert_eq!(1.0, model.predict(0.0));
        assert_eq!(2.0, model.predict(1.0));
        assert_eq!(0.0, model.predict(-1.0));
    }
}