struct LinearRegressionModel {
    theta0: f64,
    theta1: f64,
}

impl LinearRegressionModel {
    fn predict(&self, x: f64) -> f64 {
        self.theta0 + x * self.theta1
    }

    fn cost(&self, x: &[f64], y: &[f64]) -> f64 {
        let m = y.len();
        let mut sum = 0.0;
        for i in 0..m {
            sum += (self.predict(x[i]) - y[i]).powi(2);
        }
        sum / (2.0 * m as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_model_always_predicts_zero() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };

        assert_eq!(0.0, model.predict(0.0));
        assert_eq!(0.0, model.predict(1.0));
        assert_eq!(0.0, model.predict(-1.0));
    }

    #[test]
    fn zero_one_model_always_predicts_x() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 1.0 };

        assert_eq!(0.0, model.predict(0.0));
        assert_eq!(1.0, model.predict(1.0));
        assert_eq!(-1.0, model.predict(-1.0));
    }

    #[test]
    fn one_one_model_always_predicts_x_plus_one() {
        let model = LinearRegressionModel { theta0: 1.0, theta1: 1.0 };

        assert_eq!(1.0, model.predict(0.0));
        assert_eq!(2.0, model.predict(1.0));
        assert_eq!(0.0, model.predict(-1.0));
    }

    #[test]
    fn all_zero_model_with_one_zero_y_cost_zero() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0];
        let y = &[0.0];

        assert_eq!(0.0, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn all_zero_model_with_multiple_zero_y_cost_zero() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0, 0.0, 0.0];
        let y = &[0.0, 0.0, 0.0];

        assert_eq!(0.0, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn all_zero_model_with_one_for_y_cost_zero_point_five() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0];
        let y = &[1.0];

        assert_eq!(0.5, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn all_zero_model_with_one_and_zero_for_y_cost_zero_point_twenty_five() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0, 0.0];
        let y = &[1.0, 0.0];

        assert_eq!(0.25, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn all_zero_model_with_zero_and_one_for_y_cost_zero_point_twenty_five() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0, 0.0];
        let y = &[0.0, 1.0];

        assert_eq!(0.25, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn all_zero_model_with_one_and_five_for_y_cost_six_point_five() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 0.0 };
        let x = &[0.0, 0.0];
        let y = &[1.0, 5.0];

        assert_eq!(6.5, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn zero_one_model_with_one_for_x_and_one_for_y_cost_zero() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 1.0 };
        let x = &[1.0];
        let y = &[1.0];

        assert_eq!(0.0, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn zero_two_model_with_zero_point_five_for_x_and_one_for_y_cost_zero() {
        let model = LinearRegressionModel { theta0: 0.0, theta1: 2.0 };
        let x = &[0.5];
        let y = &[1.0];

        assert_eq!(0.0, model.cost(&x[..], &y[..]));
    }

    #[test]
    fn one_two_model_with_zero_point_five_for_x_and_two_for_y_cost_zero() {
        let model = LinearRegressionModel { theta0: 1.0, theta1: 2.0 };
        let x = &[0.5];
        let y = &[2.0];

        assert_eq!(0.0, model.cost(&x[..], &y[..]));
    }
}