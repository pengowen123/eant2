#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#[cfg(test)]
mod tests {
    use cge::functions as f;
    use cge::network::Network;
    use cge::stack::Stack;
    use cge::node::Node;

    const EPS: f64 = 0.0001;

    fn eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn eqv(a: Vec<f64>, b: Vec<f64>) -> bool {
        for n in 0..a.len() {
            if !((a[n] - b[n]).abs() < EPS) {
                return false;
            }
        }

        true
    }

    #[test]
    fn magnitude() -> () {
        let data = vec![2.0, 2.0];
        let data_2 = 2.82842;
        assert!(eq(f::magnitude(&data), data_2));
    }

    #[test]
    fn normalize() -> () {
        let data = vec![2.0, 2.0];
        let data_2 = vec![0.7071, 0.7071];
        assert!(eqv(f::normalize(&data), data_2));
    }

    #[test]
    fn concat() -> () {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let data_2 = vec![1.0, 2.0, 3.0, 4.0];
        assert!(eqv(f::concat(&data), data_2));
    }

    #[test]
    fn reverse() -> () {
        let data = vec![1.0, 2.0];
        let data_2 = vec![2.0, 1.0];
        assert!(eqv(f::reverse(&data), data_2));
    }

    #[test]
    fn mul_vec() -> () {
        let data = vec![1.0, 1.0];
        let data_2 = 2.0;
        let data_3 = vec![2.0, 2.0];
        assert!(eqv(f::mul_vec(&data, data_2), data_3));
    }

    #[test]
    fn sum_vec() -> () {
        let data = vec![1.0, 1.0];
        let data_2 = 2.0;
        assert!(eq(f::sum_vec(&data), data_2));
    }

    #[test]
    fn add_vec() -> () {
        let data = vec![1.0, 1.0];
        let data_2 = vec![2.0, 2.0];
        let data_3 = vec![3.0, 3.0];
        assert!(eqv(f::add_vec(&data, &data_2), data_3));
    }

    #[test]
    fn sub_vec() -> () {
        let data = vec![1.0, 1.0];
        let data_2 = vec![1.0, 1.0];
        let data_3 = vec![0.0, 0.0];
        assert!(eqv(f::sub_vec(&data, &data_2), data_3));
    }

    #[test]
    fn prod_vec() -> () {
        let data = vec![2.0, 3.0, 4.0];
        let data_2 = 24.0;
        assert!(eq(f::prod_vec(&data), data_2));
    }

    #[test]
    fn div_vec() -> () {
        let data = vec![2.0, 2.0];
        let data_2 = 2.0;
        let data_3 = vec![1.0, 1.0];
        assert!(eqv(f::div_vec(&data, data_2), data_3));
    }

    #[test]
    fn negative_vec() -> () {
        let data = vec![1.0, 1.0];
        let data_2 = vec![-1.0, -1.0];
        assert!(eqv(f::negative_vec(&data), data_2));
    }

    #[test]
    fn vector_by_vector() -> () {
        let data = vec![2.0, 2.0];
        let data_2 = vec![2.0, 2.0];
        let data_3 = 8.0;
        assert!(eq(f::vector_by_vector(&data, &data_2), data_3));
    }

    #[test]
    fn mul_vec_2() -> () {
        let data = vec![2.0, 3.0];
        let data_2 = vec![2.0, 3.0];
        let data_3 = vec![4.0, 9.0];
        assert!(eqv(f::mul_vec_2(&data, &data_2), data_3));
    }
}
