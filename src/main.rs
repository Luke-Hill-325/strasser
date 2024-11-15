use nalgebra::base::*;
use nalgebra::matrix;
use std::iter::zip;

fn main() {
    let m = matrix![1.0,3.0,3.0,6.0;
                    4.0,2.0,8.0,2.0;
                    3.0,3.0,4.0,5.0;
                    2.0,6.0,3.0,1.0];
    let m_inverse = m.try_inverse().unwrap();

    let mult = mul_traditional(&m, &m_inverse);

    println!("{:.3}\n{:.3}\n{:.3}", m, m_inverse, mult);
}

fn mul_traditional(a: &Matrix4<f64>, b: &Matrix4<f64>) -> Matrix4<f64> {
    let b = &b.transpose();
    return Matrix4::<f64>::from_fn(|i, j| 
                                    a.row(i).zip_fold(&b.row(j), 0_f64, |acc, a, b| acc + a * b)
                                    );
}
