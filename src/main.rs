use nalgebra::base::*;
use nalgebra::matrix;
use nalgebra::dmatrix;
use std::iter::zip;
use std::time::Instant;

fn main() {
    let m = matrix![1.0,3.0,3.0,6.0;
                    4.0,2.0,8.0,2.0;
                    3.0,3.0,4.0,5.0;
                    2.0,6.0,3.0,1.0];
    let m_inverse = m.try_inverse().unwrap();

    let mut now = Instant::now();
    let strass = strassen(&m, &m_inverse);
    let strassen_bench = now.elapsed();
    now = Instant::now();
    let mult = mul_4x4(&m, &m_inverse);
    let basic_bench = now.elapsed();

    println!("m: {:.3}\ninverse:\n{:.3}\nmult:\n{:.3}\nstrassen:\n{:.3}\nbasic_benchmark: {:?}\nstrassen_benchmark: {:?}", m, m_inverse, mult, strass, basic_bench, strassen_bench);
}

fn as_submatrices(m: &Matrix4<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    return (DMatrix::<f64>::from(m.index((..2,..2))),
            DMatrix::<f64>::from(m.index((..2,2..))),
            DMatrix::<f64>::from(m.index((2..,..2))),
            DMatrix::<f64>::from(m.index((2..,2..))));
}

fn mul_4x4(a: &Matrix4<f64>, b: &Matrix4<f64>) -> Matrix4<f64> {
    let b = &b.transpose();
    return Matrix4::<f64>::from_fn(|i, j| 
                                    a.row(i).zip_fold(&b.row(j), 0_f64, |acc, a, b| acc + a * b)
                                    );
}

fn mul_d(a: &DMatrix<f64>, b: &DMatrix<f64>, d: usize) -> DMatrix<f64> {
    let b = &b.transpose();
    return DMatrix::<f64>::from_fn(d, d, |i, j| 
                                    a.row(i).zip_fold(&b.row(j), 0_f64, |acc, a, b| acc + a * b)
                                    );
}

fn mul_2x2(a: &Matrix2<f64>, b: &Matrix2<f64>) -> Matrix2<f64> {
    let b = &b.transpose();
    return Matrix2::<f64>::from_fn(|i, j| 
                                    a.row(i).zip_fold(&b.row(j), 0_f64, |acc, a, b| acc + a * b)
                                    );
}

fn strassen(a: &Matrix4<f64>, b: &Matrix4<f64>) -> Matrix4<f64> {
    let a_s = as_submatrices(a);
    let b_s = as_submatrices(b);
    //println!("{}", mul_d(&a_s.0, &(b_s.0 - b_s.3), 2));
    /*let subproducts = (mul_d(&a_s.0, &(b_s.0 - b_s.3)),
                       mul_d(&(a_s.0 + a_s.1), b_s.3),
                       mul_d(&(a_s.2 + a_s.3), b_s.0),
                       mul_d(&a_s.3, &(b_s.2 - b_s.0)),
                       mul_d(&(a_s.0 + a_s.3), &(b_s.0 + b_s.3)),
                       mul_d(&(a_s.1 - a_s.3), &(b_s.2 + b_s.3)),
                       mul_d(&(a_s.0 - a_s.2), &(b_s.0 + b_s.1)))*/
    let sub_muls = (mul_d(&a_s.0, &b_s.0, 2) + mul_d(&a_s.1, &b_s.2, 2),
                    mul_d(&a_s.0, &b_s.1, 2) + mul_d(&a_s.1, &b_s.3, 2),
                    mul_d(&a_s.2, &b_s.0, 2) + mul_d(&a_s.3, &b_s.2, 2),
                    mul_d(&a_s.2, &b_s.1, 2) + mul_d(&a_s.3, &b_s.3, 2));
    return Matrix4::<f64>::from_row_iterator(sub_muls.0.row(0).iter()
                                             .chain(sub_muls.1.row(0).iter())
                                             .chain(sub_muls.0.row(1).iter())
                                             .chain(sub_muls.1.row(1).iter())
                                             .chain(sub_muls.2.row(0).iter())
                                             .chain(sub_muls.3.row(0).iter())
                                             .chain(sub_muls.2.row(1).iter())
                                             .chain(sub_muls.3.row(1).iter()).cloned());
}

