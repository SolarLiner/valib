
fn clipper_params(Delta_t: f64, pdist: f64) -> (f64, i32, f64, f64, f64, f64, f64) {

    let out1 = (2500000000000*Delta_t.powi(2) + 245000000000*Delta_t*pdist + 26094500000*Delta_t + 56329500*pdist + 5745609)/(125000000000000*Delta_t.powi(3) + 6375000000000*Delta_t.powi(2)*pdist + 3205475000000*Delta_t.powi(2) + 130316475000*Delta_t*pdist + 14396780450*Delta_t + 56329500*pdist + 5745609);
    let out2 = 1;
    let out3 = (-2500000000000*Delta_t.powi(2) - 500000*Delta_t*(240100000000*pdist.powi(2) + 48892040000*pdist + 2493867361).sqrt() + 56329500*pdist + 5745609)/(2500000000000*Delta_t.powi(2) + 245000000000*Delta_t*pdist + 26094500000*Delta_t + 56329500*pdist + 5745609);
    let out4 = (-2500000000000*Delta_t.powi(2) + 500000*Delta_t*(240100000000*pdist.powi(2) + 48892040000*pdist + 2493867361).sqrt() + 56329500*pdist + 5745609)/(2500000000000*Delta_t.powi(2) + 245000000000*Delta_t*pdist + 26094500000*Delta_t + 56329500*pdist + 5745609);
    let out5 = (1 - 50*Delta_t)/(50*Delta_t + 1);
    let out6 = (-500000*Delta_t + 25500*pdist + 2601)/(500000*Delta_t + 25500*pdist + 2601);
    let out7 = (2209 - 5000000*Delta_t)/(5000000*Delta_t + 2209);
    (out1, out2, out3, out4, out5, out6, out7)

}

fn clipper(K: f64, p1: f64, p2: f64, p3: f64, z1: f64, z2: f64, z3: f64) -> (f64, i32, f64, f64) {

    let out1 = SMatrix::<_, 3, 3>::new(0, 1, 0, 0, 0, 1, p1*p2*p3, -p1*p2 - p1*p3 - p2*p3, p1 + p2 + p3);
    let out2 = SMatrix::<_, 3, 1>::new(0, 0, 1);
    let out3 = SMatrix::<_, 1, 3>::new(K*p1*p2*p3 - K*z1*z2*z3, K*z1*z2 + K*z1*z3 + K*z2*z3 - K*(p1*p2 + p1*p3 + p2*p3), -K*z1 - K*z2 - K*z3 - K*(-p1 - p2 - p3));
    let out4 = SMatrix::<_, 1, 1>::new(K);
    (out1, out2, out3, out4)

}
