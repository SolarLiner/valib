use std::{
    fmt,
    ops::Not
};

use nalgebra::{ComplexField, SMatrix, SVector};
use num_traits::FromPrimitive;
use numeric_literals::replace_float_literals;

use crate::{
    saturators::Saturator,
    math::{newton_rhapson_steps, RootEq},
    Scalar,
    DSP
};

pub struct DiodeClipper<T> {
    pub isat: T,
    pub n: T,
    pub vt: T,
    pub vin: T,
    pub num_diodes_fwd: T,
    pub num_diodes_bwd: T,
    pub sim_tol: T,
    pub max_iter: usize,
}

impl<T: Scalar> RootEq<T, 1> for DiodeClipper<T> {
    #[inline]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn eval(&self, input: &nalgebra::SVector<T, 1>) -> nalgebra::SVector<T, 1> {
        let vout = input[0];
        let v = T::recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::exp(expin / self.num_diodes_fwd);
        let expm = T::exp(-expin / self.num_diodes_bwd);
        let res = self.isat * (expn - expm) + 2. * vout - self.vin;
        SVector::<_, 1>::new(res)
    }

    #[inline]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn j_inv(&self, input: &nalgebra::SVector<T, 1>) -> Option<SMatrix<T, 1, 1>> {
        let vout = input[0];
        let v = T::recip(self.n * self.vt);
        let expin = vout * v;
        let expn = T::exp(expin / self.num_diodes_fwd);
        let expm = T::exp(-expin / self.num_diodes_bwd);
        let res = v * self.isat * (expn / self.num_diodes_fwd + expm / self.num_diodes_bwd) + 2.;
        res.is_zero()
            .not()
            .then_some(SMatrix::<_, 1, 1>::new(res.recip()))
    }
}

impl<T: FromPrimitive> DiodeClipper<T> {
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_silicon(fwd: usize, bwd: usize, vin: T) -> Self {
        Self {
            isat: 4.352e-9,
            n: 1.906,
            vt: 23e-3,
            num_diodes_fwd: T::from_usize(fwd).unwrap(),
            num_diodes_bwd: T::from_usize(bwd).unwrap(),
            vin,
            sim_tol: 1e-3,
            max_iter: 50,
        }
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_germanium(fwd: usize, bwd: usize, vin: T) -> Self {
        Self {
            isat: 200e-9,
            n: 2.109,
            vt: 23e-3,
            num_diodes_fwd: T::from_usize(fwd).unwrap(),
            num_diodes_bwd: T::from_usize(bwd).unwrap(),
            vin,
            sim_tol: 1e-3,
            max_iter: 50,
        }
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_led(nf: usize, nb: usize, vin: T) -> DiodeClipper<T> {
        Self {
            isat: 2.96406e-12,
            n: 2.475312,
            vt: 23e-3,
            vin,
            num_diodes_fwd: T::from_usize(nf).unwrap(),
            num_diodes_bwd: T::from_usize(nb).unwrap(),
            sim_tol: 1e-3,
            max_iter: 50,
        }
    }
}

impl<T: Scalar + ComplexField + fmt::Debug> DSP<1, 1> for DiodeClipper<T> {
    type Sample = T;

    fn process(&mut self, x: [Self::Sample; 1]) -> [Self::Sample; 1] {
        self.vin = x[0];
        let mut vout = SVector::<_, 1>::new(<T as ComplexField>::tanh(x[0]));
        newton_rhapson_steps(self, &mut vout, 4);
        [vout[0]]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DiodeClipperModel<T> {
    pub a: T,
    pub b: T,
    pub si: T,
    pub so: T,
}

// These seemingly magic constants have been fit against the diode clipper circuit equation for
// combinations of up to 5 diodes in series each way.
//
// See the `clippers.ipynb` Notebook to see the rationale and working out process.
impl<T: FromPrimitive> DiodeClipperModel<T> {
    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_silicon(nf: u8, nb: u8) -> Self {
        assert!(
            (1..=5).contains(&nf) && (1..=5).contains(&nb),
            "# diodes in clipper must be within 1..=5"
        );
        let [a, b, si, so] = match (nb, nb) {
            (1, 1) => [
                14.013538783148167,
                14.013538783002625,
                9.838916458010646,
                0.05074333630270928,
            ],
            (1, 2) => [
                21.871748015161707,
                9.182532240914016,
                6.890252004666455,
                0.07133160117027257,
            ],
            (1, 3) => [
                27.693280114911275,
                7.050762765047438,
                5.608233617849939,
                0.08718106563980055,
            ],
            (1, 4) => [
                33.05651848106969,
                5.951968392121445,
                5.023914218206766,
                0.09840304732958556,
            ],
            (1, 5) => [
                51.12409002822087,
                7.8003913971037075,
                6.122022474813163,
                0.08074110462333906,
            ],
            (2, 1) => [
                9.182531655316685,
                21.87174680941982,
                6.890251666047109,
                0.07133160469274555,
            ],
            (2, 2) => [
                13.89883538213846,
                13.898835382148588,
                4.626654245793477,
                0.10628426546665204,
            ],
            (2, 3) => [
                15.688724237580493,
                9.534083123281304,
                3.4044199978910887,
                0.14584970841530884,
            ],
            (2, 4) => [
                20.207387197366668,
                8.79257073406701,
                3.179217655773826,
                0.15593734904268552,
            ],
            (2, 5) => [
                28.031927112133676,
                9.718934071656136,
                3.442900410418956,
                0.1436485121303713,
            ],
            (3, 1) => [
                7.050764727600349,
                27.693286281387294,
                5.608234736362655,
                0.08718104778576853,
            ],
            (3, 2) => [
                9.534082343136829,
                15.688723052859338,
                3.4044197732928616,
                0.14584971824082685,
            ],
            (3, 3) => [
                11.587310022245902,
                11.58731002231186,
                2.614009632051883,
                0.19058071488150277,
            ],
            (3, 4) => [
                14.13075319100304,
                10.017533709691163,
                2.296602223426621,
                0.21637254869143377,
            ],
            (3, 5) => [
                17.03033061419644,
                9.372243751551357,
                2.1657426584828094,
                0.2291822087004832,
            ],
            (4, 1) => [
                5.951966742158502,
                33.05651159454797,
                5.023913234557704,
                0.09840306669144883,
            ],
            (4, 2) => [
                8.792569843968032,
                20.20738538786123,
                3.1792173949079627,
                0.15593736194479207,
            ],
            (4, 3) => [
                10.017533796475519,
                14.130753307308828,
                2.296602240295518,
                0.2163725470818328,
            ],
            (4, 4) => [
                11.041181755308786,
                11.041181755348719,
                1.838119504765217,
                0.27022723677304133,
            ],
            (4, 5) => [
                11.223240551951672,
                8.609635416761739,
                1.4800838613669387,
                0.33640845725537083,
            ],
            (5, 1) => [
                7.800388576323131,
                51.12407559566858,
                6.122020811641261,
                0.08074112661301132,
            ],
            (5, 2) => [
                9.718935117732514,
                28.031929752363737,
                3.4429007169046835,
                0.1436484993003876,
            ],
            (5, 3) => [
                9.372245058124795,
                17.030332812445042,
                2.1657429124687315,
                0.22918218140592617,
            ],
            (5, 4) => [
                8.609635267942823,
                11.22324036552669,
                1.4800838395934892,
                0.33640846230428967,
            ],
            (5, 5) => [
                8.339132991537618,
                8.339132991523964,
                1.1325230220918208,
                0.440338545923957,
            ],
            _ => unreachable!(),
        };
        Self { a, b, si, so }
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_germanium(nf: u8, nb: u8) -> Self {
        assert!(
            (1..=5).contains(&nf) && (1..=5).contains(&nb),
            "# diodes in clipper must be within 1..=5"
        );
        let [a, b, si, so] = match (nf, nb) {
            (1, 1) => [
                16.377243363175054,
                16.37724336318019,
                8.54021415704938,
                0.057370187480517164,
            ],
            (1, 2) => [
                25.885042335041874,
                11.253874553827654,
                6.281682965857368,
                0.07844744876947145,
            ],
            (1, 3) => [
                34.487810131698026,
                9.412993465541485,
                5.468058945021779,
                0.090540745529638,
            ],
            (1, 4) => [
                48.10205558502969,
                9.841653924324676,
                5.683172785917438,
                0.08734936732972111,
            ],
            (1, 5) => [
                74.91010555238323,
                13.006634342954875,
                7.156335421839414,
                0.0695469593703475,
            ],
            (2, 1) => [
                11.253874507514725,
                25.8850422416139,
                6.281682945580695,
                0.07844744902868878,
            ],
            (2, 2) => [
                14.579485959692967,
                14.579485959584705,
                3.7836434149533478,
                0.131186663201383,
            ],
            (2, 3) => [
                19.505471519263676,
                12.1251424262162,
                3.218977637096419,
                0.1540919429113331,
            ],
            (2, 4) => [
                25.390598549056296,
                11.515340893545972,
                3.084987878362374,
                0.16106433598674907,
            ],
            (2, 5) => [
                35.57696932591939,
                13.102996075964564,
                3.4572621702410475,
                0.14396799707333885,
            ],
            (3, 1) => [
                9.412993807209658,
                34.48781118871304,
                5.468059101273827,
                0.09054074291711343,
            ],
            (3, 2) => [
                12.125142754843933,
                19.50547201767495,
                3.218977712195031,
                0.15409193929365167,
            ],
            (3, 3) => [
                12.580061365916833,
                12.58006136574436,
                2.172533025937179,
                0.22922482434392638,
            ],
            (3, 4) => [
                15.610044920725795,
                11.19900028184279,
                1.9557246559358357,
                0.25438530992348235,
            ],
            (3, 5) => [
                20.766038390005722,
                11.828743968114324,
                2.055494391522101,
                0.24222422759566611,
            ],
            (4, 1) => [
                9.841650386184702,
                48.10204124271992,
                5.6831711547599735,
                0.08734939245309749,
            ],
            (4, 2) => [
                11.515339727304719,
                25.39059620900811,
                3.084987610491443,
                0.16106435003667227,
            ],
            (4, 3) => [
                11.199002268491,
                15.610047567107346,
                1.955724963317157,
                0.25438526976983744,
            ],
            (4, 4) => [
                9.789367161124993,
                9.78936716107417,
                1.2783124100453043,
                0.39031047870114366,
            ],
            (4, 5) => [
                11.199637242408283,
                8.688306318271986,
                1.1466396427036314,
                0.43504915836012503,
            ],
            (5, 1) => [
                13.006628700955355,
                74.9100780229614,
                7.156332842362052,
                0.06954698445157906,
            ],
            (5, 2) => [
                13.102996274684337,
                35.576969804103044,
                3.4572622156825514,
                0.14396799518071915,
            ],
            (5, 3) => [
                11.828742623775012,
                20.76603619590081,
                2.0554941835424905,
                0.24222425216829363,
            ],
            (5, 4) => [
                8.688308178030308,
                11.199639530649264,
                1.1466398647883216,
                0.43504907393569836,
            ],
            (5, 5) => [
                5.8988195665816905,
                5.898819566582509,
                0.6308047004920511,
                0.7918563457384652,
            ],
            _ => unreachable!(),
        };
        Self { a, b, si, so }
    }

    #[replace_float_literals(T::from_f64(literal).unwrap())]
    pub fn new_led(nf: u8, nb: u8) -> Self {
        assert!(
            (1..=5).contains(&nf) && (1..=5).contains(&nb),
            "# diodes in clipper must be within 1..=5"
        );
        let [a, b, si, so] = match (nf, nb) {
            (1, 1) => [
                4.435_713_979_386_322e-5,
                4.435_638_644_124_075e-5,
                0.3001402495706703,
                1.676015028548096,
            ],
            (1, 2) => [
                1.5753358037082148,
                0.3863703462043009,
                0.49878617525719776,
                1.0639637950068614,
            ],
            (1, 3) => [
                2.1546856628863655,
                0.06928481836415978,
                0.3329889894271551,
                1.5186174762828244,
            ],
            (1, 4) => [
                3.0281049178820543,
                0.0032464741823931206,
                0.2970523601172462,
                1.6829046548375952,
            ],
            (1, 5) => [
                3.5144367124396108,
                0.002915891412681926,
                0.2970311389293134,
                1.683306819551628,
            ],
            (2, 1) => [
                0.3863705958666232,
                1.5753362340604948,
                0.498786238329306,
                1.0639636273268471,
            ],
            (2, 2) => [
                16.424299661398564,
                16.42429966161495,
                3.010532359651235,
                0.1658253668791719,
            ],
            (2, 3) => [
                21.52060017050886,
                13.640116199834821,
                2.543487442965256,
                0.19610839508452663,
            ],
            (2, 4) => [
                33.96161938090595,
                16.451011829388115,
                3.013879924129542,
                0.16559017534935822,
            ],
            (2, 5) => [
                35.83028468657796,
                16.305708297311323,
                2.995703016266972,
                0.1668774538192173,
            ],
            (3, 1) => [
                0.0692847926221858,
                2.154685589394149,
                0.3329889807641022,
                1.5186175172102436,
            ],
            (3, 2) => [
                13.640114653384448,
                21.52059785652759,
                2.5434871856274714,
                0.19610841501030898,
            ],
            (3, 3) => [
                12.912141619267407,
                12.912141619548203,
                1.5764737878319843,
                0.31661052277964713,
            ],
            (3, 4) => [
                17.50010626618652,
                12.926694115470912,
                1.5778897299024395,
                0.3162920770711803,
            ],
            (3, 5) => [
                18.774139848715233,
                12.74654340777339,
                1.560378164399095,
                0.32027925239938787,
            ],
            (4, 1) => [
                0.0032464910429759364,
                3.028104995079552,
                0.29705236725575596,
                1.6829046145020117,
            ],
            (4, 2) => [
                16.45101245250614,
                33.96162057167496,
                3.013880027502115,
                0.16559016966278778,
            ],
            (4, 3) => [
                12.92669460738176,
                17.500106902312837,
                1.5778897855307121,
                0.31629206590798103,
            ],
            (4, 4) => [
                3.536026014379721,
                3.53602601437973,
                0.343521712041019,
                1.4550933490658728,
            ],
            (4, 5) => [
                4.0413270488286415,
                3.499058879138013,
                0.3403145160759295,
                1.469030821133474,
            ],
            (5, 1) => [
                0.0029159084611507,
                3.5144367966222614,
                0.2970311448110932,
                1.6833067859022843,
            ],
            (5, 2) => [
                16.30570866638424,
                35.830285428404906,
                2.995703077696013,
                0.1668774503943544,
            ],
            (5, 3) => [
                12.746338329375735,
                22.09517446273478,
                1.5603548987543252,
                0.3202840316289733,
            ],
            (5, 4) => [
                3.4990583028197717,
                4.0413264147144705,
                0.34031446336973104,
                1.469031048722433,
            ],
            (5, 5) => [
                1.8259169483116064,
                1.8259169483116053,
                0.15511805295671305,
                3.2232909023681384,
            ],
            _ => unreachable!(),
        };
        Self { a, b, si, so }
    }
}

impl<T: FromPrimitive> Default for DiodeClipperModel<T> {
    fn default() -> Self {
        Self::new_silicon(1, 1)
    }
}

impl<T: Scalar + FromPrimitive> DSP<1, 1> for DiodeClipperModel<T> {
    type Sample = T;

    #[inline(always)]
    fn process(&mut self, [x]: [Self::Sample; 1]) -> [Self::Sample; 1] {
        [self.saturate(x * self.so) / self.si]
    }
}

impl<T: Scalar + FromPrimitive> Saturator<T> for DiodeClipperModel<T> {
    #[inline]
    #[replace_float_literals(T::from(literal).unwrap())]
    fn saturate(&self, x: T) -> T {
        let x = self.si / self.so * x;
        let out = if x < -self.a {
            -T::ln(1. - x - self.a) - self.a
        } else if x > self.b {
            T::ln(1. + x - self.b) + self.b
        } else {
            x
        };
        out * self.so / self.si
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use nalgebra::SVector;

    use crate::math::newton_rhapson_tolerance;

    use super::DiodeClipper;

    #[test]
    fn evaluate_diode_clipper() {
        let mut clipper = DiodeClipper::new_led(3, 5, 0.);
        let mut file = File::create("clipper.tsv").unwrap();
        writeln!(file, "\"in\"\t\"out\"\t\"iter\"").unwrap();
        for i in -4800..4800 {
            clipper.vin = i as f64 / 100.;
            let mut vout = SVector::<_, 1>::new(f64::tanh(clipper.vin));
            let iter = newton_rhapson_tolerance(&clipper, &mut vout, 1e-3);
            writeln!(file, "{}\t{}\t{}", clipper.vin, vout[0], iter).unwrap();
        }
    }
}
