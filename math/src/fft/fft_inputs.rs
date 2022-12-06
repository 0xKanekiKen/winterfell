use super::StarkField;
use crate::FieldElement;

pub trait FftInputs<B: StarkField> {
    fn len(&self) -> usize;

    fn butterfly(&mut self, offset: usize, stride: usize);

    fn butterfly_twiddle(&mut self, twiddle: B, offset: usize, stride: usize);

    fn swap(&mut self, i: usize, j: usize);

    fn interpolate(eval: &mut Self, domain_offset: B, offset: &mut B);
}

impl<B, E> FftInputs<B> for [E]
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
{
    fn len(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn butterfly(&mut self, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;
        let temp = self[i];
        self[i] = temp + self[j];
        self[j] = temp - self[j];
    }

    #[inline(always)]
    fn butterfly_twiddle(&mut self, twiddle: B, offset: usize, stride: usize) {
        let i = offset;
        let j = offset + stride;
        let temp = self[i];
        self[j] = self[j].mul_base(twiddle);
        self[i] = temp + self[j];
        self[j] = temp - self[j];
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.swap(i, j)
    }

    fn interpolate(eval: &mut [E], domain_offset: B, offset: &mut B) {
        let domain_offset = B::inv(domain_offset.into());
        for coeff in eval.iter_mut() {
            *coeff *= E::from(*offset);
            *offset *= domain_offset;
        }
    }
}
