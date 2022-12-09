use super::StarkField;
use crate::FieldElement;

// FFTINPUTS TRAIT
// ================================================================================================

/// Defines the shape of the input passed as input for FFT computation.
pub trait FftInputs<B: StarkField> {
    /// Returns the number of elements in this input.
    fn len(&self) -> usize;

    /// Combines the result of smaller discrete fourier transforms into a larger DFT.
    fn butterfly(&mut self, offset: usize, stride: usize);

    /// Combines the result of smaller discrete fourier transforms multiplied with a
    /// twiddle factor into a larger DFT.
    fn butterfly_twiddle(&mut self, twiddle: B, offset: usize, stride: usize);

    /// Swaps the element at index i with the element at index j.
    fn swap(&mut self, i: usize, j: usize);

    /// Multiplies every element in this input by the product of `init_offset` with
    /// `offset_factor` raise to power i, where i is the index at which the element
    /// is present in fftindex. Specifically:
    ///
    /// elem_{i} = elem_{i} * init_offset * offset_factor^i
    fn shift_by_series(&mut self, offset: B, increment: B);

    /// Multiplies every element in this input by `offset`. Specifically:
    ///
    /// elem_{i} = elem_{i} * offset
    fn shift_by(&mut self, offset: B);
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

    fn shift_by_series(&mut self, offset: B, increment: B) {
        let mut offset = offset;
        for d in self.iter_mut() {
            *d *= E::from(offset);
            offset *= increment;
        }
    }

    fn shift_by(&mut self, offset: B) {
        for d in self.iter_mut() {
            *d *= E::from(offset);
        }
    }
}
