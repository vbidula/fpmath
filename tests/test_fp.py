import logging

import numpy as np
import pytest

from fpmath import bit_reverse, div_pow2, rightshift, FixedPoint, FixedPointFormatWarning


class TestFixedPoint:

    def test_div_pow2(self):
        ref = np.array([0, 1, 2, 4, 8, -1, -2, -3, -4, -8], dtype=np.int64)
        np.testing.assert_equal(
            div_pow2(ref, 1),
            np.array([0, 0, 1, 2, 4, 0, -1, -1, -2, -4], dtype=np.int64)
        )

    def test_rightshift_deprecated_alias(self):
        ref = np.array([3, -3], dtype=np.int64)
        with pytest.warns(DeprecationWarning):
            res = rightshift(ref, 1)
        np.testing.assert_equal(res, div_pow2(ref, 1))

    def test_bit_reverse(self):
        ref = np.arange(10)
        np.testing.assert_equal(bit_reverse(ref, 3), np.array([0, 4, 2, 6, 1, 5, 3, 7, 0, 4]))

    def test_init(self):
        data = np.arange(10)
        fp = FixedPoint(8, 4, data, ofmode='wrap')
        assert fp.bw == 8
        assert fp.fw == 4
        reference = np.array([0, 1, 2, 3, 4, 5, 6, 7, -8, -7], dtype=np.float64)
        np.testing.assert_array_equal(fp.data, reference)

        data2 = np.array([1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.,
                          3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.25,
                          5.5, 5.75, 6., 6.25, 6.5, 6.75, 7., 7.25, 7.5,
                          7.75, 8.])
        reference2 = np.array([1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.,
                               3.25, 3.5, 3.75, 4., 4.25, 4.5, 4.75, 5., 5.25,
                               5.5, 5.75, 6., 6.25, 6.5, 6.75, 7., 7.25, 7.5,
                               7.75, -8.])
        fp2 = FixedPoint(8, 4, data2, ofmode='wrap')
        np.testing.assert_array_equal(fp2.data, reference2)

    def test_mask_bitwidth(self):
        bdata = np.array([0, 1, 3, 7, 15, 31, 63, 127], dtype=np.int32)
        fp = FixedPoint(8, 7, bdata, disbin=True, ofmode='wrap')
        np.testing.assert_equal(fp.bdata, bdata)

        fp2 = FixedPoint(8, 7, bdata << 1, disbin=True, ofmode='wrap')
        np.testing.assert_equal(fp2.data, np.array([0, 2, 6, 14, 30, 62, 126, -2], dtype=np.int64) / 2 ** 7)

    def test_cast(self):
        data = np.arange(10)
        fp = FixedPoint(8, 3, data, ofmode='wrap')
        fp_cast = fp.cast(10, 5)
        assert fp_cast.bw == 10
        assert fp_cast.fw == 5
        np.testing.assert_array_equal(fp_cast.data, fp.data)

    def test_add(self):
        data = np.arange(10) / 4
        fp = FixedPoint(8, 3, data, ofmode='wrap')
        fp2 = FixedPoint(10, 5, data, ofmode='wrap')
        with pytest.raises(ValueError):
            fp + fp2

        add_fp = fp.cast(10, 5) + fp2.cast(10, 5)
        assert add_fp.bw == 11
        assert add_fp.fw == 5
        np.testing.assert_array_equal(add_fp.data, data * 2)

    def test_addsub_fw_mismatch_warns(self):
        # Same bw, different fw: operates on raw integers like hardware, but warns about the missing cast
        fp = FixedPoint(8, 2, np.array([1.0]))
        fp2 = FixedPoint(8, 4, np.array([1.0]))
        with pytest.warns(FixedPointFormatWarning):
            add_fp = fp + fp2
        np.testing.assert_array_equal(add_fp.bdata, fp.bdata + fp2.bdata)
        with pytest.warns(FixedPointFormatWarning):
            sub_fp = fp - fp2
        np.testing.assert_array_equal(sub_fp.bdata, fp.bdata - fp2.bdata)

    def test_no_logging_handlers_added(self):
        fp = FixedPoint(8, 0, np.arange(4))
        handlers_before = {name: len(lg.handlers) for name, lg in logging.root.manager.loggerDict.items()
                           if isinstance(lg, logging.Logger)}
        for _ in range(100):
            fp = (fp + fp).truncate_msb(8)
        handlers_after = {name: len(lg.handlers) for name, lg in logging.root.manager.loggerDict.items()
                          if isinstance(lg, logging.Logger)}
        assert handlers_after == handlers_before

    def test_twos_complement(self):
        data = np.arange(16)
        fp = FixedPoint(4, 0, data, ofmode='wrap')
        neg_fp = -fp
        ref_data = np.array([0., -1., -2., -3., -4., -5., -6., -7., -8., 7., 6., 5., 4.,
                             3., 2., 1.], dtype=np.float64)
        np.testing.assert_equal(neg_fp.data, ref_data)

    def test_sub(self):
        data = np.arange(10)
        fp = FixedPoint(8, 3, data / 2)
        fp2 = FixedPoint(10, 5, data / 4)
        sub_fp = fp.cast(10, 5) - fp2.cast(10, 5)
        assert sub_fp.bw == 11
        assert sub_fp.fw == 5
        np.testing.assert_array_equal(sub_fp.data, data / 4)

    def test_sub_exhaustive(self):
        # Every 4-bit pair, including the most negative subtrahend: 0 - (-8) must be +8 in the 5-bit result
        vals = np.arange(-8, 8)
        a, b = np.meshgrid(vals, vals)
        fp_a = FixedPoint(4, 0, a.ravel(), disbin=True)
        fp_b = FixedPoint(4, 0, b.ravel(), disbin=True)
        sub_fp = fp_a - fp_b
        assert sub_fp.bwfw == (5, 0)
        np.testing.assert_array_equal(sub_fp.bdata, a.ravel() - b.ravel())

    def test_add_exhaustive(self):
        vals = np.arange(-8, 8)
        a, b = np.meshgrid(vals, vals)
        add_fp = FixedPoint(4, 0, a.ravel(), disbin=True) + FixedPoint(4, 0, b.ravel(), disbin=True)
        assert add_fp.bwfw == (5, 0)
        np.testing.assert_array_equal(add_fp.bdata, a.ravel() + b.ravel())

    def test_mul(self):
        data = np.arange(10)
        fp1 = FixedPoint(9, 4, data)
        fp2 = FixedPoint(6, 1, data)

        mult_fp12 = fp1 * fp2
        assert mult_fp12.bw == 15
        assert mult_fp12.fw == 5
        np.testing.assert_array_equal(mult_fp12.data, data * data)

        fp3 = FixedPoint(9, 6, data / 4)
        fp4 = FixedPoint(6, 4, data / 8)
        mult_fp34 = fp3 * fp4
        np.testing.assert_array_equal(mult_fp34.data, data * data / 32)

    def test_bw_limit(self):
        # 32 x 32 -> 64 bits fits int64 exactly, including the extremes
        extremes = np.array([-2**31, 2**31 - 1])
        fp = FixedPoint(32, 0, extremes, disbin=True)
        mult = fp * fp
        assert mult.bw == 64
        np.testing.assert_array_equal(mult.bdata, extremes * extremes)

        with pytest.raises(ValueError):
            FixedPoint(65, 0, np.array([0]))
        with pytest.raises(ValueError):
            FixedPoint(40, 0, np.array([1])) * FixedPoint(40, 0, np.array([1]))
        with pytest.raises(ValueError):
            fp.cast(65, 0)
        with pytest.raises(ValueError):
            mult + mult

    def test_truncate(self):
        data = np.arange(10) / 10
        fp = FixedPoint(8, 7, data)
        fp_ref = FixedPoint(6, 5, fp.bdata >> 2, disbin=True)
        fp_trunc = fp.truncate(6)
        assert fp_trunc.bw == 6
        assert fp_trunc.fw == 5
        np.testing.assert_array_equal(fp_trunc.data, fp_ref.data)

    def test_round_from_zero(self):
        data = np.array([3.5, 3.75, 4.25, 4.5, -3.0, -3.25, -3.5, -3.75, -4.5])
        fp = FixedPoint(6, 2, data)
        fp_round = fp.round_from_zero(4)
        reference = np.array([4, 4, 4, 5, -3, -3, -4, -4, -5])
        np.testing.assert_equal(fp_round.data, reference)

    def test_round_to_even(self):
        data = np.array([3.5, 3.75, 4.25, 4.5, -3.0, -3.25, -3.5, -3.75, -4.5])
        fp = FixedPoint(6, 2, data)
        fp_round = fp.round_to_even(4)
        reference = np.array([4, 4, 4, 4, -3, -3, -4, -4, -4])
        np.testing.assert_equal(fp_round.data, reference)

    def test_to_float(self):
        data = np.arange(10, dtype=np.int32)
        fp1 = FixedPoint(8, 0, data, disbin=True, ofmode='wrap')
        fp2 = FixedPoint(8, 2, data, disbin=True, ofmode='wrap')
        fp3 = FixedPoint(3, 2, data, disbin=True, ofmode='wrap')
        fp4 = FixedPoint(8, 7, data, disbin=True, ofmode='wrap')
        np.testing.assert_equal(fp1.data, data.astype(np.float64))
        np.testing.assert_equal(fp2.data, data.astype(np.float64) / 4)
        np.testing.assert_equal(fp3.data, np.array([0, 0.25, 0.5, 0.75, -1, -0.75, -0.5, -0.25, 0, 0.25]))
        np.testing.assert_equal(fp4.data, data.astype(np.float64) / 128)

    def test_from_float(self):
        data = np.array([0.51, 1, 2, 0.75, 0.1])
        fp1 = FixedPoint(8, 0, data)
        fp2 = FixedPoint(8, 2, data)
        np.testing.assert_equal(fp1.bdata, np.array([1, 1, 2, 1, 0], dtype=np.int32))
        np.testing.assert_equal(fp2.bdata, np.array([2, 4, 8, 3, 0], dtype=np.int32))
        data2 = np.array([-0.0661468505859375])
        fp3 = FixedPoint(37, 34, data2)
        np.testing.assert_equal(fp3.bdata, -1136394240)

    def test_binstr(self):
        data = np.array([3])
        fp = FixedPoint(4, 1, data)
        assert fp.binary[0] == "011.0"
        fp = FixedPoint(4, 0, data)
        assert fp.binary[0] == "0011."

    def test_copy(self):
        data = np.array([3])
        fp = FixedPoint(4, 0, data)
        fp2 = fp.copy()
        fp.bdata[0] = 4
        np.testing.assert_equal(fp2.bdata[0], 3)

    def test_rshift(self):
        # Arithmetic shift: rounds toward -inf like numeric_std shift_right() on signed
        data = np.array([3, -3, -1, -4])
        fp = FixedPoint(4, 0, data)
        res = fp >> 1
        np.testing.assert_equal(res.bdata, np.array([1, -2, -1, -2]))
        assert res.bwfw == fp.bwfw

    def test_rshift_matches_truncate(self):
        # Dropping LSBs by slicing must give the same integers as shifting
        fp = FixedPoint(8, 0, np.arange(-128, 128))
        for shift in range(1, 8):
            np.testing.assert_equal((fp >> shift).bdata, fp.truncate(8 - shift).bdata)

    def test_div_pow2_method(self):
        # Rounds toward zero like numeric_std "/" on signed
        fp = FixedPoint(4, 0, np.array([3, -3, -1, -4]))
        res = fp.div_pow2(1)
        np.testing.assert_equal(res.bdata, np.array([1, -1, 0, -2]))
        assert res.bwfw == fp.bwfw

    def test_lshift(self):
        data = np.array([1, -1])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((fp << 1).bdata, np.array([2, -2]))

    def test_and(self):
        data = np.array([3, -3])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((fp & 2).bdata, np.array([2, 0]))

    def test_or(self):
        data = np.array([3, -3])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((fp | 2).bdata, np.array([3, -1]))

    def test_xor(self):
        data = np.array([3, -3])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((fp ^ 2).bdata, np.array([1, -1]))

    def test_invert(self):
        data = np.array([3, -3])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((~fp).bdata, np.array([-4, 2]))

    def test_sum(self):
        data = np.arange(10)
        fp = FixedPoint(8, 3, data)
        fp_sum = fp.sum(axis=0)
        np.testing.assert_equal(fp_sum.data, data.sum())
        assert fp_sum.bwfw == (12, 3)

    def test_saturate(self):
        fp = FixedPoint(18, 17, np.array([0.999999966165970, 2, -100]), ofmode="saturate")
        np.testing.assert_equal(fp.bdata, np.array([131071, 131071, -131072]))

    # --- ofmode: narrowing operations wrap (default) or saturate; widening and raw-bit loading never clamp ---

    @staticmethod
    def _wrap(values, bw):
        return ((values + (1 << (bw - 1))) % (1 << bw)) - (1 << (bw - 1))

    @staticmethod
    def _clip(values, bw):
        return np.clip(values, -(1 << (bw - 1)), (1 << (bw - 1)) - 1)

    def test_default_ofmode_is_wrap(self):
        assert FixedPoint(4, 0, np.array([0])).ofmode == "wrap"
        np.testing.assert_equal(FixedPoint(4, 0, np.array([9.0])).bdata, np.array([-7]))

    def test_truncate_msb_ofmode(self):
        vals = np.arange(-128, 128)
        wrapped = FixedPoint(8, 0, vals, disbin=True, ofmode="wrap").truncate_msb(4)
        saturated = FixedPoint(8, 0, vals, disbin=True, ofmode="saturate").truncate_msb(4)
        np.testing.assert_equal(wrapped.bdata, self._wrap(vals, 4))
        np.testing.assert_equal(saturated.bdata, self._clip(vals, 4))
        assert saturated.bwfw == (4, 0)
        assert saturated.ofmode == "saturate"

    def test_neg_ofmode(self):
        vals = np.arange(-8, 8)
        np.testing.assert_equal((-FixedPoint(4, 0, vals, disbin=True, ofmode="wrap")).bdata, self._wrap(-vals, 4))
        np.testing.assert_equal((-FixedPoint(4, 0, vals, disbin=True, ofmode="saturate")).bdata, self._clip(-vals, 4))

    def test_lshift_ofmode(self):
        vals = np.arange(-8, 8)
        for shift in range(1, 4):
            np.testing.assert_equal((FixedPoint(4, 0, vals, disbin=True, ofmode="wrap") << shift).bdata,
                                    self._wrap(vals << shift, 4))
            np.testing.assert_equal((FixedPoint(4, 0, vals, disbin=True, ofmode="saturate") << shift).bdata,
                                    self._clip(vals << shift, 4))
        with pytest.raises(ValueError):
            FixedPoint(60, 0, np.array([1])) << 5

    @pytest.mark.parametrize("method", ["round_from_zero", "round_to_even"])
    def test_round_overflow_ofmode(self, method):
        # 7.75 in 6 bits (fw=2) rounds up to 8, which does not fit into 4 bits
        data = np.array([7.75, -8.0])
        wrapped = getattr(FixedPoint(6, 2, data, ofmode="wrap"), method)(4)
        saturated = getattr(FixedPoint(6, 2, data, ofmode="saturate"), method)(4)
        np.testing.assert_equal(wrapped.bdata, np.array([-8, -8]))
        np.testing.assert_equal(saturated.bdata, np.array([7, -8]))

    def test_disbin_never_saturates(self):
        # Raw DUT patterns are reinterpreted, never clamped: 16383 in 14 bits is -1
        fp = FixedPoint(14, 0, np.array([16383, 8192, 100]), disbin=True, ofmode="saturate")
        np.testing.assert_equal(fp.bdata, np.array([-1, -8192, 100]))
        np.testing.assert_equal(fp.copy().bdata, fp.bdata)

    def test_widening_ignores_ofmode(self):
        vals = np.arange(-8, 8)
        a, b = np.meshgrid(vals, vals)
        fa = FixedPoint(4, 0, a.ravel(), disbin=True, ofmode="saturate")
        fb = FixedPoint(4, 0, b.ravel(), disbin=True, ofmode="saturate")
        np.testing.assert_equal((fa + fb).bdata, a.ravel() + b.ravel())
        np.testing.assert_equal((fa - fb).bdata, a.ravel() - b.ravel())
        np.testing.assert_equal((fa * fb).bdata, a.ravel() * b.ravel())

    # --- equality ---

    def test_eq(self):
        a = FixedPoint(8, 2, np.array([1.5, -2.25]))
        assert a == FixedPoint(8, 2, np.array([1.5, -2.25]))
        assert a == FixedPoint(8, 2, np.array([1.5, -2.25]), dtype=np.int32, ofmode="saturate")
        assert a != FixedPoint(8, 2, np.array([1.5, -2.0]))
        assert a != FixedPoint(9, 2, np.array([1.5, -2.25]))      # same value, different format
        assert a != FixedPoint(8, 2, np.array([1.5]))             # different shape
        assert a != a.bdata                                        # not a FixedPoint
        with pytest.raises(TypeError):
            hash(a)
