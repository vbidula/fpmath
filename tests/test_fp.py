import numpy as np

from fpmath import bit_reverse, rightshift, FixedPoint


class TestFixedPoint:

    def test_rightshift(self):
        ref = np.array([0, 1, 2, 4, 8, -1, -2, -4, -8], dtype=np.int64)
        np.testing.assert_equal(
            rightshift(ref, 1),
            np.array([0, 0, 1, 2, 4, 0, -1, -2, -4], dtype=np.int64)
        )

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
        try:
            fp + fp2
        except ValueError:
            pass

        add_fp = fp.cast(10, 5) + fp2.cast(10, 5)
        assert add_fp.bw == 11
        assert add_fp.fw == 5
        np.testing.assert_array_equal(add_fp.data, data * 2)

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
        data = np.array([3, -3])
        fp = FixedPoint(4, 0, data)
        np.testing.assert_equal((fp >> 1).bdata, np.array([1, -1]))

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
