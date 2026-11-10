import numpy as np
import pytest

from fpmath import FixedPoint, UFixedPoint


def _wrap_u(values, bw):
    return np.asarray(values) % (1 << bw)


def _clip_u(values, bw):
    return np.clip(values, 0, (1 << bw) - 1)


class TestUFixedPoint:

    # --- construction ---

    def test_disbin_reinterprets_modulo(self):
        fp = UFixedPoint(8, 0, np.array([0, 5, 255, -1, 256, -256]), disbin=True)
        np.testing.assert_equal(fp.bdata, np.array([0, 5, 255, 255, 0, 0]))
        # DUT outputs arrive as unsigned integers and are kept as they are, regardless of ofmode
        dut = UFixedPoint(14, 0, np.array([16383, 8192]), disbin=True, ofmode="saturate")
        np.testing.assert_equal(dut.bdata, np.array([16383, 8192]))

    def test_from_float(self):
        fp = UFixedPoint(8, 2, np.array([0.0, 1.25, 3.3, 63.75]))
        np.testing.assert_equal(fp.bdata, np.array([0, 5, 13, 255]))
        np.testing.assert_equal(fp.data, np.array([0.0, 1.25, 3.25, 63.75]))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_from_float_out_of_range(self, dtype):
        big = 2.0 ** 60 + 3072  # exactly representable in float64
        data = np.array([70000.0, -1.0, big, -big, np.inf, -np.inf])
        sat = UFixedPoint(16, 0, data, dtype=dtype, ofmode="saturate")
        np.testing.assert_equal(sat.bdata, np.array([65535, 0, 65535, 0, 65535, 0]))
        wrap = UFixedPoint(16, 0, data[:4], dtype=dtype)
        np.testing.assert_equal(wrap.bdata, np.array([int(v) % 2 ** 16 for v in data[:4]]))

    def test_from_float_63_bit_extremes(self):
        data = np.array([2.0 ** 63, -1.0, 1e30, 2.0 ** 62])
        sat = UFixedPoint(63, 0, data, ofmode="saturate")
        np.testing.assert_equal(sat.bdata, np.array([2 ** 63 - 1, 0, 2 ** 63 - 1, 2 ** 62]))
        wrap = UFixedPoint(63, 0, data[[0, 1, 3]])
        np.testing.assert_equal(wrap.bdata, np.array([0, 2 ** 63 - 1, 2 ** 62]))

    def test_bw_limits(self):
        # Stored non-negative in a signed dtype: one bit less than the dtype width
        with pytest.raises(ValueError):
            UFixedPoint(64, 0, np.array([0]))
        with pytest.raises(ValueError):
            UFixedPoint(32, 0, np.array([0]), dtype=np.int32)
        fp = UFixedPoint(31, 0, np.array([2 ** 31 - 1]), disbin=True, dtype=np.int32)
        assert fp.bdata[0] == 2 ** 31 - 1
        with pytest.raises(ValueError):
            UFixedPoint(60, 0, np.array([1])) << 4

    @pytest.mark.parametrize("dtype", [np.uint32, np.float64])
    def test_dtype_must_be_signed_integer(self, dtype):
        with pytest.raises(TypeError):
            UFixedPoint(8, 0, np.array([1]), disbin=True, dtype=dtype)

    # --- arithmetic, exhaustive over 4-bit operands ---

    @staticmethod
    def _pairs():
        a, b = np.meshgrid(np.arange(16), np.arange(16))
        return a.ravel(), b.ravel()

    def test_add_exhaustive(self):
        a, b = self._pairs()
        res = UFixedPoint(4, 0, a, disbin=True) + UFixedPoint(4, 0, b, disbin=True)
        assert type(res) is UFixedPoint and res.bwfw == (5, 0)
        np.testing.assert_equal(res.bdata, a + b)

    def test_sub_wraps_with_borrow(self):
        a, b = self._pairs()
        res = UFixedPoint(4, 0, a, disbin=True) - UFixedPoint(4, 0, b, disbin=True)
        assert type(res) is UFixedPoint and res.bwfw == (5, 0)
        np.testing.assert_equal(res.bdata, (a - b) % 32)
        # The extra MSB is the borrow: set exactly when b > a
        np.testing.assert_equal(res.bdata >> 4, (b > a).astype(int))

    def test_mul_exhaustive(self):
        a, b = self._pairs()
        res = UFixedPoint(4, 0, a, disbin=True) * UFixedPoint(4, 0, b, disbin=True)
        assert type(res) is UFixedPoint and res.bwfw == (8, 0)
        np.testing.assert_equal(res.bdata, a * b)

    def test_mixing_with_signed_raises(self):
        u = UFixedPoint(8, 0, np.array([3]), disbin=True)
        s = FixedPoint(8, 0, np.array([3]), disbin=True)
        for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y):
            with pytest.raises(ValueError):
                op(u, s)
            with pytest.raises(ValueError):
                op(s, u)
        assert u != s and s != u

    def test_neg_raises(self):
        with pytest.raises(TypeError):
            -UFixedPoint(8, 0, np.array([3]), disbin=True)

    def test_sum(self):
        res = UFixedPoint(8, 0, np.array([255, 255, 255]), disbin=True).sum(axis=0)
        assert res.bwfw == (10, 0)
        assert res.bdata[0] == 765

    # --- narrowing with ofmode ---

    @pytest.mark.parametrize("ofmode, ref", [("wrap", _wrap_u), ("saturate", _clip_u)])
    def test_truncate_msb_ofmode(self, ofmode, ref):
        vals = np.arange(256)
        res = UFixedPoint(8, 0, vals, disbin=True, ofmode=ofmode).truncate_msb(4)
        assert type(res) is UFixedPoint and res.bwfw == (4, 0)
        np.testing.assert_equal(res.bdata, ref(vals, 4))

    @pytest.mark.parametrize("ofmode, ref", [("wrap", _wrap_u), ("saturate", _clip_u)])
    def test_lshift_ofmode(self, ofmode, ref):
        vals = np.arange(16)
        for shift in range(1, 4):
            res = UFixedPoint(4, 0, vals, disbin=True, ofmode=ofmode) << shift
            np.testing.assert_equal(res.bdata, ref(vals << shift, 4))

    @pytest.mark.parametrize("method", ["round_from_zero", "round_to_even"])
    def test_round_overflow_ofmode(self, method):
        # 15.75 in 6 bits (fw=2) rounds up to 16, which does not fit into 4 bits
        data = np.array([15.75, 0.25])
        wrapped = getattr(UFixedPoint(6, 2, data), method)(4)
        saturated = getattr(UFixedPoint(6, 2, data, ofmode="saturate"), method)(4)
        np.testing.assert_equal(wrapped.bdata, np.array([0, 0]))
        np.testing.assert_equal(saturated.bdata, np.array([15, 0]))

    def test_round_ties(self):
        fp = UFixedPoint(6, 2, np.array([2.5, 3.5, 2.25, 2.75]))
        np.testing.assert_equal(fp.round_from_zero(4).bdata, np.array([3, 4, 2, 3]))
        np.testing.assert_equal(fp.round_to_even(4).bdata, np.array([2, 4, 2, 3]))

    # --- shifts, widening and bitwise operations ---

    def test_rshift_is_logical(self):
        vals = np.arange(256)
        fp = UFixedPoint(8, 0, vals, disbin=True)
        for shift in range(1, 8):
            np.testing.assert_equal((fp >> shift).bdata, vals >> shift)
            np.testing.assert_equal(fp.truncate(8 - shift).bdata, vals >> shift)
            np.testing.assert_equal(fp.div_pow2(shift).bdata, vals >> shift)

    def test_cast_zero_extends(self):
        fp = UFixedPoint(8, 0, np.array([0xFF, 0x80]), disbin=True).cast(10, 1)
        assert type(fp) is UFixedPoint and fp.bwfw == (10, 1)
        np.testing.assert_equal(fp.data, np.array([255.0, 128.0]))

    def test_bitwise_stays_in_range(self):
        vals = np.arange(16)
        fp = UFixedPoint(4, 0, vals, disbin=True)
        np.testing.assert_equal((~fp).bdata, 15 - vals)
        np.testing.assert_equal((fp & 0b1010).bdata, vals & 0b1010)
        np.testing.assert_equal((fp | 0b1010).bdata, vals | 0b1010)
        np.testing.assert_equal((fp ^ 0b1010).bdata, vals ^ 0b1010)

    def test_binary_and_float(self):
        fp = UFixedPoint(4, 1, np.array([0b1111]), disbin=True)
        assert fp.binary[0] == "111.1"
        np.testing.assert_equal(fp.data, np.array([7.5]))

    # --- dtype ---

    OPS = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "rshift": lambda a, b: a >> 3,
        "lshift": lambda a, b: a << 3,
        "div_pow2": lambda a, b: a.div_pow2(3),
        "and": lambda a, b: a & 0x0F,
        "invert": lambda a, b: ~a,
        "cast": lambda a, b: a.cast(20, 8),
        "reinterpret": lambda a, b: a.reinterpret(0),
        "truncate": lambda a, b: a.truncate(10),
        "round_from_zero": lambda a, b: a.round_from_zero(10),
        "round_to_even": lambda a, b: a.round_to_even(10),
        "truncate_msb": lambda a, b: a.truncate_msb(10),
        "sum": lambda a, b: a.sum(axis=0),
        "getitem": lambda a, b: a[1:3],
        "copy": lambda a, b: a.copy(),
        "empty": lambda a, b: a.empty(),
        "bit_reverse": lambda a, b: a.bit_reverse(),
        "numpy_method": lambda a, b: a.reshape(-1),
    }

    @pytest.mark.parametrize("ofmode", ["wrap", "saturate"])
    @pytest.mark.parametrize("op", OPS)
    def test_ops_keep_type_and_dtype_and_match_int64(self, op, ofmode):
        # 15 bits, so that the 30-bit product still fits the 31 bits available in int32
        rng = np.random.default_rng(0)
        raw_a, raw_b = rng.integers(0, 2 ** 15, size=(2, 64))
        a32 = UFixedPoint(15, 4, raw_a, disbin=True, dtype=np.int32, ofmode=ofmode)
        b32 = UFixedPoint(15, 4, raw_b, disbin=True, dtype=np.int32, ofmode=ofmode)
        a64 = UFixedPoint(15, 4, raw_a, disbin=True, ofmode=ofmode)
        b64 = UFixedPoint(15, 4, raw_b, disbin=True, ofmode=ofmode)
        res32, res64 = self.OPS[op](a32, b32), self.OPS[op](a64, b64)
        assert type(res32) is UFixedPoint
        assert res32.dtype == np.int32
        assert res32.bdata.dtype == np.int32
        if op != "empty":
            assert res32 == res64

    # --- regressions from the Synth project ---

    @pytest.mark.parametrize("hz, step", [(440, 4723), (10000, 107343)])
    def test_phase_acc_step(self, hz, step):
        # phase_acc.vhd: 25-bit unsigned frequency (14.11) times the 18-bit BASE_STEP, keeping the top 26 bits.
        # At 10 kHz the MSB of the frequency is set, which a signed model gets wrong.
        base_step = int(2 ** 18 / 100e6 * 2 ** 18)
        freq = UFixedPoint(25, 11, np.array([hz << 11]), disbin=True)
        res = (freq * UFixedPoint(18, 0, np.array([base_step]), disbin=True)).truncate(26)
        assert res.bdata[0] == step

    def test_sin_table_quadrant(self):
        # Quadrant = 2 MSBs of the unsigned 18-bit phase
        fp = UFixedPoint(18, 0, np.array([0b11 << 16 | 5]), disbin=True)
        assert (fp >> 16).bdata[0] == 3
        assert fp.truncate(2).bdata[0] == 3

    # --- explicit conversions ---

    def test_to_signed_is_exact(self):
        vals = np.arange(256)
        u = UFixedPoint(8, 3, vals, disbin=True, dtype=np.int32, ofmode="saturate")
        s = u.to_signed()
        assert type(s) is FixedPoint and s.bwfw == (9, 3)
        assert s.dtype == np.int32 and s.ofmode == "saturate"
        np.testing.assert_equal(s.bdata, vals)
        np.testing.assert_equal(s.data, u.data)
        # Signed difference of two unsigned values after explicit conversion
        a, b = UFixedPoint(4, 0, np.array([3]), disbin=True), UFixedPoint(4, 0, np.array([5]), disbin=True)
        assert (a.to_signed() - b.to_signed()).bdata[0] == -2

    def test_as_signed_reinterprets(self):
        u = UFixedPoint(8, 0, np.array([0, 127, 128, 255]), disbin=True, dtype=np.int32)
        s = u.as_signed()
        assert type(s) is FixedPoint and s.bwfw == (8, 0) and s.dtype == np.int32
        np.testing.assert_equal(s.bdata, np.array([0, 127, -128, -1]))

    def test_as_unsigned_reinterprets(self):
        s = FixedPoint(8, 2, np.array([0, 127, -128, -1]), disbin=True, ofmode="saturate")
        u = s.as_unsigned()
        assert type(u) is UFixedPoint and u.bwfw == (8, 2) and u.ofmode == "saturate"
        np.testing.assert_equal(u.bdata, np.array([0, 127, 128, 255]))

    def test_conversion_round_trips(self):
        vals = np.arange(-128, 128)
        s = FixedPoint(8, 0, vals, disbin=True)
        assert s.as_unsigned().as_signed() == s
        u = UFixedPoint(8, 0, np.arange(256), disbin=True)
        assert u.as_signed().as_unsigned() == u
        assert u.to_signed().as_unsigned().truncate_msb(8) == u

    def test_conversion_errors(self):
        u = UFixedPoint(8, 0, np.array([1]), disbin=True)
        with pytest.raises(TypeError):
            u.as_unsigned()
        # A 64-bit signed value has no unsigned counterpart in int64 storage
        with pytest.raises(ValueError):
            FixedPoint(64, 0, np.array([1]), disbin=True).as_unsigned()
