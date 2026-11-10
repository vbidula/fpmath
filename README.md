# fpmath

Implements fixed point math in Python. Useful for different personal projects, such as writing HDL testbenches in 
Python.
## Design notes

### Arithmetic follows the hardware, casts are explicit

The binary point (`fw`) is a mental construct: hardware only operates on integers, so `FixedPoint` operations
work on the raw integers and never align binary points or resize operands implicitly. If the firmware forgets a
cast, it produces a wrong result, and the model is meant to reproduce exactly that instead of hiding it.
Combining operands with different fraction widths emits a `FixedPointFormatWarning`; escalate it to an error in a
test suite with:

```ini
[pytest]
filterwarnings = error::fpmath.FixedPointFormatWarning
```

Shifts follow VHDL `numeric_std`: `>>` is an arithmetic shift rounding toward -inf (`shift_right()` or slicing
off LSBs), while `div_pow2()` rounds toward zero (`x / 2**n`).

### Overflow handling (`ofmode`)

Widening operations (`+`, `-`, `*`, `cast()`, `sum()`) grow the bit width so they never overflow, and are not
affected by `ofmode`. Saturation in hardware is explicit logic that comes with narrowing, so `ofmode` applies only
to operations that can lose MSBs:

| Operation | `wrap` (default) | `saturate` |
|---|---|---|
| `truncate_msb()` | drops MSBs, like VHDL slicing `x(3 downto 0)` | clamps to the representable range |
| `round_from_zero()`, `round_to_even()` | rounding past the maximum wraps (7.75 → -8 in 4 bits) | clamps (7.75 → 7) |
| unary `-` | `-(-8)` stays -8 in 4 bits, like `numeric_std` | becomes 7 |
| `<<` | MSBs shifted out are dropped | clamps |
| `from_float()` | out-of-range floats wrap | clamps |

Decisions behind this:

- **`wrap` is the default.** It is what plain VHDL does. A saturating default would make the model look correct
  even when the firmware is missing a saturation stage. Declare `ofmode="saturate"` on signals that the firmware
  explicitly saturates; results inherit `ofmode` from their left operand.
- **Raw binary input is never clamped.** `FixedPoint(..., disbin=True)` always reinterprets the integers as a
  `bw`-bit two's complement pattern, regardless of `ofmode`. This is how DUT outputs are loaded: cocotb returns
  them as unsigned integers (-1 in 14 bits arrives as 16383), and clamping them would corrupt every negative value.
  Narrowing operations go through a separate internal path (`_narrow()`) that applies `ofmode`.
- **`numeric_std` `resize()` is not modelled.** Narrowing a `signed` with `resize()` keeps the original sign bit
  and the remaining LSBs, which is neither wrapping nor saturation. Prefer slicing or explicit saturation in the
  firmware.

### Storage dtype

`dtype` only selects the numpy integer type used to store the values; it must be a signed integer type and limits
`bw` to its number of bits (results that would not fit raise a `ValueError`). Every operation keeps the dtype of
its operand, and operations on two `FixedPoint` with different dtypes use the wider one, which never changes values.
Results are identical for any dtype wide enough to hold them.
