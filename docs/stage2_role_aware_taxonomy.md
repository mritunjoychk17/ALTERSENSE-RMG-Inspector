# Stage 2 Role-Aware Taxonomy

## Why

Not every workstation is a sewing station.

Some workers:

- get fabric
- fix or align fabric
- inspect fabric
- pass fabric to the next operator

Those stations should not be forced into `sew`.

## Station Roles

### `sewing`

Use for stations where the worker operates a sewing machine.

Allowed labels:

- `idle`
- `get`
- `put`
- `align`
- `pass`
- `inspect`
- `sew`
- `adjust_machine`
- `uncertain`

### `prep_pass`

Use for stations where the worker mainly prepares, aligns, checks, or forwards fabric.

Allowed labels:

- `idle`
- `get`
- `put`
- `align`
- `pass`
- `inspect`
- `uncertain`

### `generic`

Use when the station role is still unknown.

Allowed labels:

- `idle`
- `get`
- `put`
- `align`
- `pass`
- `inspect`
- `sew`
- `adjust_machine`
- `uncertain`

## Current Repo Defaults

- `station 2` -> `sewing`
- `station 4` -> `sewing`

If a queue CSV has a `station_role` column, that explicit value should override the default mapping.

## Label Meaning

- `idle`: worker is present but not doing productive hand work
- `get`: worker retrieves material toward their own work area
- `put`: worker places or sets material on the table or machine area
- `align`: worker fixes, arranges, smooths, or repositions fabric
- `pass`: worker forwards or hands material to the next stage or zone
- `inspect`: worker checks or verifies fabric/work quality
- `sew`: worker actively feeds or guides fabric through the machine needle/feed area
- `adjust_machine`: worker manipulates machine hardware or settings
- `uncertain`: evidence is too weak or ambiguous

## Rule

Having a sewing machine at the station does not mean every segment should be `sew`.

At a `sewing` station, valid labels still include:

- `align`
- `pass`
- `inspect`
- `get`
- `put`

Use `sew` only when the frame clearly shows real machine-side sewing interaction near the needle/feed area.
