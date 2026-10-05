# Training Data

## General Workflow

1. Store your data in some format (heavily recommended to use a binpack-like format)
2. Convert data files from your this format to a data format that `Trainer` can ingest if needed
3. Shuffle the individual converted files
4. Interleave the shuffled files

## Builtin Data Loaders

You can easily write a dataloader for your own format if you wish, but bullet already contains loaders for the most common formats.

### Binpacks

Stockfish, Monty and Viridithas format "binpacks" can be loaded using `SfBinpackLoader`, `MontyBinpackLoader` and `ViriBinpackLoader` respectively.
Binpacks stores entire games contiguously to achieve great compression and hence need a filter function to be passed to remove e.g. noisy positions.

I would recommend using Viriformat binpacks as they are the most commonly used amongst people who generate their own data (and thus there are reference
implementations in many programming languages, as well as many utilites available).

### ChessBoard aka "bulletformat"

This is a simple and fast-to-load data format that can be loaded with `DirectSequentialDataLoader`.
It is suitable for training small networks.
However it is recommended to generate and store data in a binpack-like format, and only convert to this format if bottlenecked by data loading speed.
It throws away information such as the side-to-move (the record is stored stm-relative), halfmove counter, castling rights, etc.
The `bullet-utils` binary contains utilities for shuffling and interleaving these data files, as well as converting from some other data formats.

In particular, you can convert to this data type from a text file that contains a list of data points in the following form:
- each line is of the form `<FEN> | <score> | <result>`
- `score` is white relative and in centipawns
- `result` is white relative and of the form `1.0` for win, `0.5` for draw, `0.0` for loss

## Fixed-size shuffle and interleave

`bullet-utils shuffle` and `bullet-utils interleave` accept `--record-size 40`
for shogi PackedSfenValue data. The default record size is 32 bytes. Supply
`--seed` to reproduce the record order for the same inputs, memory limit, and
interleave settings. `record` and `block` modes require complete records;
`concat` copies the input bytes in their listed order. Record size must be
positive. The shuffle memory budget must be at least 1 MiB and fit one complete
record; `--mem-used-mb` counts 1,048,576-byte units.

```sh
cargo run --release -p bullet-utils -- shuffle \
    --input input.psv --output shuffled.psv --mem-used-mb 256 \
    --record-size 40 --seed 123
cargo run --release -p bullet-utils -- interleave a.psv b.psv \
    --output combined.psv --record-size 40 --mode block --seed 123
```

`interleave` takes at least two inputs; they may be listed before, after, or
between the options. `concat` does not use `--record-size`.

### Outputs

Keep inputs unchanged while processing. For `interleave`, the output must be
separate from every input, including symbolic and hard links to one. `shuffle`
may write back to its own input (`--input x --output x`): the input is read
completely, or split into temporary files, before the output is replaced.

`shuffle`, `interleave`, and `convert` write to a temporary file beside the
output and rename it over the output only after successful completion, so a
failed run leaves an existing output untouched. The `montybinpack` and
`viribinpack` subcommands write to their output directly.

- An output that is a symbolic link is written through: the temporary file is
  created beside the link target, the target is replaced, and the link stays.
  A dangling link gets its target created.
- Devices, FIFOs, and other existing non-regular outputs such as `/dev/null`
  are written in place without a temporary file. `interleave` checks the size
  of what it wrote, so it reports an error for outputs that do not retain data.
- Existing read-only outputs are refused, and the directory of the output must
  permit creating and renaming files, even when the existing file is writable.
- Existing file permissions restrict the temporary file from creation and are
  retained on publication. New files follow the process umask.
- Replacement changes the file identity: other hard links keep the previous
  contents, and ownership or extended access controls are not copied.
- The output file is synced before the rename, and on Unix its directory is
  synced afterwards, so a completed command survives power loss on filesystems
  that honour both. If the directory sync is rejected, a warning is printed and
  the command still succeeds.
- On Windows, readers that deny file replacement can prevent publication and
  leave the existing output intact.

### Disk space

Disk-backed shuffling (input larger than `--mem-used-mb`) stores a full copy of
the input as temporary parts in the current working directory, then writes the
temporary output beside the destination. An existing output is kept until the
rename, so replacing one needs room for the parts, the new output, and the old
output at the same time: about three times the input size, or twice when
the output does not exist yet. `interleave` and `convert` need room for the new
output in addition to an existing one.

### Temporary files and interruption

| Name | Location | Created by |
| --- | --- | --- |
| `.bullet-output-<output name>.<random>` (file) | directory of the output, or of the link target | `shuffle`, `interleave`, `convert` |
| `.bullet-shuffle-<output name>.<random>/` (directory) | current working directory | disk-backed `shuffle` |

Each invocation removes only the entries it created, on success and on errors
it reports itself; an unrelated `tmp` directory is left intact. There is no
signal handling: interrupting the command (Ctrl-C, `kill`, a closed terminal)
or losing power leaves these hidden entries behind, and they can be as large
as the dataset. The destination itself is not modified in that case. Leftover
entries are never reused, so they are safe to delete once no running command
is writing to that output; `ls -A` shows them.

### Exit status

Successful commands, `--help`, and `--version` return 0. Usage errors (missing
or unknown arguments, invalid values, fewer than two `interleave` inputs)
return 2. Processing errors return 1. Scripts should treat any nonzero status
as failure instead of requiring a particular failure code.
