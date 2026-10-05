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

Keep inputs unchanged while processing. The output must be separate from every
input, including symbolic and hard links. Both commands write to a temporary
file beside the output and replace the output only after successful completion.
Output destinations must be regular files or new file paths. Devices, FIFOs,
and streams such as `/dev/null` or `/dev/stdout` are not supported. Existing
read-only outputs are refused, and the parent directory must permit temporary
file creation and replacement, even when the existing file itself is writable.
Existing file permissions also restrict temporary files from creation and are
retained on publication. New files follow the process umask. Replacement changes the file identity:
hard links keep the previous contents, output symbolic links are replaced, and
ownership or extended access controls are not copied.

Disk-backed shuffling creates a unique `.bullet-shuffle-*` directory in the
current working directory. Each invocation removes only its own directory;
an unrelated `tmp` directory is left intact. Normal errors also clean up the
temporary files. A forced process termination can leave `.bullet-shuffle-*`
directories or `.bullet-output-*` files behind; remove them only after confirming
that no running command uses them. Atomic replacement requires the filesystem
to support a rename beside the destination, and does not guarantee recovery
after power loss. On Windows, readers that deny file replacement can prevent
publication and leave the existing output intact.

Usage errors return exit status 2; processing errors return 1. Successful
commands, help, and version requests return 0. Scripts should treat any nonzero
status as failure instead of requiring a particular failure code.
