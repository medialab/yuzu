# Contributing to `yuzu`

## Running the unit tests

```bash
cargo test
```

To run heavier tests based on embedding models:

```bash
cargo test --test heavy
```

To run *all* tests at once:

```bash
cargo test --all-targets
```