# Publishing burn-xlstm

The first release is prepared as `burn-xlstm` **0.1.0**, licensed under
Apache-2.0. It has not yet been published. The package name was unregistered
when checked on 2026-11-02; this check does not reserve it.

## First-time account setup

1. Sign in to [crates.io](https://crates.io/) using GitHub.
2. Verify your email in [account settings](https://crates.io/settings).
3. Create a publishing API token at [API tokens](https://crates.io/settings/tokens).
4. Run `cargo login` locally and enter the token at its prompt.

## Release commands

From a clean checkout of the commit you intend to release:

```sh
cargo publish --dry-run --locked
cargo publish --locked
```

The dry run packages and builds the crate without uploading it. The second
command publishes it. The archive includes `LICENSE`, `NOTICE`, the README,
source, tests, and reference fixtures.

After a successful upload, tag that same commit:

```sh
git tag -a v0.1.0 -m 'Release 0.1.0'
git push origin v0.1.0
```

Update the README's unpublished status and installation example after the
release is available. Users can then depend on:

```toml
burn-xlstm = "0.1.0"
```

Future releases need a new version in `Cargo.toml`; published versions cannot be
overwritten. See the official [Cargo publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
