# Incremental Windows installer (unreleased)

User-approved 2026-10-23; issue [85](https://github.com/rensutheart/napari-vipp/issues/85).
Development build: `0.16.0a8.dev1`; this is not a public release.

## Routes and consent

- Automatic prefers incremental only for an older, healthy, owned managed
  installation with the same base Python/ABI/compute track and an exact new VIPP
  wheel whose complete dependency closure is already satisfied.
- Incremental update only blocks with a reason if those conditions fail.
- Fresh installation creates an independently resolved replacement and is the
  repair/new-install/fallback route. It does not modify an unrelated environment.
- Changing the method invalidates prior approval. Local checking does not wait
  for an online resolution. Fresh-route approval explicitly authorizes subsequent
  resolution; all non-package consent fields and current ownership are rechecked
  before any candidate mutation. Exact package results remain in the setup log.

## Implementation boundary

Incremental means independent, verified copies of installed dependency RECORD
files into a new candidate, not a live in-place pip upgrade and not writable
hard links. Old VIPP code and bytecode are excluded. Dependency entry points are
regenerated for the new candidate interpreter; copied RECORDs are rewritten.
Only the new VIPP wheel is installed by pip. The existing acceptance, journal,
ownership/shortcut/registration promotion, cancellation and rollback boundaries
remain in force. The old accepted environment remains the rollback boundary.

Metadata/closure review is local; every retained file hash is verified during
copy. Stale, corrupt, editable, foreign, redirected, overlapping or unsupported
layouts fail closed. This first path does not incrementally change dependency
versions: choose Fresh installation if the new release needs different ones.
Copying and antivirus scanning still cost time and require replacement disk space.

Fresh routes keep pip's download cache enabled and additionally retain exact
SHA-256-addressed verified wheel artifacts. Only misses are downloaded through
an isolated, hash-locked pip child; installation uses local hash-locked files with
`--no-index --no-deps`. Existing invalid/foreign cache files are not overwritten.
Cache retention is not yet an automatic size-bounded eviction policy.

## Qualification and handoff

Run focused installer/controller/GUI/cache/copy tests, including stale approval,
failure, cancellation, byte/hash damage, fresh fallback and launchers. Exercise
real pip/candidate/acceptance flows only in explicit disposable roots, with
per-instance known-folder services, no shortcuts/registration, and no host
installation mutation. Inspect a new owned native preview window and build the
development EXE from the tested source/wheel. Existing CPU/CUDA installations,
unfinished AI work, public tags and release bytes are not test fixtures.

Record actual outcomes and limits under
`VIPP-local-tests/incremental-installer-20261023`. A built EXE is not itself a
passing native install/update or a published release. Replace only the user's
explicitly selected installation after qualification, never all managed roots.
The companion manual change is in `vipp-mkdocs-incremental-installer` and is
explicitly unreleased.
