# Incremental Windows installer (unreleased)

User-approved 2026-10-23; issue [85](https://github.com/rensutheart/napari-vipp/issues/85).
Development build: `0.16.0a8.dev5`; this is not a public release. The earlier
`dev1`/`dev2`/`dev3`/`dev4` artifacts and failed trials remain retained as qualification evidence.

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

## Explicit registration repair

A stale Windows Apps & Features entry is not permission to overwrite it during
an ordinary update. Keep the existing update/uninstall ownership guards strict.
Offer a separate read-only registration repair review only for a completely
verified, same-installation entry. Bind confirmation to exact typed registry
values, ownership bytes and cached uninstaller identity; save an exclusive,
durable backup before writing. Reject unknown fields/types/subkeys, foreign or
partial identities and stale confirmation. Recheck the installation after a
successful repair, without automatically updating or opening VIPP. The repair
does not rebuild packages or change user workflows.

The obsolete cached uninstaller may already have been retired. An exact missing
path is recorded and displayed as missing, not treated as verified old executable
bytes. This narrow case can still restore the complete same-installation entry
to the currently verified ownership/uninstaller after explicit confirmation.
Present-but-altered or redirected old files remain blocked. A change in that
old-file state invalidates the review. No old uninstaller is ever executed.

## Qualification and handoff

Open VIPP is a separate launch boundary. Discover the selected environment and
its declared base Python without probing another interpreter; use those DLL
directories first, exclude the frozen setup directory and cross-runtime Python/Qt
loader overrides from the child, and temporarily reset/restore the frozen Windows
DLL directory around process creation. Do not alter persistent PATH, special-case
a caller/tool/vendor, or reinstall packages to correct inherited search state.
The read-only diagnostic `--open-installed` route requires an explicit compute
track and verified owned, healthy installed-version identity. It never enters
setup preparation, transaction recovery, registration repair or package apply;
the same backend Open path is used by the setup button. This permits frozen
launch regression testing against an existing disposable installation.

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
