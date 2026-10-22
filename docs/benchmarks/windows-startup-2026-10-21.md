# Windows startup investigation — 2026-10-21

## Slow-launch trace and first-frame follow-up

The user recorded `VIPP-startup.etl` with Windows Performance Recorder during
a slow installed-shortcut launch. The 213.069 s recording began at
2026-10-21 19:12:56.084621 UTC. This was **not** a post-reboot cold launch.
The ETL reports **1,496,739 lost events**. Reports also warn about duplicate and
unmatched events, so the following observations are incomplete and cumulative
durations can overlap. They must not be subtracted from wall-clock startup time
as an exact predicted saving.

- The launcher Python process began 16.085 s into the trace; the main VIPP child
  began at 34.227 s. Approximately 18 s elapsed before child creation.
- The child's retained log reports readiness after 141.514 s and 35.281 s of
  process CPU time. Creating the viewer took 83.219 s and importing VIPP took
  45.009 s. CPU execution alone does not account for the elapsed delay.
- `xperf -a minifilterdelay -apfp` reports **129.272 s accumulated delay for
  WdFilter.sys across pythonw.exe processes**, versus 11.892 s for fileinfo.sys
  and about 0.6–3.2 s each for the other reported filters. The installed driver
  identifies itself as Microsoft's antimalware file-system filter.
- Microsoft's `Get-MpPerformanceReport` records **3,905 valid real-time scans**
  for these Python processes, with a **32.153 ms median** and **153.664 s
  cumulative scan duration** across the recording. Main-child scans include
  post-readiness background work. One unmatched 204.547 s Zarr record began
  before that process existed and was excluded; its raw report total is invalid
  as a startup diagnosis.
- Restricting valid main-child scans to before readiness gives 2,848 scans,
  120.749 s cumulative duration, and 85.717 s covered by the union of their
  recorded intervals. The interval union removes overlap between scans, but
  still does not prove that every scan blocked the startup-critical thread.
- Before the child starts, the launcher has 460 scans totaling 13.126 s.
  **316 are font files**, totaling **11.371 s**. This explains a substantial
  part of the user's reported wait before the first splash frame. Deferring
  process creation alone cannot remove font discovery during initial painting.

These independent reports identify Defender scanning as a substantial
contributor to the reproduced slow launch. They do not establish that every
second is caused by Defender, identify Trellix as a cause, or provide a
version-controlled comparison with the other computer. No antivirus exclusion,
security setting, font registration or power policy was changed. Microsoft's
[performance analyzer reference](https://learn.microsoft.com/en-us/defender-endpoint/performance-analyzer-reference)
documents the scan report. Raw local reports remain in
`tmp/startup-investigation-20261021/system-trace-2113`.

### First-frame correction, unreleased after 0.16.0a5

The standalone Windows splash now paints a packaged bitmap with a live,
font-independent elapsed clock before font preparation. A worker prefetches
registered local fixed-drive fonts and initializes Qt's thread-safe font
database. The ordinary stage/error controls are then built on the GUI thread,
preserving all cached elapsed and status updates. Startup log preparation and
child process creation also run off the GUI thread, after the first paint.
The elapsed clock is not reset to disguise startup work.

In a native Windows diagnostic using the installed dependencies and current
source, with process creation deliberately delayed four seconds, the corrected
first paint occurred at **1.617 s**, displaying **Elapsed 0:00**. The maximum
20 ms heartbeat gap was **0.426 s**, with 219 callbacks. A previous synchronous
process-creation test blocked the UI for 4.041 s. These are uncontrolled warm
diagnostics, not installed-shortcut or reboot-cold acceptance measurements.
The current focused startup, splash, font-prefetch and architecture checks pass
**64 tests**. The generated PNG was inspected on the native Qt platform; an
offscreen rendering with missing glyphs was rejected before packaging.

The complete development build was installed through the ordinary GUI repair
flow, run `d918a7d6-252e-4fc7-b3fd-9c2b1932f59c`. Ownership was committed at
23:12:55 SAST, with status `succeeded`, no registration warning, and a passing
CUDA check on the RTX 4050. The Desktop shortcut targets the new owned
environment's `vipp-app.exe --desktop`. The original running VIPP session and
its environment were preserved. The intermediate process-thread-only repair
was cancelled normally before this complete build was applied.

- Complete wheel SHA-256:
  `fd483db933ff70c9ef5ed7d86cff0fafc990c098867f389f494ada6cfb17837f`.
- Complete setup SHA-256:
  `2db5e2f22f90007f5ae95a7b066bbd114dadd0769c10ee55cc695c43ff89b6c6`.
- All 298 runtime source files and the first-frame PNG in the wheel match the
  checkout. This packaging comparison does not read the installed runtime;
  installed-file verification is deliberately deferred until after first-launch
  measurements to avoid warming those files through verification.

### Complete build: actual Desktop-shortcut measurements

After setup and regression activity had finished, three ordinary launches used
the actual Desktop `VIPP.lnk`, without application import hooks, source-path
overrides, diagnostic recovery roots or smoke-exit flags. The original user's
VIPP session stayed open. No reboot or cache eviction was performed. The first
launch below is the first measured launch of the newly installed environment,
not a verified cold boot.

| Launch | New splash window first noticed (s) | Painted screenshot obtained by (s) | Interface ready (s) | Splash process exit (s) |
| --- | ---: | ---: | ---: | ---: |
| First after installation | Not measured | Not measured | 114.115 | 114.599 |
| Repeat 1 | 1.129 | 1.627 | 11.922 | 12.418 |
| Repeat 2 | 1.143 | 1.536 | 12.355 | 12.846 |

Both repeat screenshots show **0:00** on the elapsed clock. Window discovery
polls every 200 ms and adds API latency; screenshot completion is a conservative
visible-by bound, not a video-exact first-pixel measurement. The first attempt's
external window observer had a scripting error before polling began. Its later
screenshot at elapsed 20 seconds is excluded from splash-onset evidence; the
independent first-launch ready/process-exit measurements remain valid.

**All three launches offered existing interrupted-workflow copies and paused
automatic calculation of the sample workflow.** Ready here means the interface
and recovery dialog, not a finished default sample analysis. The copies were
kept for later. These totals must not be presented as a controlled comparison
with earlier runs that completed the initial sample calculation, or with the
other computer. The first child's wall/CPU times were 104.714/15.188 s, versus
10.636/10.109 s and 10.796/10.156 s on the two repeats. Substantial variable waits
remain on the first launch, despite the fast repeat splash observations.

After all launches, all **348 packaged runtime/resource files** in the installed
environment matched the reviewed wheel byte-for-byte. This includes the 298
Python modules and first-frame PNG. Two untouched benchmark-only synthetic
working copies from the closed test instances were moved to an investigation
backup after validating their timestamps, default sample parameters and free
leases. Every backup hash was rechecked. Existing user recovery copies and the
original running session were preserved. The final updated instance was left
open, with the recovery dialog dismissed using **Keep for later**.

### Ordinary-text clock follow-up, 2026-10-22 (source preview only)

The initial clock now paints ordinary UI-font digits from a packaged glyph
strip instead of using a seven-segment display. Static text and glyphs are
rendered together on native Windows; smooth scaling is enabled. Fonts still
prepare after the first frame. The full text splash retains its previous
styling. The fallback uses the ordinary splash if either raster asset is absent.
This follow-up is not in the installed development build measured above.

Two native previews compare the original splash from `57248e0` with the revised
early splash, both staged at 0:03. The early frame is deliberately held for its
screenshot; these images are not actual-launch timing observations. They are
retained under `tmp/startup-investigation-20261021/text-clock-comparison`.

In a fresh native process with uncontrolled warm caches, Qt imports took
0.447 s, QApplication creation 0.121 s, and explicit font-database discovery
0.114 s (334 families). Building and painting the original text splash afterward
took 1.141 s, giving 1.823 s overall from the first Python statement. The
0.114 s measures that explicit database call, not all text preparation or a
cold font cost. The earlier 316 font scans / 11.371 s cumulative duration
remain evidence of a much slower scanning condition, not an exact predicted
saving from this change.

A separate native source diagnostic imposed four seconds of font delay and
four seconds of process-creation delay. The revised ordinary-text clock painted
at 1.626 s with 0:00 displayed, and the 20 ms heartbeat's maximum gap was
0.528 s. No actual application child was started by these preview diagnostics.

The focused splash/launcher/font/architecture checks passed 54 tests in the
existing CPU development environment. An initial run in the older root `.venv`
passed 50 cases and failed three launcher cases because that environment lacks
`centrosome`; the passing CPU-environment rerun includes those cases. Source lint,
22 manual content contracts, 50 manual routes and the strict manual build
passed. The changed launching page was inspected at desktop and 600 px widths.

### Sharper lettering and matching information, 2026-10-22 (source preview)

This supersedes the source-only glyph-strip preview above. The early splash
now paints licensed DejaVu Sans outlines, packaged as VIPP Splash Sans, so text
scales with the display without looking up or registering a system font. The
background and wordmark are prepared at 4x resolution. Runtime version text,
the current stage/message, step count and elapsed time are live fields; the
progress bar uses the same snapshot percentage and the profile note uses the
same source as the ordinary splash. Grey version text increases from 11 to
12 px, step/time text from 10 to 11 px, and the footer from 9 to 10 px in both
paths. Font permission/disclaimer notices are packaged alongside the outlines.

Native early/full previews were inspected and exported losslessly at 1280x780
(2x), staged at step 2 and 0:03. These are rendered previews, not live-launch
timings. They are under `tmp/startup-investigation-20261021/sharp-splash-preview`.
A diagnostic imposing four seconds each of font and process-creation delay
painted at 2.408 s with 0:00 on the clock; the 20 ms heartbeat's largest gap was
0.760 s. This run overlapped other development checks and is not comparable
with earlier uncontrolled warm timings or an installed-shortcut measurement.

The focused startup/launcher/font/architecture suite passed 54 tests. After the
final high-resolution normal-splash logo change, all 19 splash cases passed
again. Lint and wheel builds passed; both runtime modules and all four new
artwork/lettering/license resources in the final wheel match source bytes, with
no obsolete wheel-only branding files. The manual passed 22 content contracts,
50 routes and its strict build; the changed page was inspected at desktop and
600 px widths. This typography follow-up has not been installed or published.

### Inter Medium vector implementation, 2026-10-22 (unreleased)

After the status-line comparison, the user selected Inter Medium. Its prepared
outlines now paint the status in both the early and ordinary splash; it does
not switch typeface when the other text controls finish initializing. DejaVu
Sans regular remains the early grey lettering. The two subsets and their
licenses are packaged separately, and neither typeface is installed or
registered by the runtime renderer. Inter is shaped at the actual 14 px design
size before its outlines are normalized for scalable painting.

Native Windows checks used per-process scaling overrides to exercise actual
device ratios of 1.0, 1.25, 1.5 and 2.0. All six standard stage messages fit,
and a normalized status-line capture matched pixel-for-pixel before and after
font preparation at every ratio. Ready/error colours were checked. A small
white-to-off-white transition found during this check was corrected by making
the status colour explicit; regression checks cover starting, ready and failed
states at all four scale factors. The default splash fitted this computer's
available desktop at each setting; this does not qualify every physical monitor
combination, accessibility setting or smaller screen.

The expanded startup/splash/font/architecture suite passed **66 tests** and
application documentation checks passed **26 tests**. Source/scripts lint,
manifest validation, wheel builds, 22 manual content contracts, 50 routes and
the strict manual build passed. Desktop and 600 px manual renders were reviewed.
All **308 runtime-Python and branding files** in the final wheel match source
bytes, with no extra wheel-only branding resources.

A real native launch from the unpacked wheel, using the installed dependencies,
completed the default CPU sample and exited normally after readiness. Recovery
and timing-history data were isolated under the investigation directory. Its
48.534 s process lifetime includes normal shutdown and is a functional smoke
check, not an installed-shortcut comparison with the recovery-paused launches.
The later final package differs from that smoke package only in the splash's
phase-colour consistency; the app/sample modules are unchanged.

A controlled native launcher diagnostic, with four-second font and process
creation delays imposed, painted at **1.511 s**, showing **0:00**. The largest
20 ms heartbeat gap was 0.471 s. This validates first paint ahead of that work;
it is not a reboot-cold or ordinary installed-shortcut launch. Evidence and
the status-line-only preview are in
`tmp/startup-investigation-20261021/inter-vector-final`. The implementation is
not installed or published yet.

### Installed development-build validation (preceding clock revision)

The installed-build qualification below predates both source-only typography
follow-ups; it must not be presented as installation of those follow-ups.

The focused startup/font/architecture checks pass 64 tests. A full CPU development
run completed 14,286 cases: 13,779 passed, 479 skipped, and 28 failed. Twenty-seven
failures were caused by stale a3 editable-distribution metadata alongside a5
source/recorded manifests. The other was the deferred-uninstall helper waiting
for the real concurrently running installer's shared single-instance mutex,
then reaching its test's 60 s timeout. The standalone helper rerun passed.

After setup closed, the development environment's editable metadata was refreshed
to a5. All affected reproduction files, the manifest-version widget case and the
uninstall-helper case then passed: **57 passed in 95.23 s**. All original failed
cases are covered by that passing rerun. No scientific version guard or installer
mutex was relaxed. This is a full run plus corrected targeted reruns, not a claim
that one new uninterrupted full run passed.

Source/scripts lint and npe2 manifest validation passed. Whole-directory lint
also visited pre-existing untracked `tmp` dependency trees and reported unrelated
errors there. Package and development-installer builds passed. The companion
manual passed 22 content contracts, 50 application-to-manual routes and a strict
MkDocs build; desktop light/dark and a 600 px narrow dark view were reviewed.
An attempted local preview server was rejected by automatic approval review;
direct local-file rendering completed the narrow check without a server.

## Finding and limits

VIPP eagerly loaded an OME-Zarr/Dask and OME-XML dependency tree while importing
the workflow interface, before the user selected any file. Moving those imports
to the reader/writer calls removed 1,293 modules from an isolated interface
import and reduced its median duration from **7.627 s to 3.722 s** in four
alternating measurements per variant on the same computer and environment.
This is an import improvement, not a claim that complete startup is twice as
fast. The earlier integrated a5 build took **142.159 s on its first launch**
and **158.799 s on a later repeat**, measured through splash-process exit.
Its largest remaining stages were creating the napari viewer (about 61 s) and
importing VIPP (about 41 s). A real cold launch after a reboot is still required.

The hardware follow-up below recorded a much faster, instrumented normal-workflow
shortcut launch (16.265 s) and approximately 2,834 MiB/s in a limited read-only
storage probe. The same machine can therefore load the application much faster.
The later slow-launch system trace attributes substantial file-filter delay to
Microsoft Defender, and its scan report records thousands of real-time scans
of VIPP's Python files. It also identifies hundreds of font scans before the
application child starts. See the trace follow-up below for measurements and
the limits of this attribution.

An installed VIPP session was already running when investigation began, so the
initial investigation did not reproduce the reported cold delay. Those initial
installed launches reached readiness in 19.9–48.6 s. After the latest development
build was installed, actual Desktop-shortcut launches reproduced a multi-minute
delay, as recorded below. No reboot, cache eviction, antivirus setting change
or driver change was performed. Neither the initial observations nor the latest
stage timings alone established the host-level cause of the long waits; the
later system trace supplies that attribution evidence.

The changes are unreleased after `0.16.0a5`. The initial measurements below used
the earlier source baseline `9f75a17` and the installed `0.16.0a2` environment.
Following the user's version correction, both checkouts were refreshed and the
changes integrated on application `57248e0` (`v0.16.0a5`) and manual `c7f4579`.
Those earlier measurements do not qualify the latest combined build. Native
installation and latest-version checks are recorded separately below.

The user confirmed that the desktop's approximately 30 s launch uses a different
VIPP version (exact version not supplied). The reported cross-machine ratio
therefore mixes software-version and hardware/environment differences. It is
not a controlled hardware comparison. The same-machine import experiment
below holds the interpreter and dependencies fixed.

## Environment and method

- Intel Core i7-13700H; 34,029,125,632 physical RAM bytes (about 31.7 GiB).
- SK hynix BC901 1 TB NVMe drive; Windows Balanced power scheme.
- NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB reported memory.
- Installed VIPP `0.16.0a2`, napari `0.9.1`, Python 3.12, PyQt6 `6.11.0`,
  NumPy `2.5.1`, SciPy `1.18.0`, scikit-image `0.26.0`, CuPy CUDA 13 `14.1.1`.
- Source baseline `9f75a171cc8875e8d3084bb7e5b4655fb476fbaa` (`0.16.0a3`).
- Native Windows Qt application launches used the existing
  `smoke_exit_after_ready` path. Each synthetic test window closed normally;
  the existing user's application was left running. Timing history was
  redirected to investigation-only files.
- Full startup comparisons reused the installed interpreter and dependencies,
  with only the source root changed when testing the checkout.
- Import comparisons ran in eight fresh processes in before/after/after/before
  order, repeated twice. The baseline substituted only the two original reader
  modules from the recorded commit; all other checkout code was shared.
  These imports did not construct a viewer. Module counts include the small
  common diagnostic harness.

Machine-readable, path-redacted results are in the
[measurement record](windows-startup-2026-10-21.json). Raw import traces,
cProfile output and local logs remain under `tmp/startup-investigation-20261021`
because they contain local paths and environment details.

## Installed startup measurements

| Launch | Instrumentation | Ready (s) | Process CPU at ready (s) |
| --- | --- | ---: | ---: |
| Auto, first measured | Stage timestamps and importtime | 48.633 | 42.609 |
| Auto, detailed profile | Stage timestamps and cProfile | 39.340 | 36.391 |
| CPU | Stage timestamps | 19.895 | 18.297 |
| Auto, repeat | Stage timestamps | 42.121 | 37.234 |

These runs have uncontrolled file/cache and competing-work conditions, and
instrumentation differs. In particular, the CPU/Auto totals are **not** a
controlled measurement of the GPU penalty.

The first Auto launch spent 15.846 s creating/loading the napari viewer,
16.998 s importing VIPP, 1.968 s constructing its interface, and 13.178 s
preparing and showing the welcome workflow. Napari exposes lazy imports, so
the viewer stage includes much of napari's actual import work.

The detailed Auto profile recorded:

- 27.2 s cumulatively inside Python import loading;
- 14,377 `stat` calls, with about 7.0 s self time;
- 3,448 Python code-file opens, with about 2.7 s self time;
- approximately 2.0 s generating the complete bundled sample collection even
  though startup selects only the synthetic volume;
- approximately 1.1 s in `probe_compute_environment`, including its nested
  runtime/library checks. Provider import cost is additional and accounted
  elsewhere in the call tree.

The default user CuPy kernel cache had no files written on the measurement day;
its latest modification was 2026-10-06. This does not support fresh compilation
as an explanation of the measured launches, but it is not a complete trace of
driver or first-use CUDA activity during the original unrecorded cold launch.

Nested cumulative times must not be added together. The profile is dominated
by import/file access, not a large scientific operation. SecurityCenter listed
Windows Defender and Trellix, and the Defender service was running; that does
**not** prove scanning caused the cold delay. At this initial stage no trace
attributing file latency to a scanner had been captured. The later trace below
records Defender's contribution during a reproduced slow launch.

## Import change and measurement

`core/io/ome_zarr.py` now loads Zarr, Dask callbacks and ome-zarr only inside
the functions that use them. `core/io/tiff.py` loads ome-types only when there
is OME-XML to parse. Export, label metadata, preview cancellation and exact
source-window behavior retain their existing implementations and dependencies.
In the deferred-only measurements below, the first actual use of a format pays
its import cost once per process. The follow-up background warm-up below moves
that work ahead of later file selections, after interface readiness.

| Isolated interface import | Before (s) | After (s) |
| --- | ---: | ---: |
| Measurement 1 | 8.009 | 3.712 |
| Measurement 2 | 7.628 | 3.732 |
| Measurement 3 | 7.003 | 3.830 |
| Measurement 4 | 7.625 | 3.367 |
| Median | 7.627 | 3.722 |
| Loaded modules | 2,580 | 1,287 |

One complete source launch with importtime fell from 25.235 s to 18.121 s;
trace records fell from 3,867 to 3,035. This single full-startup pair has more
noise and cache-order bias than the alternating import experiment. It is
supporting evidence only, not a release acceptance threshold or a prediction
of the three-minute cold launch after this fix.

## Retained diagnostics and next cold measurement

### Follow-up: load readers in the background after readiness

The measurements above describe the original deferred-import change. Following
user feedback, supported GUI launches now also warm the required OME reader
stack after the interface reports ready. An application-owned 500 ms timer
starts one dedicated serial thread for `ome_types`, `ome_zarr.reader` and
`ome_zarr.writer`. No reader constructors, source files, Qt widgets, plugin
discovery for napari, optional JVM readers or accelerator probes run in that worker.

Normal file requests share Python's import cache and module locks rather than
waiting on a separate warm-up gate. The direct reader imports remain the
fallback after a failed warm-up. A small TIFF selected while `ome_types` is
pending uses the asynchronous source path to keep the GUI off that module lock.
Both standalone launch and the napari loading host schedule once, after their
existing readiness boundary. Closing the app cancels imports still queued;
an import already executing finishes on its non-daemon thread before Python
exits. The interface never joins the worker. This does not promise zero initial
file latency if a file is selected before its library has finished warming.

Focused tests cover import-free construction, one-shot background ownership,
Qt heartbeat activity, concurrent real imports sharing one module execution,
failure logging and normal retry, cancel-before-start, and TIFF worker routing.
The earlier full-suite result below applies to the deferred-import baseline;
follow-up validation is recorded separately.

Native GUI smoke measurements in the installed CUDA environment (current source,
uncontrolled warm OS caches) recorded the following. Both OME libraries were
absent at the ready event; background loading started afterward. A 20 ms Qt
heartbeat remained active. The file was a newly written 8 × 8 uint16 OME-TIFF;
every read matched its exact expected pixel values.

| Native check | UI ready (s) | Background imports (s) | First TIFF read (s) |
| --- | ---: | ---: | ---: |
| Warm-up, first measured | 42.061 | 9.363 | 0.063 |
| No warm-up | 13.231 | — | 0.846 |
| Warm-up, repeat | 12.825 | 2.465 | 0.027 |

The repeat warm-up completed about 2.994 s after readiness, including its initial
500 ms delay. It processed 140 heartbeat events during that interval, with a
maximum observed inter-tick gap of 118 ms. The first measured warm-up processed
457 events, with a 194 ms maximum gap. These are smoke observations rather than
a UI latency bound or a randomized performance trial. In particular, the first
launch's 42 s total demonstrates continuing host/cache variability, not work
performed before readiness by the reader worker. No import work was moved
back onto the startup path. Large file decoding and other reader first-use work
are not covered by the tiny TIFF timing.

Follow-up validation completed across three independent test processes covering
all 446 test files and 13,373 collected cases: **12,896 passed, 474 skipped,
three failed**. The groups reported 1,004/5,649/6,243 passes respectively; the
new warm-up and startup lifecycle checks passed, as did manifest validation,
repository lint with scratch trees excluded, package builds, manual checks and
the strict manual build. The two updated manual pages and help link were
rechecked in light/dark themes at desktop and 390 px widths. A native
smoke-exit launch closed cleanly after starting the delayed warm-up.

The failures were:

- `test_widget.py::test_batch_parameter_override_preview_uses_detached_item_workflow`
  exceeded its 10 s preview wait during the full widget run. An isolated rerun
  passed (19.36 s including setup).
- Both parameterizations of
  `test_ui_inspector_sigma_filter_presentation.py::test_registration_help_restores_ordinary_form_before_queued_resize`
  reported a 106 px stacked label hint against 138 px of natural text. Isolated
  current-code reruns reproduced them. A control loading the original product
  modules from baseline commit `9f75a17`, with the same interpreter and source
  filenames, passed initially and reproduced both failures on repetition. A
  control retaining original eager I/O also reproduced them. This is evidence
  of a baseline layout issue/variability, not a clean full-suite qualification
  for the new code. No wrapping logic, batch behavior or test deadline was
  changed as part of reader warming.

`app.main()` now wraps either startup reporter with a standard-library-only
timing reporter. It writes flushed JSON lines prefixed `VIPP startup timing:`
to the existing child diagnostic log: stage, monotonic elapsed seconds,
process CPU seconds, preceding stage and preceding-stage seconds. It also
records failure timing. Status-channel authentication, protocol version,
milestone ordering and readiness semantics are unchanged. Pythonw without a
stderr stream still works, although it cannot retain these lines without the
launcher's redirected log.

The [profiling tool](../../scripts/profile_vipp_startup.py) can measure either
the installed environment or a checkout without modifying that environment:

```powershell
python scripts/profile_vipp_startup.py --python PATH_TO_VIPP_PYTHON --output-dir tmp/startup-cold --importtime
python scripts/profile_vipp_startup.py --python PATH_TO_VIPP_PYTHON --output-dir tmp/startup-warm --importtime
python scripts/profile_vipp_startup.py --python PATH_TO_VIPP_PYTHON --source-root src --output-dir tmp/startup-source
```

Use the first command as the first VIPP launch after a reboot, followed by the
second after the first exits. Keep power conditions the same and record AC or
battery operation. Output directories must be new; the tool refuses to overwrite
earlier logs. It retains authenticated, token-redacted startup events, per-stage
durations, timeout/exit information, a raw child log and optional importtime.
For old installed releases, stage durations use protocol wall timestamps; new
source builds use the monotonic timing records. Total process time includes
the normal 750 ms smoke-exit delay and Qt shutdown. The launcher splash and
Python work before the child app's first event are outside the stage totals.
This tool measures the standalone application child rather than launching a
second splash process.

If the next cold run still takes minutes, first identify its long stage and
then trace file I/O or the relevant GPU operation within that stage. Avoid
attributing the delay to a scanner or changing security policy without direct
evidence. Additional release opportunities are demand-driven sample creation
and finer import progress milestones; neither is implemented by this change.

## Validation

The focused startup/protocol, I/O, OME-Zarr preview, exact-window and Qt preview
suite passed: **131 tests**. A fresh-subprocess regression rejects attempts to
import ome-types, ome-zarr, Dask or Zarr through the public I/O and preview API.
Existing numerical/metadata/reader tests exercise deferred first-use loading.
Timing tests check deterministic durations, protocol forwarding, failure and
Pythonw behavior.

The complete CPU/offscreen suite finished with **12,891 passed, 474 skipped,
one failed** in 3,300.91 s. The single failure was
`test_widget.py::test_source_refresh_blocks_batch_until_representative_is_recalculated`:
its final batch-completion wait exceeded 15 s. An isolated rerun passed
(19.44 s including setup). This does not make the full run green or establish
the cause of the ordering/load-sensitive timeout. No batch behavior or test
timeout was changed to hide it. GPU hardware, platform and public-corpus tests
retain their stated skip conditions; the test environment used napari `0.9.2`
and had no CuPy, while startup profiling used the installed CUDA environment.

The retained profiling tool also completed a native source smoke launch with
exit code 0, readiness and all monotonic stage records. That run overlapped the
full CPU test suite and is a tool check rather than a performance comparison.
Plugin-manifest validation, package build, application documentation checks
(26 tests), companion manual content/routing checks and strict MkDocs build
passed. The two changed manual pages were inspected in light/dark themes at
desktop and 390 px widths, including the new help link. Repository lint passed
with the pre-existing untracked `tmp/` trees excluded; the literal `ruff check .`
also traverses those scratch/third-party trees and reports unrelated violations.

## Latest-main integration and installer follow-up

The application checkout was refreshed to `57248e0d8a7cc7c81fb0533fa8b705d4ddb370f7`
(`v0.16.0a5`), and the companion manual to `c7f4579`. The original work was
retained in local Git stashes before fast-forwarding both checkouts. Merge
resolution preserved the latest TIFF unit normalization, OME-Zarr writer
compatibility, standalone dock registration and installer registration guards.
The startup and installer changes remain unreleased after `0.16.0a5`.

The first native development installer, built before that refresh, passed its
package and CUDA acceptance checks but failed at final Desktop-shortcut
publication with Windows sharing violation 32. Its candidate was rolled back;
the old active installation and a recovery journal were retained. The retained
installation history contains the same Desktop-shortcut sharing failure on
August 25, August 27 and September 22, as well as October 21. An older August 10
failure was a package-download timeout. These records establish a repeated
shortcut failure; they do not identify the program holding the file. A later
Restart Manager query found no remaining holder. No security settings changed.

The new installer renders `.lnk` files inside its private run directory, copies
their bytes to same-volume `.tmp` files, and then publishes them atomically.
Shortcut publication, rollback and recovery tolerate brief Windows sharing
locks for about 30 seconds per operation. Ownership records, retained exact-build
wheels and cached setup publication use bounded short retries as well.
Non-retryable errors still stop immediately; ownership/hash checks, atomic
replacement and recovery reporting remain required. A persistent shortcut lock
has a specific plain-language message and still requires a fresh review before
another apply.

On the latest combined source, the focused startup/reader/installer suite passed
**446 tests with 2 skips**. The final expanded installer suite passed **356 tests
with 2 skips**, and source-widget checks passed **19 tests**. The four previously
troublesome inspector/batch checks passed in a separate latest-source run. These
overlapping focused suites are not a new complete-suite qualification. The
earlier complete-suite results above apply to the older source baseline.
Native checks exercised real Windows sharing violations during replacement and
cleanup, and actual PowerShell shortcut rendering in an isolated test folder.
Lint, manifest validation, package builds, 22 manual content contracts, 50 manual
routes and a strict manual build passed.

The local development installer embeds wheel SHA-256
`6a5e4a9c55e9a3f3f7e2a72eda50909907e1ce0d9f89b56faf667b0c1b1406e8`.
Its own SHA-256 is
`820226f63985db66d9262ac7603097add9973af30d43779629d97ad3f04c3ab0`.
All 297 runtime Python source files matched the packaged wheel, and the frozen
installer contained the shared file-lock module. This is an unsigned local
development artifact, not a published release.

## Latest installed build: normal Desktop-shortcut launches

The native installer completed successfully on 2026-10-21 at approximately
16:07:43 SAST, using its ordinary CUDA 13 update route. The retained result says
`succeeded`, with no registration warning, technical error or rollback. It also
recovered the previous failed transaction before presenting the new review.
The run identifier is `76c4a3d7-31bf-4ab9-beb6-303fa9924cc5`. The Desktop shortcut
now targets that managed environment's `vipp-app.exe --desktop`. All 297
installed runtime Python files matched the exact wheel embedded in the setup;
the ownership record's wheel hash matched as well. No setup recovery journal
remained. This is evidence for this successful update, not qualification of
every installer lifecycle or persistent-lock scenario.

An external standard-library observer opened the actual existing `VIPP.lnk`
with `os.startfile`, recorded the application's retained ready milestone, and
waited for the splash launcher process to exit. It did not alter the shortcut,
inject source, pass profiling flags, or request a smoke exit. Polling interval
was 50 ms; splash-process exit is a proxy for splash disappearance, rather than
a video measurement of the first usable frame. The final app was left open.
All runs used AC power (100% battery), the same installed build, and the default
Auto compute selection. Other desktop applications remained open. There was
no reboot or cache control, so these are ordinary installed-launch observations,
not a controlled cold/warm benchmark or proof of improvement over a2.

| Installed launch | Shortcut to ready (s) | Shortcut to splash-process exit (s) | Child CPU at ready (s) |
| --- | ---: | ---: | ---: |
| First after installation, 16:14 SAST | 141.663 | 142.159 | 18.109 |
| Intermediate repeat, recovery offered, 16:19 SAST | 9.619 | 10.067 | 7.875 |
| Repeat with normal initial workflow, 16:26 SAST | 158.245 | 158.799 | 18.609 |

The intermediate repeat is **not a comparable full-startup result**: it offered
recovery of the synthetic workflow from our first test and paused the normal
initial calculation. Both measured slow runs displayed the normal bundled
synthetic volume. Before the last run, only the two verified test-session
recovery directories were moved to a preserved scratch backup, with their
generation files and hashes inventoried. No original user workflows were moved
or deleted. A recovery offer after the first test was closed through napari's
normal Close dialog is a separate lifecycle observation that needs follow-up;
its precise cause was not established here.

| Child stage or launch overhead | First (s) | Normal-workflow repeat (s) |
| --- | ---: | ---: |
| Shortcut request to child timing origin, approximately | 12.804 | 21.695 |
| Import napari entry point | 3.676 | 3.591 |
| Create napari viewer, including its lazy imports | 61.142 | 60.764 |
| Import VIPP widget and dock registration | 40.440 | 41.129 |
| Build VIPP interface | 0.997 | 0.950 |
| Prepare/display initial workflow | 22.597 | 30.113 |
| Child ready to splash-process exit, approximately | 0.496 | 0.554 |

The approximate launch overhead is the externally observed ready time minus
the child's monotonic ready time and therefore includes observation latency.
The child totals were 128.859 s and 136.550 s, while process CPU time was only
18.109 s and 18.609 s. This demonstrates substantial time beyond that process's
CPU work. It does not by itself identify filesystem filters, storage latency,
scheduling, a synchronous external call or another cause. The repeat's worker
summary showed `Auto · 2 CPU`; the earlier first-run observation showed
`Auto · 1 GPU · 1 CPU`. The requested mode and sample were the same, but the
actual backend choice was not fixed, so the workflow-stage difference is not
a controlled GPU comparison. The roughly 102 s spent in viewer creation and
VIPP imports precedes those calculations in both runs.

Reader warming is scheduled after readiness, so it cannot account for those
pre-ready waits. The installed measurements show that deferring the OME reader
dependency tree has **not resolved the overall startup problem on this host**.
At this point, file/import wait attribution inside the viewer and widget-import
stages, plus launcher overhead, was the next diagnostic target. These stage
observations alone did not identify a scanner. The later system trace at the
top of this record supplies that evidence. A verified reboot test remains
outstanding, and no security-setting change was made.

The three raw observer records, original startup logs and test-recovery backup
remain under `tmp/startup-investigation-20261021` and the normal local VIPP log
folder. Their path-redacted timings and installation result are included in
the companion measurement JSON under `latest_installed_build`.

## Hardware and wait follow-up (before the system trace)

The user asked whether the CPU, SSD or package layout explains the difference
from the other computer. Local follow-up on the same installed build found:

- Windows identifies the disk as a SK hynix BC901 1 TB NVMe drive, reports its
  status as `OK`, and reports 65,948,893,184 free bytes on C: at the check. This
  is a coarse status query, not a complete SMART or hardware-health assessment.
- A bounded read-only probe opened an existing installed CUDA DLL with Windows
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` and read 128 MiB into an
  aligned buffer. File open took 0.000475 s, and the reads took 0.045165 s:
  **2,834 MiB/s** (about 3.0 GB/s). No source file was modified. Windows file
  caching was bypassed for these reads; controller caches were not cleared.
  This tests a small sequential-read workload, not the small-file latency,
  paging or concurrent load during the earlier slow launches.
- A fresh installed `pythonw.exe` process constructed a hidden napari viewer
  and imported the VIPP widget in **24.444 s with 21.922 s of process CPU**,
  including cProfile and per-operation instrumentation overhead. It performed
  9,453 import-path stat calls (5.318 s), 2,361 loader file reads (3.974 s),
  and 161 extension-creation calls (0.912 s). These are function aggregates;
  nested import times must not be added into an independent total. Only a
  hidden viewer was created: no workflow or recovery store was constructed.
- An independently profiled full application child, using the installed
  interpreter, status channel and normal initial synthetic calculation,
  reached ready in **38.689 s with 35.375 s of process CPU**. It used isolated
  diagnostic recovery/timing-history locations and closed normally after
  readiness. Fifteen `fsync` calls across that diagnostic session took only
  **0.021 s** in total. This rules against a consistently slow status-channel
  flush in this observed fast session, not an intermittent flush stall during
  an earlier run. cProfile adds overhead, so it is not an ordinary startup
  benchmark.
- A later launch through the **actual Desktop shortcut**, with a diagnostic
  import-timing hook, a safe Python-thread stack sampler, isolated recovery and
  timing-history locations, and the existing smoke-exit flag, completed the
  normal initial workflow. It reached ready in **15.763 s externally** and
  ended the splash process at **16.265 s**. The child recorded **14.188 s wall
  time and 13.266 s process CPU** at ready. Viewer creation took 5.428 s and
  VIPP widget imports took 2.552 s. This is an instrumented diagnostic with
  uncontrolled warm caches, not a replacement for the two ordinary-launch
  observations. It does demonstrate that the same CPU, disk and installed
  package tree can complete the normal startup work far faster.

Python bytecode caches were present in the installed package tree; the app
loads the extracted managed Python environment rather than unpacking the
development installer on every launch. Its many individual module files still
produce thousands of path checks and opens. That layout and broad dependency
imports make per-file overhead a plausible amplification mechanism. These
measurements do not prove that an alternative package container would remove
the multi-minute waits.

In the slow child runs, elapsed time exceeded process CPU time by approximately
111 s and 118 s. In the fast traced child, that difference was approximately
0.9 s. This argues against the CPU's fixed execution speed or the SSD's bulk
read throughput being the primary explanation for the variable delay. File
cache state, filesystem/security filtering, scheduling and other external
waits were the remaining possibilities at this stage. Registered security
products alone were not evidence of causation. The subsequent slow-launch trace
at the top of this record attributes substantial delay to Defender.

Windows Performance Recorder was available, but starting the FileIO/Minifilter
trace returned `0xc5585011`, "Failed to enable the policy to profile system
performance." A subsequent status check confirmed that no trace was recording.
No profiling policy, antivirus setting or other security configuration changed.
The user subsequently recorded an administrator-enabled trace; its analysis is
reported at the top of this record.

The first shortcut diagnostic used a native faulthandler watchdog and terminated
with access violation `0xC0000005` before readiness. That result is excluded;
its time until the failure splash was manually closed is not a startup time.
The subsequent successful diagnostic used an ordinary Python-thread sampler
instead. Only diagnostic instances were closed. The original open VIPP session
remained running, and neither the installed application nor the shortcut was
modified by these probes.

Raw evidence is retained in `installed-import-waits-a5-1`,
`exact-app-child-profile-a5-1`, `shortcut-wait-trace-a5-2`,
`installed-shortcut-traced-a5-2.json` and `read-only-storage-probe-a5.json`
beneath the existing scratch evidence directory. A path-redacted summary is
recorded under `latest_installed_build.hardware_wait_followup` in the JSON.
