# Support

napari-vipp is alpha research software. Validate results on representative data
before scientific interpretation, publication, or operational use.

## Where To Start

1. Review the [documentation index](docs/README.md) and
   [user guide](docs/user-guide.md).
2. Check the [operator tips](docs/operator-tips.md) for execution and UI
   problems, and the [I/O guide](docs/io-user-guide.md) for file-format issues.
3. Search [existing issues](https://github.com/rensutheart/napari-vipp/issues)
   and [discussions](https://github.com/rensutheart/napari-vipp/discussions) for
   a known limitation, answer, or workaround.

Choose the route that matches the question:

| Route | Use it for |
| --- | --- |
| [GitHub Issues](https://github.com/rensutheart/napari-vipp/issues/new/choose) | Reproducible defects, incorrect documentation, and focused feature requests. |
| [GitHub Discussions](https://github.com/rensutheart/napari-vipp/discussions) | Installation help, workflow questions, ideas, and sharing reusable examples. |
| [image.sc](https://forum.image.sc/tag/napari) | Broader bioimage-analysis and method questions that benefit from the community beyond VIPP. Include `napari-vipp` and its version in the topic. |

When a discussion reveals a reproducible software defect, open or link a
focused issue so the fix can be tracked.

## What To Include

For a useful bug report, include:

- napari-vipp, napari, Python, and operating-system versions;
- the installation command or environment details;
- a minimal sequence of actions that reproduces the problem;
- the full traceback or exact error text;
- a workflow JSON when it can be shared safely; and
- synthetic or public sample data rather than restricted research data.

For GPU setup or fallback problems, first open **Compute setup and memory**, run
the check, and choose **Save privacy-redacted support report…**. Attach that JSON
file instead of copying the raw advanced diagnostic. From Windows PowerShell,
the same export is available with:

```powershell
vipp-compute-doctor --track cuda13 --support-bundle ".\vipp-compute-support.json"
```

The `--json` output is intended for local inspection and is not the redacted
support format.

Remove patient, participant, specimen, institution, file-path, and acquisition
identifiers before attaching logs, screenshots, metadata, or workflows.

## Scope

These public channels cannot provide emergency support, guarantee a response
time, validate an unpublished biological conclusion, or recover data that was
not backed up.

Report security concerns privately using [SECURITY.md](SECURITY.md).
