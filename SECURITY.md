# Security Policy

napari-vipp is a local desktop plugin that can read microscopy data, write
analysis outputs, load workflow JSON, and generate Python scripts. Security
reports are welcome, particularly for unsafe path handling, unexpected file
overwrites, untrusted workflow behavior, dependency vulnerabilities, or code
execution that is not clearly initiated by the user.

## Supported Versions

This project is in alpha development. Security fixes are made on the current
default branch and, when practical, in the latest published alpha. Older alpha
releases are not maintained.

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's
[private vulnerability reporting form](https://github.com/rensutheart/napari-vipp/security/advisories/new)
when it is available. If the form is unavailable, contact the maintainer
privately through the contact details on the
[maintainer's GitHub profile](https://github.com/rensutheart) and share only the
minimum detail needed to establish a private reporting channel.

Please include:

- the affected version or commit;
- operating system and Python version;
- prerequisites and a minimal reproduction;
- the expected and observed behavior;
- the potential impact; and
- any suggested mitigation, if known.

The maintainer will aim to acknowledge a complete report within seven days,
keep the reporter informed while it is assessed, and coordinate disclosure
after a fix or mitigation is available. These targets are not a service-level
guarantee for this volunteer-maintained project.

## Research Data

Do not include confidential, identifiable, embargoed, or ethics-restricted
microscopy data in a report. Reduce the case to synthetic data or a minimal
metadata-only example whenever possible.
