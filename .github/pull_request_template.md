## Summary

<!-- What problem does this change solve, and how? -->

## Verification

<!-- List focused tests and any manual checks for the behavior changed here.
The exact-commit CI matrix owns the routine full-suite and package checks. -->

- [ ] `python -m npe2 validate src/napari_vipp/napari.yaml`
- [ ] `python -m ruff check .`
- [ ] Relevant tests pass

## Release Impact

<!-- Check only domains changed by this PR. The release operator takes the
union of these declarations for every PR merged since the prior public tag.
An unchecked domain may carry its existing qualification evidence forward.
Shared infrastructure can affect more than one domain. -->

- [ ] Core/UI behavior
- [ ] Workflow, schema, batch, or provenance
- [ ] GPU/scientific provider or shared compute infrastructure
- [ ] Windows installer, repair, update, rollback, or ownership
- [ ] Runtime dependencies or build toolchain
- [ ] Packaging or release automation
- [ ] Documentation or site content

<!-- Name any carried-forward baseline or write "None": -->

Carried-forward qualification:

## Scientific And User Impact

- [ ] User-facing behavior and limitations are documented
- [ ] Scientific references and validation evidence are included where needed
- [ ] Workflow/schema compatibility implications are described
- [ ] `CHANGELOG.md` is updated for a user-visible change

## Responsible Contribution

- [ ] No confidential, identifiable, embargoed, or ethics-restricted data is included
- [ ] Third-party code, data, and assets have compatible licenses and attribution
- [ ] Material generative-AI assistance is disclosed below and has been reviewed by a human contributor

<!-- Describe material AI assistance, or write "None". -->
