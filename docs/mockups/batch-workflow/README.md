# Batch workflow comparison mockup

Interactive design snapshot, preserved for review before implementing the new
batch workflow window. This is a mockup, not the application's batch-processing
implementation. File operations and processing results are simulated.

Open [rendered.html](rendered.html) in a browser to compare the current window
with the proposed design. The controls above the windows demonstrate themes,
window widths, workflow states, larger sample lists, and larger parameter tables.

The editable fragment is
[vipp-batch-workflow-comparison.html](vipp-batch-workflow-comparison.html).
`rendered.html` is the matching standalone browser snapshot.

## Interaction checks

From the `qa-runtime` directory, run:

```sh
npm ci
node check.mjs
```

These checks cover table editing and workflow-value inheritance, optional bulk
edits, filtering, column navigation, pagination visibility, file-reveal actions,
and simulated run controls. They do not replace visual or native-app testing.

The mockup includes illustrative parameters. In particular, the Binary Threshold
above/below choice is not yet implemented in the application; the follow-up is
recorded in [the planning document](../../planning.md).
