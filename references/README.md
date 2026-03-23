# References

This directory is the human-facing map for the current repo, not a paper or pitch folder.

Read in this order:

1. `runtime-flow.md`
   the end-to-end execution path from UI request to winner selection and artifact write-out
2. `backend-modules.md`
   what each backend module owns and who calls it
3. `frontend-modules.md`
   what the UI modules do and how they consume backend payloads

Scope notes:

- these docs describe the active `app/*`, `benchmark/*`, and `ui/*` implementation
- `__init__.py` files are intentionally omitted unless they do more than re-export symbols
- `external/*` is vendored code and is not the primary implementation path
