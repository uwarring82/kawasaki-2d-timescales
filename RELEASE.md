# Release procedure

This project archives tagged releases to a persistent identifier (Zenodo DOI),
per the FAIR-Findability gate. The steps below mint a DOI for a tag.

## Pre-tag checklist (done for v1.0.0)

- [x] Version aligned: `pyproject.toml`, `src/kawasaki2d/__init__.py`,
      `CITATION.cff`, `codemeta.json`, `.zenodo.json` all at the release version.
- [x] `date-released` / `dateModified` / `datePublished` set to the release date.
- [x] `CHANGELOG.md` updated with the release section.
- [x] `.zenodo.json` present with metadata and licence.
- [x] DOI placeholder `10.5281/zenodo.XXXXXXX` in `CITATION.cff`, `codemeta.json`,
      `README.md` (replaced after Zenodo mints the real DOI).
- [x] CI green (`pytest`, 70 tests; Numba and pure-Python paths agree bitwise).
- [x] Working tree clean; all `results/` manifests `dirty: false`.

## Cutting the release

1. **Tag** (already cut locally for v1.0.0):
   ```bash
   git tag -a v1.0.0 -m "kawasaki2d v1.0.0 — first archival release"
   ```
2. **Push** the branch and tag to the remote (requires a configured remote):
   ```bash
   git push origin <branch> --follow-tags
   ```
3. **Zenodo** (one-time setup): enable the GitHub↔Zenodo integration for the
   repository, then create a **GitHub Release** from the `v1.0.0` tag. Zenodo
   captures the release and mints a DOI automatically.
4. **Wire the DOI back**: replace `10.5281/zenodo.XXXXXXX` with the minted DOI in
   `CITATION.cff`, `codemeta.json`, `.zenodo.json` (if used), and `README.md`,
   then commit as `docs: record Zenodo DOI for v1.0.0` (a new dated entry — never
   rewrite the tagged commit).

## Notes

- The DOI minting (steps 2–3) is an outward-facing, credentialed action performed
  by a maintainer; it is intentionally not automated here.
- Zenodo issues a **concept DOI** (all versions) and a **version DOI** (this
  release). Cite the concept DOI for "the software", the version DOI for exact
  reproducibility.
