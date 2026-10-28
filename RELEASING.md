# Releasing textblast

How a release is made, for maintainers. Users install with `pip install textblast`; see the README.

## Versions

- **0.9.x:** fixes only. Commands, configuration and output tables stay the same.
- **0.10, 0.11, …:** new features or changed defaults. The changelog says what changes for existing runs.
- **1.0:** when the outputs can be promised stable.

Every pull request that changes behaviour, output, installation or documentation adds a line under `## Unreleased`
at the top of [`CHANGELOG.md`](CHANGELOG.md).

## Once per repository (done for 0.9.0)

- **Zenodo:** the repository is switched on at zenodo.org (account → GitHub). Every published GitHub release becomes
  a new version under the concept DOI [10.5281/zenodo.23729155](https://doi.org/10.5281/zenodo.23729155). Zenodo
  takes its metadata from `.zenodo.json` (not from `CITATION.cff` when both exist).
- **Trusted publishing:** TestPyPI and PyPI each have a publisher for the workflow `wheels.yml` with the environment
  `testpypi` or `pypi`. On GitHub, both environments exist, `pypi` with a required reviewer, and the repository
  variables `PUBLISH_TESTPYPI` and `PUBLISH_PYPI` are `true`. Setting a variable to anything else switches its upload
  off.

## A release, step by step

1. **Prepare** on a branch `release-X.Y.Z`:
   - the version in `pyproject.toml`, `src/textblast/__init__.py`, `CITATION.cff` and `.zenodo.json`
     (`tests/test_citation.py` checks that they agree);
   - `## Unreleased` in `CHANGELOG.md` becomes `## X.Y.Z (YYYY-MM-DD)`, and a new empty `## Unreleased` goes above it;
   - the README and the guide checked against what changed.

   Open a pull request; merge it when CI and Wheels are green.
2. **Tag** the merge commit on `main`:
   ```
   git switch main
   git pull
   git tag -a vX.Y.Z -m "textblast X.Y.Z"
   git push origin vX.Y.Z
   ```
3. **The tag's Wheels run** builds the macOS and Linux wheels and the sdist, installs each wheel on clean runners
   (macOS 14 and 15, Ubuntu 22.04 and 24.04) for the install proof, uploads to TestPyPI, and then waits for approval
   before PyPI. If a macOS job is cancelled because no runner picked it up, re-run the failed jobs
   (`gh run rerun <run id> --failed`); that is GitHub's capacity, not the code.
4. **Test** the TestPyPI package in a fresh virtualenv, with Python 3.10 or newer (macOS's own `python3` is 3.9, with
   which pip only answers that no matching distribution was found):
   ```
   python3.10 -m venv /tmp/tb-test
   /tmp/tb-test/bin/pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ textblast==X.Y.Z
   /tmp/tb-test/bin/textblast engine info
   /tmp/tb-test/bin/textblast engine check
   ```
   `engine info` must say `(found via bundled)` and `engine check` that all tests passed. From a clone, the
   quickstart (`textblast run --config examples/quickstart/run.yaml --run_dir /tmp/tb-test-run`) must find 23 hits
   between 18 document pairs.
5. **Approve PyPI:** on the run's page, *Review deployments* → `pypi` → *Approve and deploy*. PyPI never accepts the
   same version twice: if something is wrong, reject the deployment, fix it, and release X.Y.Z+1.
6. **GitHub release** from the tag, with the changelog entry as its notes:
   ```
   gh release create vX.Y.Z --verify-tag --title "textblast X.Y.Z" --notes-file notes.md
   ```
   Zenodo archives it within minutes; check the new version at the concept DOI.

## Between releases

CI runs every Monday on `main` as well, so that new releases of Polars, NumPy or Python that break textblast show
up before users meet them. A failure there gets an issue and a fix in the next patch release.
