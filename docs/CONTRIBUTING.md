# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Developer Installation

If something goes wrong at any point during installing the library please see how
[our CI/CD on GitHub Actions](.github/workflows/build-main.yml) installs and builds the
project as it will always be the most up-to-date.

## Get Started!

Ready to contribute? Here's how to set up `bioio` for local development.

1. Fork the `bioio` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/bioio.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd bioio/
    just install
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests with [just](https://github.com/casey/just):

    ```bash
    just build
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Just Commands

For development commands we use [just](https://github.com/casey/just).

```bash
just
```
```
Available recipes:
    build                    # run lint and then run tests
    clean                    # clean all build, python, and lint files
    default                  # list all available commands
    generate-docs            # generate Sphinx HTML documentation
    install                  # install with all deps
    lint                     # lint, format, and check all files
    release                  # release a new version
    serve-docs               # generate Sphinx HTML documentation and serve to browser
    tag-for-release version  # tag a new version
    test                     # run tests
    update-from-cookiecutter # update this repo using latest cookiecutter-py-package
```

## Deploying

A reminder for the maintainers on how to deploy.
1) Make sure all your changes are committed and merged into main.
2) Make sure branch is clean:
    ```bash
    git checkout main
    git stash
    git pull
    ```
3) Create tag and push new version to GitHub like so:
    ```bash
    just tag-for-release "vX.Y.Z"
    just release
    ```
4) Wait for a [GitHub Action](https://github.com/bioio-devs/bioio/actions) to automatically publish to [PyPI](https://pypi.org/project/bioio/)
5) [Create GitHub release](https://github.com/bioio-devs/bioio/releases/new) for the corresponding version created.

    6a) Select tag for version created

    6b) Ensure GitHub automatically generates releases notes ([click "Generate Release Notes"](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes))

    6c) Double check format is similar to previous releases

    6d) Publish release
