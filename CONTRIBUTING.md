# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `sequential_ft_transformer` for local development.

1. Download a copy of `sequential_ft_transformer` locally.
2. Install pipx.
  
  - For Ubuntu 23.04 or above:
      ```console
      $ sudo apt update
      $ sudo apt install pipx
      $ pipx ensurepath
      ```  
  
  - For Ubuntu 22.04 or below:
      ```console
      $ python3 -m pip install --user pipx
      $ python3 -m pipx ensurepath
      ```

3. Install the poetry package: 
    ```console
    $ pipx install poetry
    ```
4. Install `sequential_ft_transformer` using `poetry`:

    ```console
    $ poetry install
    ```

5. Use `git` (or similar) to create a branch for local development and make your changes:

    ```console
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

6. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.

7. Commit your changes and open a pull request.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.

## Code of Conduct

Please note that the `sequential_ft_transformer` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
