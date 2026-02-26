# Contributing to cuHadron

Thank you for your interest in contributing to cuHadron. We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the issue or feature
- Steps to reproduce (for bugs)
- Your environment: GPU model, CUDA version, OS, and target ARCH (e.g. `sm_80`, `sm_90`) if relevant

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure your code follows the existing style
5. Add appropriate copyright headers to new files
6. Sign your commits (see DCO below)
7. Submit a pull request

**Adding new tests:** See the [Adding New Tests](README.md#adding-new-tests) section in the README for how to add a test to an existing category or a new category (including `test_requirements.txt`, `run_tests.sh`, and README table updates).

## Developer Certificate of Origin (DCO)

We require that all contributions are signed off under the [Developer Certificate of Origin (DCO)](https://developercertificate.org/).

The DCO is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing.

### DCO Text

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Signing Your Commits

To sign off your commits, add a `Signed-off-by` line to your commit messages:

```
This is my commit message

Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically by using the `-s` flag when committing:

```bash
git commit -s -m "Your commit message"
```

### Configuring Git for Sign-off

Make sure your Git configuration has your name and email set:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Code Style

- For C/CUDA code: Follow the existing style in the codebase
- For shell scripts: Use proper indentation and comments
- Add Apache 2.0 copyright headers to all new source files

### Copyright Header Template

For new files, use the following header:

```c
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to cuHadron!
