# OpenCL Parallel Programming Assignments

This repository contains a collection of assignments and example projects completed as part of my parallel programming course focused on OpenCL. These projects demonstrate fundamental concepts in parallel computing using OpenCL, including data parallelism, memory management, kernel execution, and optimization for high-performance computing.

## Getting Started

To run these assignments, ensure you have the following:

- **OpenCL SDK**: Install an OpenCL SDK suitable for your GPU or other compatible device.
- **Compiler**: A C or C++ compiler compatible with OpenCL (e.g., GCC, Clang).

### Directory Setup

Place the following folders (found in `Dependencies/`) in the root directory alongside the `.cpp`, `.cl`, and `CMakeLists.txt` files:

- `common` – Contains helper functions and utilities for the projects.
- `OpenCL-ICD-Loader` – Contains the OpenCL ICD (Installable Client Driver) loader, which facilitates interaction with OpenCL drivers.

The directory should look like this:

```plaintext
├── common/                # Utility functions and helper files
├── OpenCL-ICD-Loader/     # OpenCL ICD loader library
├── your_project.cpp       # Main project source files
├── your_kernel.cl         # OpenCL kernel files
└── CMakeLists.txt         # Build configuration file