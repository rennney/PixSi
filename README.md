 #+title: Signal Processing for Pixelated LarTPC
#+EXPORT_FILE_NAME: index.html
#+setupfile: docs/setup-rto.org

* Introduction

This package provides charge reconstruction of pixle-based LarTPC detectors

* Install

~PixSi~ installs in the "usual Python way".  Use of ~pip~ and a Python
virtual environment is recommended.

#+begin_example
python3 -m venv venv
source venv/bin/activiate
cd pixsi
pip install -e .
#+end_example

* Usage

The ~pixsi~ command line user interface provides online help:

#+begin_src shell :exports both :results output code :wrap example
pochoir
#+end_src

#+RESULTS:
#+begin_example
Usage: pixsi [OPTIONS] COMMAND [ARGS]...

  PixSi command line interface

Options:
  -s, --store PATH     File for primary data storage (input/output)
  -o, --outstore PATH  File for output (primary only input)
  --help               Show this message and exit.

Commands:
  run  Runs Toy Simulation + Reconstruction
#+end_example

