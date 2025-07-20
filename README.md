# Optimization-in-engineering

[![Python](https://img.shields.io/badge/python-3.12.9-blue.svg)](#)  
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-red.svg)](#)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#LICENSE)

This repository contains code and documentation for four optimization case studies from my Master’s thesis:

1. **Problem 1 (LP)**: Scaffolding support system  
2. **Problem 2 (NLP)**: Simply supported beam design  
3. **Problem 3 (NLP)**: Two-bar truss design  
4. **Problem 4 (MOO)**: Water‑tank column — minimize mass & maximize vibration frequency  

All problems are implemented in Python 3.12.9 and validated in MATLAB R2024b

## 📂 Repository Structure
Optimization-in-engineering/
├── README.md ← this file
├── requirements.txt ← Python dependencies
└── problems/
├── problem1_lp/ ← LP: Scaffolding
│ ├── description.md
│ ├── Lp_solution.py
│ └── output.csv
├── problem2_nlp_beam/ ← NLP: Beam design
│ ├── description.md
│ ├── solution.py
│ ├── solution.m
│ └── results/
│ ├── plot.png
│ └── output.csv
├── problem3_nlp_truss/ ← NLP: Truss design
│ ├── description.md
│ ├── solution.py
│ ├── solution.m
│ └── results/
│ ├── plot.png
│ └── output.csv
└── problem4_moo_column/ ← MOO: Column optimization
├── description.md
├── solution.py
├── solution.m
└── results/
├── pareto_front.png
└── pareto_data.csv
