from setuptools import setup, find_packages

setup(
    name="quantum-digital-twin",
    version="0.2.0", # Updated version
    description="Layer-based calibration-driven digital twin for noisy quantum computers",
    author="Your Name", # Please replace
    author_email="sornbold@gmail.com", # Please replace
    url="https://github.com/sozoluffy/Toward_Quantum_Digital_Twins", # Please replace
    packages=find_packages(),
    install_requires=[
        "qiskit-terra>=0.24",
        "qiskit-aer>=0.12", # or qiskit_aer
        "numpy>=1.20",
        "scipy>=1.8",
        "PyYAML>=5.4",
    ],
    python_requires=">=3.8",
    # Optional: Add entry point if needed later
    # entry_points={
    #     "console_scripts": [
    #         "qdt-sim=simulator.simulate:main_cli_function", # Example
    #     ],
    # },
    classifiers=[ # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose a license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
