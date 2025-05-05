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
        # Removed the conflicting 'qiskit-terra' dependency
        "qiskit>=1.0",       # Added qiskit >= 1.0 as the core requirement
        "qiskit_aer>=0.13",  # Use the new standalone Aer package and match requirements.txt
        "numpy>=1.20",
        "scipy>=1.8",
        "PyYAML>=5.4",
        # Add other runtime dependencies here if any
    ],
    python_requires=">=3.8",
    # Define optional dependencies for development/testing
    extras_require={
        "test": [
            "pytest>=7.0",
            "matplotlib>=3.5", # Added matplotlib here as it's used in examples and tests might need it
            # Add other test-specific dependencies here
        ],
    },
    # Optional: Add entry point if needed later
    entry_points={
        "console_scripts": [
             "qdt-sim = simulator.simulate:run_with_twin", # From entry_points.txt
        ],
    },
    classifiers=[ # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose a license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
