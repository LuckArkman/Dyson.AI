from setuptools import setup, find_packages

setup(
    name="zeroram-gen",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "lz4",
        "psutil",
        "requests",
        "langchain",
        "google-generativeai"
    ],
    author="Dyson.AI Team",
    description="Motor de inferência LLM baseado em Zero RAM e Disk-Based Operations.",
    python_requires=">=3.8",
)
