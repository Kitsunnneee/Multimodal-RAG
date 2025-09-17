from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimodal-rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Multimodal RAG system with Google's Gemini and Vertex AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-rag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "google-cloud-aiplatform>=1.36.0",
        "langchain-core>=0.1.0",
        "langchain-google-vertexai>=0.0.10",
        "langchain-text-splitters>=0.0.1",
        "langchain-community>=0.0.10",
        "unstructured[all-docs]>=0.11.0",
        "pypdf>=3.17.0",
        "pydantic>=2.0.0",
        "lxml>=4.9.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "opencv-python>=4.7.0",
        "tiktoken>=0.4.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-rag=multimodal_rag.cli:main",
        ],
    },
)
