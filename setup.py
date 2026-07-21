from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="litellm_prompt_optimizer",
    version="0.5.0",
    description="Provider-agnostic prompt optimization pipeline powered by LiteLLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "litellm",
        "deepeval",
        "python-docx",
        "python-dotenv"
    ],
    entry_points={
        "console_scripts": [
            "prompt-optimizer = prompt_optimizer.cli:main",
        ],
    },
    python_requires=">=3.8",
)