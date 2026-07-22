from pathlib import Path
from setuptools import setup, find_packages

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="litellm_prompt_optimizer",
    version="0.5.2",  # bumped -- PyPI won't let you re-upload 0.5.0
    description="Provider-agnostic prompt optimization pipeline powered by LiteLLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),

    # NEW: makes sure non-.py files listed in package_data get bundled
    # into the built wheel/sdist, not just .py source files.
    include_package_data=True,
    package_data={
        "prompt_optimizer": ["README.md"],
    },

    install_requires=[
        "litellm==1.91.4",  
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