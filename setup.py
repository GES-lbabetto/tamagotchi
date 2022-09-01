import setuptools


setuptools.setup(
    name="tamagotchi",
    version="0.1.0",
    description="",
    long_description="",
    packages=["tamagotchi"],
    package_data={
        "tamagotchi": ["data/*", "pages/*"],
    },
    entry_points={
        "console_scripts": [
            "tamagotchi=tamagotchi.tamagotchi:main",
        ]
    },
    install_requires=[],
)
