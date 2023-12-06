import setuptools

setuptools.setup(
    name="Non-Negative Matrix Factorization Python",
    version="0.0.1",
    author="Deron Smith",
    description="The Non-Negative Matrix Factorization Python (NMF-PY) Tool aims to provide all the functionality "
                "found in the EPA's PMF 5.0 GUI tool. PMF is an extension of NMF with a modified loss function, "
                "use of uncertainty inputs, output constraints and additional error/uncertainty calculations.",
    packages=["src"]
)
