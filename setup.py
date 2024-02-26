import setuptools

setuptools.setup(
    name="esat",
    version="2024.0.1",
    author="Deron Smith",
    description="The EPA's Environmental Source Apportionment Toolkit (ESAT) aims to provide all the functionality "
                "found in the EPA's PMF 5.0 GUI tool. PMF is a source appointment using matrix factorization "
                "application that uses a modified loss function, use of uncertainty inputs, output constraints and "
                "additional error/uncertainty calculations.",
    packages=["src"]
)
