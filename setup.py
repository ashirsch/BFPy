from setuptools import setup


def readme():
	with open('README.md') as f:
		return f.read()


setup(
    name='kemitter',
    version='1.0.0a',
    description='A Python environment for wide-angle energy-momentum spectroscopy',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Alex Hirsch (Zia Lab)',
    author_email='alexander_hirsch@brown.edu',
    url='https://github.com/ashirsch/kemitter',
    license='LGPL',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3 :: Only'
    ],
    keywords='spectroscopy optics imaging data analysis',
    py_modules=['kemitter'],
    install_requires=[
        'cvxpy',
        'numba',
        'matplotlib',
        'spe2py',
        'scipy',
        'numpy'
    ]
)