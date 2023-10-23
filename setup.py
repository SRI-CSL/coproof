import setuptools
from glob import glob

setuptools.setup(name="CoProver",
                 version="1.0",
                 description="Tools for automated theorem proving, for DARPA PEARLS",
                 author="Shankar Natarajan",
                 author_email="shankar.natarajan@sri.com",
                 entry_points={
                     "console_scripts": [
                         "cmdpred_server = coprover.cmdpred.app:start_service",
                         "lemmaret_server = coprover.lemmaret.server.app:start_service"              
                         ]
                     },
                     install_requires=['torch', 
                                       'torchvision',
                                       'matplotlib', 'tqdm',
                                       'transformers[torch]',
                                       'sentence-transformers',
                                       'scikit-learn',
                                       'pandas',
                                       'tensorboard',
                                       'Pillow',
                                       'simplet5',
                                       'flask'],
                                       packages=['coprover'],
                                       package_dir={'':'src'},
                                       )
