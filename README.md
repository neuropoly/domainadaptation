# Segmentation Domain Adaptation for MRI
Repository for the Domain Adaptation work using the self-ensembling
(mean teacher) for the domain adaptation of MRI images.

## Installing requirements
Requirements for this project:

* (required) Python 3.6 (use a virtual environment);
* (required) [Spinal Cord Toolbox](https://github.com/neuropoly/spinalcordtoolbox)(SCT)
* (required) [medicaltorch](https://github.com/perone/medicaltorch)
* (optional) FSLeyes/FSLview/FSLutils

These requirements are not included in the `setup.py` requirements, since
they aren't *pip-installable*, so you need to install them before
installing the project.

## Documentation
All the documentatio is in Sphinx. First, create a Python 3.6 environment and then do:

```
~# git clone https://github.com/neuropoly/domainadaptation.git
~# cd domainadaptation
~# pip install -e .
~# cd docs
~# make html
```

The output HTML will be generated inside the `docs/build/html` folder.
