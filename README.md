This is a simple python library developed by Francesco Paolo Liuzzi and Paul Maximilian Magos for the Machine Learning course 
of the University of Pisa AI Master's Degree.

---

First time you clone the repo:
```bash
> python3 -m venv batNNvenv
> source batNNvenv/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
```
Activate local env :
```bash 
> source batNNvenv/bin/activate
```
Build the library:
```bash
> python setup.py bdist_wheel
```
Install the library:
```bash
> pip install /dist/*.wheel
```
Once you have installed it you can call it like any other library: 
```{jupyter}
import batNNlib
from batNNlib import batNN
```
Notice that this installation, if performed this way, will install the library in the local env.
To install it in any other environment such as your global env, deactivate the local one and activate yours:
```bash
> deactivate

% conda activate myenv
or 
% pyenv activate myenv 
or etc...
```