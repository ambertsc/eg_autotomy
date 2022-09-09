# Adaptive autotomy in co-evolved soft body/policy agents

## Setup (Linux)

```
virtualenv my_env --python=python3.8
source my_env/bin/activate
```

### Evogym and dependencies

You'll need some developer tools if they're not already installed on your system.

```
sudo apt-get install xorg-dev libglu1-mesa-dev
```

```
git clone git@github.com:yunshengtian/GPyOpt.git
cd GPyOpt
pip install -e .
cd ../

git clone git@github.com:yunshengtian/neat-python.git
cd neat-python
pip install -e .
cd ../
```

```
git clone git@github.com:riveSunder/eg_autotomy.git
cd eg_autotomy
pip install -r requirements.txt
cd ../
```

```
git clone --recurse-submodules git@github.com:EvolutionGym/evogym.git
cd evogym
pip install -e .
cd ../
```

### eg_autotomy

```
cd eg_autotomy
pip install -e .
```

Run tests

```
python -m eg_tests.test_all
```
