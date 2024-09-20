set virtual env

```virtualenv -p python3 py3env```


activate virtual env

```source py3env/bin/activate```

pip install
```pip install --upgrade -r requirements.txt```


refresh requirements.txt

```pip freeze > requirements.txt```


run the code


```python3 src/proposed_test2.py```


For CB-FastICA

```python3 src/Fast_ICA_catBoost_GPU.py```