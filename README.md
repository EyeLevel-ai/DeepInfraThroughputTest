UV initialized. uses a local `.env` file.
```
uv run locust --processes -1 -f locustfile.py --host https://api.deepinfra.com
```
```
uv run locust --processes -1 -f locustfile_images.py --host https://api.deepinfra.com
```

test if things are properly configured with:
```
uv run python deepinfra_test.py
```

This also has `cost_estimate_analysis.py`, which does a grid search over several key parameters to profile cost and latency. This can be explored, after running, by running `viz_server.py`