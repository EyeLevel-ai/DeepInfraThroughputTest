UV initialized. uses a local `.env` file.
```
uv run locust --processes -1 -f locustfile.py --host https://api.deepinfra.com
```

test if things are properly configured with:
```
uv run python deepinfra_test.py
```