To view cost estimates, explore
```
sanity_check.ipynb
```

To do load testing, run
```
uv run locust -f locustfile.py --host https://api.deepinfra.com --processes -1
```
press enter, and configure the number of "users". Each user sends as many requests as they can in sequence for as long as the test is running. Randomly samples from `sample_requests`.

---
To serve as a baseline, one of our benchmark claims has approximately 6,381 chunks per claim