To view cost estimates, explore
```
sanity_check.ipynb
```

To do load testing on deepinfa, run
```
uv run locust -f locustfile_deepinfra.py --host https://api.deepinfra.com --processes -1
```

to test phoenux (kvant) run
```
uv run locust -f locustfile_phoenix.py --host https://maas.ai-2.kvant.cloud --processes -1
```
press enter, and configure the number of "users". Each user sends as many requests as they can in sequence for as long as the test is running. Randomly samples from `sample_requests`.

The dream target for eyelevel is 850 requests per second completed. the number of users required to achieve that depends on the latency of the request.