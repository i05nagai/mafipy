## Tests

```
cd /path/to/repository
pip install -r requirements.txt
pip install -r requirements-dev.txt
python setup.py test
```

## Benchmark
[airspeed\-velocity/asv](https://github.com/airspeed-velocity/asv) is used for benchmarking.
For detailed explanation, see [asv documentation](https://asv.readthedocs.io/en/latest/).

With virtualenv,

```
cd /path/to/repository
pip install -r requirements-dev.txt
python setup.py benchmark --run-type=NEW
```

After benchmarking, the following command generates html files to view the results in `benchmarks/asv_files/html`.

```
python setup.py benchmark_publish
```

To see this, run html server and open `http://127.0.0.1:8080/` with a web browser.

```
python setup.py benchmark_preview                                                                                                                                           [fix-build-failure] 1s
running benchmark_preview
executing 'asv preview'
· Serving at http://127.0.0.1:8080/
· Press ^C to abort
```
