# Installation

## From Source

```bash
git clone https://github.com/communitiesuk/aef-bng.git
cd aef-bng
uv sync --group dev
```

## From Git (Databricks)

```bash
%pip install git+https://github.com/communitiesuk/aef-bng.git
dbutils.library.restartPython()
```

## From Wheel (Databricks)

```bash
uv build --wheel --out-dir dist/
```

Then on Databricks:

```python
%pip install "/Workspace/path/to/aef_bng-*.whl[spark]"
dbutils.library.restartPython()
```

## Extras

|   Extra   |                                      Use case                                      |
|-----------|------------------------------------------------------------------------------------|
| `[spark]` | Databricks Connect using a local machine connecting to a remote Databricks cluster |
