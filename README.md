## Setup

```
brew install just
```

```
just setup
```

## Run

Run entire pipeline with:
```
dvc repr
```

Run individual sections of the pipeline with:
```
dvc repro {}
```

where `{}` is any stage in `dvc.yaml` 

