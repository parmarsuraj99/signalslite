`pip install signalslite`

## Why:
- I wanted a pipeline that can generate features quickly so I can add, remove, build more features whenever needed. So it should be able to do everything from scratch in couple of hours. A relational DB would increase workload of setting things up. So I decided to use  parquet files split into daily structure. This is fast without any additional setup.

- Least friction to get started. It should effortlessly run on consumer grade laptops. Consequently, automate the whole pipeline on cloud, so makes sense to make it "lite", use **parallelization** when possible, allow for free data sources. It can utilize **cuda** if available, but is able to run on **cpu** as well.

- It should be able to run in Colab default runtime. One way to setup a pipeline is to save all data to mounted drive with more storage.

- Under 1000 LoC possible? Goal is not to build the best pipeline, but instead, a wrapper on top of flexible code that new users can easily understand and modify as needed.

## Stages:

1. Daily Data Collection/updation: 
    - Yahoo/EODHD (Thanks to https://github.com/degerhan/dsignals)
    - Save in daily parquet files
    - Update daily parquet files
    - Colab seem to be slow in loading data from yahoo. Will update.
2. Generate primary features:
    - Technicla indicators (RSI, MACD, SMA, EMA, etc on various timeframes)
    - flexible enough to accomodate fundamental data and news vectors data since things are independent of each other
3. Secondary features:
    - Generate features from primary features
    - like crossovers, ratios between technical features, etc
4. Scaling:
    - bringing the cross sectional features to same scale [0, 1]
    - Now data looks similar to Numerai classic data
5. Targets:
    - Generate your own targets for trading strategies
    - or use Numerai Signals targets
6. Modelling:
    - your best models in Numerai classic should work here
7. Scheduling:
    - Run the pipeline daily
    - should be able to run on cloud

## Notes:
- This is a work in progress. I will keep adding more features and examples. 
- more tests,
- more documentation,
- more examples,
- more flexibility,
- more speed,
- more parallelization,
- more cloud support,
- more data sources,
- more targets,
- more models,
- more everything

Hope you like it and find it useful. Please let me know if you have any suggestions or feedback. Thanks!