`pip install signalslite`

Why?
- I wanted a pipeline that can generate features quickly so I can add, remove, build more features whenever needed. So it should be able to do everything from scratch in couple of hours. Myu initial choice for database was a relational database, could have been ideal. However, it would increase workload of setting things up and writing operation of all stages of data pipeline was slow, so I decided to use  parquet files split into daily structure. This is fast.
- I also wanted it to run on most users' systems, so It should effortlessly run on consumer grade laptops. Consequently, I wanted to automate the whole pipeline on cloud, so makes sense to make it lite, use parallelization when possible, allow for free data sources.
- It should be able to run in Colab default runtime. One way to setup a pipeline is to save all data to mounted drive with more storage.
- Under 1000 LoC possible?