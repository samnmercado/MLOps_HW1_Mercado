from dagster import job, op
from repository import my_repository

@job
def stock_data_pipeline():
    # your pipeline steps here
    pass

# Execute the pipeline
result = stock_data_pipeline.execute_in_process()

# Check the result
if result.success:
    print("Pipeline ran successfully!")
else:
    print("Pipeline failed.")

