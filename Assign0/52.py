import json
import ctypes
from dask.distributed import Client
import dask.dataframe as dd
import warnings
import time
warnings.simplefilter('ignore')

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def PA0(path_to_user_reviews_csv):
    client = Client()
    # Helps fix any memory leaks.
    client.run(trim_memory)
    client = client.restart()
    dashboard_port = client.scheduler_info()['services']['dashboard']
    print('http://localhost:{}'.format(dashboard_port))

    data =dd.read_csv(path_to_user_reviews_csv)

    helpful_df = data['helpful'].str.extract(r'\[(\d+), (\d+)\]').astype(int)
    data['helpful_votes'], data['total_votes'] = helpful_df[0], helpful_df[1]

    data['reviewTime'] = data['reviewTime'].astype(str)
    data['Year'] = data['reviewTime'].str.split(', ').str.get(1).str.strip().astype(int)

    merge_df = data.groupby('reviewerID', sort=False).agg({
        'asin': 'count',
        'overall': 'mean',
        'Year': 'min',
        'helpful_votes': 'sum',
        'total_votes': 'sum',
    }, split_out=10)

    merge_df.columns = ['number_products_rated',
                        'avg_ratings',
                        'reviewing_since',
                        'helpful_votes',
                        'total_votes']
    
    
    submit = merge_df.describe().compute().round(2)

    with open('results_PA0.json', 'w') as outfile:
        json.dump(json.loads(submit.to_json()), outfile)