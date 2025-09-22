import pandas as pd
import requests
from tqdm import tqdm
import concurrent.futures
from urllib.parse import quote

# fetch metadata

def fetch_metadata(id, metadata_path):
    """
    Fetch metadata for an entry from InterPro.

    Args:
        id (str): ID (e.g., "PIRSF500444")
        metadata_path (str): Path to save the fetched metadata (e.g. pirsf)

    Returns:
        dict: Metadata dictionary containing details about the entry.
    """
    safe_id = quote(id)
    if metadata_path is "cathgene3d":
        safe_id = "G3DSA:" + safe_id
    url = f"https://www.ebi.ac.uk/interpro/api/entry/{metadata_path}/{safe_id}/"
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching {metadata_path} data for {safe_id}: HTTP {response.status_code}")
        return {}
    
    data = response.json()
    return data


def process_column_efficiently(
    df, 
    interpro_column='PIRSF', 
    max_workers=10, 
    batch_size=50,
    output_column='pirsf_info',
    metadata_path='pirsf'
):
    """
    Efficiently process InterPro column and fetch information for each entry in batches.
    """
    # Create copy
    result_df = df.copy()
    result_df[output_column] = [[] for _ in range(len(df))]

    # Get all unique IDs
    all_interpro_ids = set()
    for value in df[interpro_column].dropna():
        if isinstance(value, str):
            ids = [id.strip() for id in value.split(';') if id.strip()]
            all_interpro_ids.update(ids)

    all_interpro_ids = list(all_interpro_ids)
    print(f"Found {len(all_interpro_ids)} unique InterPro IDs to fetch")
    
    # call fetch_metadata with the first id to test if we get a valid response
    test_id = all_interpro_ids[0]
    test_data = fetch_metadata(test_id, metadata_path)
    if not test_data:
        print(f"Failed to fetch data for test ID {test_id}. Exiting.")
        return result_df
    else:
        print(f"Successfully fetched test data for {test_id}")
        print(f"Sample fetched data: {test_data}")

    # Cache for fetched results
    interpro_cache = {}

    def fetch_batch(interpro_ids_batch):
        """Fetch multiple InterPro entries in one go"""
        results = {}
        for interpro_id in interpro_ids_batch:
            try:
                data = fetch_metadata(interpro_id, metadata_path)
                if data:
                    results[interpro_id] = data
            except Exception as e:
                print(f"Error fetching {interpro_id}: {e}")
        return results

    # Break IDs into batches
    batches = [all_interpro_ids[i:i+batch_size] for i in range(0, len(all_interpro_ids), batch_size)]

    print(f"Fetching InterPro data in {len(batches)} batches...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, batch): tuple(batch) for batch in batches}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            batch_results = future.result()
            interpro_cache.update(batch_results)

    print(f"Successfully fetched {len(interpro_cache)} InterPro entries")

    # Populate DataFrame
    print("Processing rows...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        value = row[interpro_column]
        if pd.isna(value) or not isinstance(value, str):
            continue
        interpro_ids = [id.strip() for id in value.split(';') if id.strip()]
        row_interpro_info = [interpro_cache[id] for id in interpro_ids if id in interpro_cache]
        result_df.at[idx, output_column] = row_interpro_info
        
    # save to tsv.gz
    print("Saving to file...")
    result_df.to_csv(f"swissprot_filtered_with_alphafold_link_and_{metadata_path}.tsv.gz", sep="\t", index=False, compression='gzip')
    print("File saved.")

    return result_df

if __name__ == "__main__":
    # open up swissprot_filtered_with_alphafold_link.tsv.gz
    df = pd.read_csv("swissprot_filtered_with_alphafold_link.tsv.gz", sep="\t")
    print("Loaded dataframe")

    # filter to only rows with column_name
    interpro_column = "Gene3D"
    filtered_df = df[df[interpro_column].notna()]

    # run for Pfam
    process_column_efficiently(
        filtered_df,
        interpro_column=interpro_column,
        output_column='gene3d_info',
        metadata_path='cathgene3d'
    )
