import json
import logging
import os
from typing import List, Dict

import cf_xarray as cfxr
import harmony
import netCDF4
import numpy as np
import unittest
import pytest
import requests

from requests.auth import HTTPBasicAuth
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

import cmr

VALID_LATITUDE_VARIABLE_NAMES = ['lat', 'latitude']
VALID_LONGITUDE_VARIABLE_NAMES = ['lon', 'longitude']

assert cfxr, "cf_xarray adds extensions to xarray on import"
GROUP_DELIM = '__'

@pytest.fixture(scope="session")
def env(pytestconfig):
    return pytestconfig.getoption("env")


@pytest.fixture(scope="session")
def cmr_mode(env):
    if env == 'uat':
        return cmr.CMR_UAT
    else:
        return cmr.CMR_OPS


@pytest.fixture(scope="session")
def harmony_env(env):
    if env == 'uat':
        return harmony.config.Environment.UAT
    else:
        return harmony.config.Environment.PROD


@pytest.fixture(scope="session")
def request_session():
    with requests.Session() as s:
        s.headers.update({'User-agent': 'l2ss-py-autotest'})
        yield s


@pytest.fixture(scope="session")
def bearer_token(env: str, request_session: requests.Session) -> str:
    url = f"https://{'uat.' if env == 'uat' else ''}urs.earthdata.nasa.gov/api/users/find_or_create_token"

    try:
        # Make the request with the Base64-encoded Authorization header
        resp = request_session.post(
            url,
            auth=HTTPBasicAuth(os.environ['CMR_USER'], os.environ['CMR_PASS'])
        )

        # Check for successful response
        if resp.status_code == 200:
            response_content = resp.json()
            return response_content.get('access_token')

    except Exception as e:
        logging.warning(f"Error getting the token (status code {resp.status_code}): {e}", exc_info=True)

    # Skip the test if no token is found
    pytest.skip("Unable to get bearer token from EDL")


def verify_dims(merged_group, origin_group, both_merged):
    for dim in origin_group.dimensions:
        if both_merged:
            unittest.TestCase().assertEqual(merged_group.dimensions[dim].size, origin_group.dimensions[dim].size)
        else:
            unittest.TestCase().assertGreaterEqual(merged_group.dimensions[dim].size, origin_group.dimensions[dim].size)


def verify_attrs(merged_obj, origin_obj, both_merged):
    ignore_attributes = [
        'request-bounding-box', 'request-bounding-box-description', 'PODAAC-dataset-shortname',
        'PODAAC-persistent-ID', 'time_coverage_end', 'time_coverage_start'
    ]

    merged_attrs = merged_obj.ncattrs()
    origin_attrs = origin_obj.ncattrs()

    for attr in origin_attrs:
        if attr in ignore_attributes:
            # Skip attributes which are present in the Java implementation,
            # but not (currently) present in the Python implementation
            continue

        if not both_merged and attr not in merged_attrs:
            # Skip attributes which are not present in both merged and origin.
            # This is normal operation as some attributes may be omited b/c
            # they're inconsistent between granules
            continue

        merged_attr = merged_obj.getncattr(attr)
        if both_merged and isinstance(merged_attr, int):
            # Skip integer values - the Java implementation seems to omit
            # these values due to its internal handling of all values as
            # Strings
            continue

        origin_attr = origin_obj.getncattr(attr)
        if isinstance(origin_attr, np.ndarray):
            unittest.TestCase().assertTrue(np.array_equal(merged_attr, origin_attr))
        else:
            if attr != "history_json":
                unittest.TestCase().assertEqual(merged_attr, origin_attr)


def verify_variables(merged_group, origin_group, subset_index, both_merged, file=None):
    for var in origin_group.variables:
        merged_var = merged_group.variables[var]
        origin_var = origin_group.variables[var]

        verify_attrs(merged_var, origin_var, both_merged)

        if both_merged:
            # both groups require subset indexes
            merged_data = merged_var[subset_index[0]]
            origin_data = origin_var[subset_index[1]]
        else:
            if len(origin_var.shape) == 2:
                row = origin_var.shape[0]
                col = origin_var.shape[1]
                merged_data = merged_var[subset_index][:row, :col]
            elif len(origin_var.shape) == 3:
                first = origin_var.shape[0]
                row = origin_var.shape[1]
                col = origin_var.shape[2]
                merged_data = merged_var[subset_index][:first,:row, :col]
            else:
                merged_data = np.resize(merged_var[subset_index], origin_var.shape)
            
            origin_data = origin_var

        equal_nan = True
        if merged_data.dtype.kind == 'S':
            equal_nan = False

        # verify variable data
        if isinstance(origin_data, str):
            unittest.TestCase().assertEqual(merged_data, origin_data)
        else:
            unittest.TestCase().assertTrue(np.array_equal(merged_data, origin_data, equal_nan=equal_nan))


def verify_groups(merged_group, origin_group, subset_index, file=None, both_merged=False):
    if file:
        logging.info("verifying groups ....." + file)

    verify_dims(merged_group, origin_group, both_merged)
    verify_attrs(merged_group, origin_group, both_merged)
    verify_variables(merged_group, origin_group, subset_index, both_merged, file=file)

    for child_group in origin_group.groups:
        merged_subgroup = merged_group[child_group]
        origin_subgroup = origin_group[child_group]
        verify_groups(merged_subgroup, origin_subgroup, subset_index, both_merged=both_merged)


@retry(retry=retry_if_exception_type(requests.RequestException), wait=wait_exponential(min=3, max=60), stop=stop_after_attempt(10))
def download_file(url, local_path, headers=None):
    try:
        with requests.get(url, stream=True, headers=headers) as response:
            response.raise_for_status()  # Check if the request was successful
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
            logging.info(f"File downloaded successfully: {local_path}")
    except requests.RequestException as e:
        logging.error(f"Failed to download the file. Exception: {e}")
        raise e  # Re-raise the exception to trigger the retry


def get_latest_granules(collection_concept_id, number_of_granules, env, token):
    
    cmr_url = f"https://cmr.{'uat.' if env == harmony.config.Environment.UAT else ''}earthdata.nasa.gov/search/granules.json"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "collection_concept_id": collection_concept_id,
        "sort_key": "-start_date",  # Sort by start_date in descending order
        "page_size": number_of_granules  # Retrieve the latest 'x' granules
    }
    
    # Make the request to CMR
    response = requests.get(cmr_url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        granules = response.json().get("feed", {}).get("entry", [])
        
        if granules:
            granule_ids = [granule["id"] for granule in granules]
            return granule_ids
        else:
            return None
    else:
        return None

@pytest.mark.timeout(600)
def test_concatenate(collection_concept_id, harmony_env, bearer_token):

    max_results = 2
    harmony_client = harmony.Client(env=harmony_env, token=bearer_token)
    collection = harmony.Collection(id=collection_concept_id)
    latest_granule_ids = get_latest_granules(collection_concept_id, max_results, harmony_env, bearer_token)

    if latest_granule_ids is None:
        if harmony_env == harmony.config.Environment.UAT:
            pytest.skip(f"No granules found for UAT collection {collection_concept_id}")
        raise Exception('Bad Request', 'Error: No matching granules found.')

    request = harmony.Request(
        collection=collection,
        concatenate=True,
        max_results=max_results,
        granule_id=latest_granule_ids,
        skip_preview=True,
        format="application/x-netcdf4",
    )

    request.is_valid()

    logging.info("Sending harmony request %s", harmony_client.request_as_url(request))

    try:
        job1_id = harmony_client.submit(request)
    except Exception as ex:
        if str(ex) == "('Bad Request', 'Error: No matching granules found.')":
            if harmony_env == harmony.config.Environment.UAT:
                pytest.skip(f"No granules found for UAT collection {collection_concept_id}")
        raise ex

    logging.info(f'\n{job1_id}')

    logging.info(harmony_client.status(job1_id))

    logging.info('\nWaiting for the job to finish')

    results = harmony_client.result_json(job1_id)

    logging.info('\nDownloading results:')

    futures = harmony_client.download_all(job1_id)
    file_names = [f.result() for f in futures]
    logging.info('\nDone downloading.')
    
    filename = file_names[0]

    # Handle time dimension and variables dropping
    merge_dataset = netCDF4.Dataset(filename, 'r')

    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    original_files = merge_dataset.variables['subset_files']
    history_json = json.loads(merge_dataset.history_json)
    assert len(original_files) == max_results

    for url in history_json[0].get("derived_from"):
        local_file_name = os.path.basename(url)
        download_file(url, local_file_name, headers)

    for i, file in enumerate(original_files):
        origin_dataset = netCDF4.Dataset(file)
        verify_groups(merge_dataset, origin_dataset, i, file=file)
