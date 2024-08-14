import json
import logging
import os
import pathlib
import shutil
from typing import List, Dict

import cf_xarray as cfxr
import harmony
import netCDF4
import numpy as np
import podaac.subsetter.subset
import unittest
import pytest
import requests
import xarray

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
    tokens = []
    headers: dict = {'Accept': 'application/json'}
    url: str = f"https://{'uat.' if env == 'uat' else ''}urs.earthdata.nasa.gov/api/users"

    # First just try to get a token that already exists
    try:
        resp = request_session.get(url + "/tokens", headers=headers,
                                   auth=HTTPBasicAuth(os.environ['CMR_USER'], os.environ['CMR_PASS']))
        response_content = json.loads(resp.content)

        for x in response_content:
            tokens.append(x['access_token'])

    except:  # noqa E722
        logging.warning("Error getting the token - check user name and password", exc_info=True)

    # No tokens exist, try to create one
    if not tokens:
        try:
            resp = request_session.post(url + "/token", headers=headers,
                                        auth=HTTPBasicAuth(os.environ['CMR_USER'], os.environ['CMR_PASS']))
            response_content: dict = json.loads(resp.content)
            tokens.append(response_content['access_token'])
        except:  # noqa E722
            logging.warning("Error getting the token - check user name and password", exc_info=True)

    # If still no token, then we can't do anything
    if not tokens:
        pytest.skip("Unable to get bearer token from EDL")

    return next(iter(tokens))


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


def verify_variables(merged_group, origin_group, subset_index, both_merged, file=None, variables_in_original=None):

    variables = origin_group.variables
    if variables_in_original:
        variables = variables_in_original

    for var in variables:
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


def verify_groups(merged_group, origin_group, subset_index, file=None, variables_in_original=None, both_merged=False):
    if file:
        logging.info("verifying groups ....." + file)

    verify_dims(merged_group, origin_group, both_merged)
    verify_attrs(merged_group, origin_group, both_merged)
    verify_variables(merged_group, origin_group, subset_index, both_merged, file=file, variables_in_original=variables_in_original)

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


@pytest.mark.timeout(600)
def test_concatenate(collection_concept_id, harmony_env, bearer_token):

    max_results = 2

    harmony_client = harmony.Client(env=harmony_env, token=bearer_token)
    collection = harmony.Collection(id=collection_concept_id)

    request = harmony.Request(
        collection=collection,
        concatenate=True,
        max_results=max_results,
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

    # Get original variables in both original files sometimes we might concatenate a Night and Day granule
    # which can have different variables
    variables_in_original = []
    for i, file in enumerate(original_files):
        origin_dataset = netCDF4.Dataset(file)

        variables = list(origin_dataset.variables)
        if len(variables_in_original) == 0:
            variables_in_original = variables
        else:
            for var in variables_in_original:
                if var not in variables:
                    variables_in_original.remove(var)

    for i, file in enumerate(original_files):
        origin_dataset = netCDF4.Dataset(file)
        verify_groups(merge_dataset, origin_dataset, i, file=file, variables_in_original=variables_in_original)
        origin_dataset.close()
