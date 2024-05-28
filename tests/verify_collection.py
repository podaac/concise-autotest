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


@pytest.fixture(scope="function")
def granule_json(collection_concept_id: str, cmr_mode: str, bearer_token: str, request_session) -> dict:
    '''
    This fixture defines the strategy used for picking a granule from a collection for testing

    Parameters
    ----------
    collection_concept_id
    cmr_mode
    bearer_token

    Returns
    -------
    umm_json for selected granule
    '''
    cmr_url = f"{cmr_mode}granules.umm_json?collection_concept_id={collection_concept_id}&sort_key=-start_date&page_size=1"

    response_json = request_session.get(cmr_url, headers={'Authorization': f'Bearer {bearer_token}'}).json()

    if 'items' in response_json and len(response_json['items']) > 0:
        return response_json['items'][0]
    elif cmr_mode == cmr.CMR_UAT:
        pytest.skip(f"No granules found for UAT collection {collection_concept_id}. CMR search used was {cmr_url}")
    elif cmr_mode == cmr.CMR_OPS:
        pytest.fail(f"No granules found for OPS collection {collection_concept_id}. CMR search used was {cmr_url}")


@pytest.fixture(scope="function")
def original_granule_localpath(granule_json: dict, tmp_path, bearer_token: str,
                               request_session: requests.Session) -> pathlib.Path:
    urls = granule_json['umm']['RelatedUrls']

    def download_file(url):
        local_filename = tmp_path.joinpath(f"{granule_json['meta']['concept-id']}_original_granule.nc")
        response = request_session.get(url, headers={'Authorization': f'Bearer {bearer_token}'}, stream=True)
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        return local_filename

    granule_url = None
    for x in urls:
        if x.get('Type') == "GET DATA" and x.get('Subtype') in [None, 'DIRECT DOWNLOAD'] and '.bin' not in x.get('URL'):
            granule_url = x.get('URL')

    if granule_url:
        return download_file(granule_url)
    else:
        pytest.skip(f"Unable to find download URL for {granule_json['meta']['concept-id']}")


@pytest.fixture(scope="function")
def collection_variables(cmr_mode, collection_concept_id, env, bearer_token):
    collection_query = cmr.queries.CollectionQuery(mode=cmr_mode)
    variable_query = cmr.queries.VariableQuery(mode=cmr_mode)

    collection_res = collection_query.concept_id(collection_concept_id).token(bearer_token).get()[0]
    collection_associations = collection_res.get("associations")
    variable_concept_ids = collection_associations.get("variables")

    if variable_concept_ids is None and env == 'uat':
        pytest.skip('There are no umm-v associated with this collection in UAT')

    variables = []
    for i in range(0, len(variable_concept_ids), 40):
        variables_items = variable_query \
            .concept_id(variable_concept_ids[i:i + 40]) \
            .token(bearer_token) \
            .format('umm_json') \
            .get_all()
        variables.extend(json.loads(variables_items[0]).get('items'))

    return variables


def get_bounding_box(granule_umm_json):
    # Find Bounding box for granule
    try:

        longitude_list = []
        latitude_list = []
        polygons = granule_umm_json['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry'].get(
            'GPolygons')
        lines = granule_umm_json['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry'].get('Lines')
        if polygons:
            for polygon in polygons:
                points = polygon['Boundary']['Points']
                for point in points:
                    longitude_list.append(point.get('Longitude'))
                    latitude_list.append(point.get('Latitude'))
                break
        elif lines:
            points = lines[0].get('Points')
            for point in points:
                longitude_list.append(point.get('Longitude'))
                latitude_list.append(point.get('Latitude'))

        if not longitude_list or not latitude_list:  # Check if either list is empty
            raise ValueError("Empty longitude or latitude list")

        north = max(latitude_list)
        south = min(latitude_list)
        west = min(longitude_list)
        east = max(longitude_list)

    except (KeyError, ValueError):

        bounding_box = granule_umm_json['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry'][
            'BoundingRectangles'][0]

        north = bounding_box.get('NorthBoundingCoordinate')
        south = bounding_box.get('SouthBoundingCoordinate')
        west = bounding_box.get('WestBoundingCoordinate')
        east = bounding_box.get('EastBoundingCoordinate')

    return north, south, east, west


def get_coordinate_vars_from_umm(collection_variable_list: List[Dict]):
    lon_var, lat_var, time_var = {}, {}, {}
    for var in collection_variable_list:
        if 'VariableSubType' in var['umm']:
            var_subtype = var.get('umm').get('VariableSubType')
            if var_subtype == "LONGITUDE":
                lon_var = var
            if var_subtype == "LATITUDE":
                lat_var = var
            if var_subtype == "TIME":
                time_var = var

    return lat_var, lon_var, time_var


def get_science_vars(collection_variable_list: List[Dict]):
    science_vars = []
    for var in collection_variable_list:
        if 'VariableType' in var['umm'] and 'SCIENCE_VARIABLE' == var['umm']['VariableType']:
            science_vars.append(var)
    return science_vars


def get_variable_name_from_umm_json(variable_umm_json) -> str:
    if 'umm' in variable_umm_json and 'Name' in variable_umm_json['umm']:
        name = variable_umm_json['umm']['Name']

        return "/".join(name.strip("/").split('/')[1:]) if '/' in name else name

    return ""


def create_smaller_bounding_box(east, west, north, south, scale_factor):
    """
    Create a smaller bounding box from the given east, west, north, and south values.

    Parameters:
    - east (float): Easternmost longitude.
    - west (float): Westernmost longitude.
    - north (float): Northernmost latitude.
    - south (float): Southernmost latitude.
    - scale_factor (float): Scale factor to determine the size of the smaller bounding box.

    Returns:
    - smaller_bounding_box (tuple): (east, west, north, south) of the smaller bounding box.
    """

    # Validate input
    if east <= west or north <= south:
        raise ValueError("Invalid input values for longitude or latitude.")

    # Calculate the center of the original bounding box
    center_lon = (east + west) / 2
    center_lat = (north + south) / 2

    # Calculate the new coordinates for the smaller bounding box
    smaller_east = (east - center_lon) * scale_factor + center_lon
    smaller_west = (west - center_lon) * scale_factor + center_lon
    smaller_north = (north - center_lat) * scale_factor + center_lat
    smaller_south = (south - center_lat) * scale_factor + center_lat

    return smaller_east, smaller_west, smaller_north, smaller_south


def get_lat_lon_var_names(dataset: xarray.Dataset, file_to_subset: str, collection_variable_list: List[Dict]):
    # Try getting it from UMM-Var first
    lat_var_json, lon_var_json, _ = get_coordinate_vars_from_umm(collection_variable_list)
    lat_var_name = get_variable_name_from_umm_json(lat_var_json)
    lon_var_name = get_variable_name_from_umm_json(lon_var_json)

    if lat_var_name and lon_var_name:
        return lat_var_name, lon_var_name

    logging.warning("Unable to find lat/lon vars in UMM-Var")

    # If that doesn't work, try using cf-xarray to infer lat/lon variable names
    try:
        latitude = [lat for lat in dataset.cf.coordinates['latitude']
                         if lat.lower() in VALID_LATITUDE_VARIABLE_NAMES][0]
        longitude = [lon for lon in dataset.cf.coordinates['longitude']
                         if lon.lower() in VALID_LONGITUDE_VARIABLE_NAMES][0]
        return latitude, longitude
    except:
        logging.warning("Unable to find lat/lon vars using cf_xarray")

    # If that still doesn't work, try using l2ss-py directly
    try:
        # file not able to be flattened unless locally downloaded
        shutil.copy(file_to_subset, 'my_copy_file.nc')
        nc_dataset = netCDF4.Dataset('my_copy_file.nc', mode='r+')
        # flatten the dataset
        nc_dataset_flattened = podaac.subsetter.group_handling.transform_grouped_dataset(nc_dataset, 'my_copy_file.nc')

        args = {
                'decode_coords': False,
                'mask_and_scale': False,
                'decode_times': False
                }
        
        with xarray.open_dataset(
            xarray.backends.NetCDF4DataStore(nc_dataset_flattened),
            **args
            ) as flat_dataset:
                # use l2ss-py to find lat and lon names
                lat_var_names, lon_var_names = podaac.subsetter.subset.compute_coordinate_variable_names(flat_dataset)

        os.remove('my_copy_file.nc')
        if lat_var_names and lon_var_names:
            lat_var_name = lat_var_names.split('__')[-1] if isinstance(lat_var_names, str) else lat_var_names[0].split('__')[-1]
            lon_var_name = lon_var_names.split('__')[-1] if isinstance(lon_var_names, str) else lon_var_names[0].split('__')[-1]
            return lat_var_name, lon_var_name
        
    except ValueError:
        logging.warning("Unable to find lat/lon vars using l2ss-py")

    # Still no dice, try using the 'units' variable attribute
    for coord_name, coord in dataset.coords.items():
        if 'units' not in coord.attrs:
            continue
        if coord.attrs['units'] == 'degrees_north' and lat_var_name is None:
            lat_var_name = coord_name
        if coord.attrs['units'] == 'degrees_east' and lon_var_name is None:
            lon_var_name = coord_name
    if lat_var_name and lon_var_name:
        return lat_var_name, lon_var_name
    else:
        logging.warning("Unable to find lat/lon vars using 'units' attribute")

    # Out of options, fail the test because we couldn't determine lat/lon variables
    pytest.fail(f"Unable to find latitude and longitude variables.")


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


def verify_variables(merged_group, origin_group, subset_index, both_merged):
    for var in origin_group.variables:
        merged_var = merged_group.variables[var]
        origin_var = origin_group.variables[var]

        verify_attrs(merged_var, origin_var, both_merged)

        if both_merged:
            # both groups require subset indexes
            merged_data = merged_var[subset_index[0]]
            origin_data = origin_var[subset_index[1]]
        else:
            # merged group requires a subset index
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
    verify_variables(merged_group, origin_group, subset_index, both_merged)

    for child_group in origin_group.groups:
        merged_subgroup = merged_group[child_group]
        origin_subgroup = origin_group[child_group]
        verify_groups(merged_subgroup, origin_subgroup, subset_index, both_merged=both_merged)


def download_file(url, local_path, headers):
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logging.info("Original File downloaded successfully. " + local_path)
    else:
        logging.info(f"Failed to download the file. Status code: {response.status_code}")


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

    job1_id = harmony_client.submit(request)

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
