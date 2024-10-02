import requests
import json
import argparse


def main():

    parser = argparse.ArgumentParser(description="Get Collection Names from CMR")
    parser.add_argument("--token", help="launchpad token")
    parser.add_argument("--file", help="file with list of concise associations")
    parser.add_argument("--env", help="CMR environment")

    args = parser.parse_args()

    url = None
    token = args.token

    if args.env == "uat":
        url = "https://graphql.uat.earthdata.nasa.gov/api"
    elif args.env == "ops":
        url = "https://graphql.earthdata.nasa.gov/api"

    headers = {
        "Content-Type": "application/json",
        "Authorization": token
    }

    # Get providers
    providers = []
    collections = []

    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            collections_list = json.loads(file_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Content of the file:\n{file_content}")  # Print the file content
        raise e
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

    for collection in collections_list:
        provider = collection.split('-')[1]
        if provider not in providers:
            providers.append(provider)

    for provider in providers:

        offset = 0
        more_data = True

        while more_data:
            
            try:

                # Define your GraphQL query
                graphql_query_template = """
                    query {{
                      collections(provider: "{provider}", limit: 2000, offset: {offset}) {{
                        items {{
                          conceptId
                          shortName
                        }}
                      }}
                    }}
                """

                graphql_query = graphql_query_template.format(provider=provider,offset=offset)

                # Create the request payload
                payload = {"query": graphql_query}

                # Make the GraphQL request with headers
                response = requests.post(url, headers=headers, json=payload)

                # Check the status code
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json().get('data').get('collections').get('items')

                    for item in data:
                        concept_id = item.get('conceptId')
                        if concept_id in collections_list:
                            collections.append({
                                'concept_id': concept_id,
                                'short_name': item.get("shortName")
                            })

                    if len(data) < 2000:
                        more_data = False
                    else:
                        offset += 2000
                else:
                    more_data = False
                    print(f"Error: {response.status_code}\n{response.text}")
            
            except Exception:
                more_data = False

    file_path = f"{args.env}_new_collections.json"

    with open(file_path, "w") as json_file:
        json.dump(collections, json_file, indent=4)

    print(json.dumps(collections))

if __name__ == "__main__":
    main()
