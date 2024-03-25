import argparse
import requests
import json

def send_post_request(json_file):
    """
    Sends a POST request to the specified URL with the given JSON file.

    Parameters:
    - json_file (str): Path to the JSON file to be used in the request body.

    Returns:
    - Response object containing server's response.
    """

    url = "https://validation-server.onrender.com/api/upload/"
    headers = {
        "Content-Type": "application/json"
    }

    with open(json_file, 'r') as f:
        data = json.load(f)

    response = requests.post(url, headers=headers, json=data)
    
    return response

def main():
    """
    Main function that parses command-line arguments and sends a POST request.
    """

    parser = argparse.ArgumentParser(description="Send a POST request with a JSON file.")
    parser.add_argument("--f", required=True, help="Path to the JSON file to be sent with the request.")
    
    args = parser.parse_args()
    
    response = send_post_request(args.f)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")

if __name__ == "__main__":
    main()