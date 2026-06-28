import requests
import json

URL = "https://api.platform.opentargets.org/api/v4/graphql"

q = """
query {
  __type(name: "CredibleSet") {
    name
    fields {
      name
      type {
        name
        kind
      }
    }
  }
}
"""
print("Introspecting...")
res = requests.post(URL, json={'query': q}).json()
print("CredibleSet Schema:")
print(json.dumps(res, indent=2))
