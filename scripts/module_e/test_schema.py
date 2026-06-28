import requests
import json

URL = "https://api.platform.opentargets.org/api/v4/graphql"

q = """
query {
  __type(name: "StudyLocus") {
    name
    fields {
      name
      type {
        name
        kind
        ofType {
          name
          kind
        }
      }
    }
  }
}
"""
res = requests.post(URL, json={'query': q}).json()
print("StudyLocus Schema:")
print(json.dumps(res, indent=2))

q2 = """
query {
  __type(name: "L2GPrediction") {
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
res2 = requests.post(URL, json={'query': q2}).json()
print("\nL2GPrediction Schema:")
print(json.dumps(res2, indent=2))
