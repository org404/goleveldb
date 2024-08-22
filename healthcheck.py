import os
import urllib.request

with urllib.request.urlopen(f"http://localhost:{os.environ['PORT']}/ok") as response:
    assert response.status == 200
