import os

from Essentials.app import *  # noqa: F401,F403
from Essentials.app import app as flask_app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
