"""WSGI entrypoint for the quant dashboard Flask app."""

from quant_dashboard.web import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
