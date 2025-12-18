from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from app.blueprints.core import core_bp
    from app.blueprints.options import options_bp
    from app.blueprints.universe import universe_bp

    app.register_blueprint(core_bp)
    app.register_blueprint(universe_bp)
    app.register_blueprint(options_bp)

    return app
