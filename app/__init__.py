from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    from .blueprints.core.routes import core_bp
    from .blueprints.options.routes import options_bp
    from .blueprints.universe.routes import universe_bp

    app.register_blueprint(core_bp)
    app.register_blueprint(universe_bp)
    app.register_blueprint(options_bp)

    return app
