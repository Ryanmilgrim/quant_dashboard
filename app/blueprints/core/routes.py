from flask import Blueprint, render_template

core_bp = Blueprint("core", __name__)


@core_bp.route("/")
def home():
    return render_template("core/index.html")
