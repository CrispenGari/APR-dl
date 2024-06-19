import os
from flask import request, jsonify, make_response
from app import app
from ariadne import  load_schema_from_path, make_executable_schema, graphql_sync
from resolvers.queries import query

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

type_defs = load_schema_from_path("schema.graphql")
schema = make_executable_schema(
    type_defs, query
)
class Config:
    DEBUG = True
    PORT = 3001

@app.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "automatic product recommendation(apr)",
        "description": "given a product text review and review up-votes, we should be able to predict wether the product is recommended by other customers or nor and also predict number of star rattings.",
        "language": "python",
        "libraries": ["tensorflow", "keras"],
        "visit": "http://localhost:3001/graphql"
    }
    return make_response(jsonify(meta)), 200

# @app.route("/graphql", methods=["GET"], )
# def graphql_playground():
#     return PLAYGROUND_HTML, 200

@app.route("/graphql", methods=["POST"])
def graphql_server():
    data = request.get_json()
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=True
    )
    return jsonify(result), 200 if success else 400

if __name__ == '__main__':
    app.run(debug=Config().DEBUG, port=Config().PORT)