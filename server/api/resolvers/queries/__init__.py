
from ariadne import QueryType
from utils import Meta


query = QueryType()

@query.field("meta")
def meta_resolver(obj, info):
   return Meta(
        programmer= "@crispengari",
        main ="automatic product recommendation(apr)",
        description = "given a product text review and review up-votes, we should be able to predict wether the product is recommended by other customers or nor and also predict number of star rattings.",
        language = "python",
        libraries=["tensorflow", "keras"],
   ).to_json()
   
   
@query.field('predictor')
def predictor_resolver(obj, info, input):
   return {"predictions":{
      "recommend": {
        "label": 1,
        "probability": .9,
        "class_": "RECOMMENDED",
        "emoji": "👍"
      },
      "rating":{
          "rating": 5,
          "stars": "⭐" * 5,
          "probability": .98
      }
   }, "ok": True, }