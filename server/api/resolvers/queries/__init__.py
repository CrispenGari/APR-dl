
from ariadne import QueryType
from utils import Meta
from model import apr_model, apr_predictor

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
   try:
      preds = apr_predictor(
         input.get('text_review'), 
         input.get("text_review_upvote"),
         model=apr_model
      )
      return {
         "predictions":preds,
         "ok": True, 
      }
   except Exception as e:
      return {
         "ok": False,
         "error": {
            "field":"predictor",
            "message": str(e)
         }
      }