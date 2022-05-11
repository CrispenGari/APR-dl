### Automatic Product Recommender

⭐⭐⭐⭐⭐ Automatic Product Recommender **(APR)** is an Artificial Intelligence API for recommending products based on customer reviews. 👍👎

<img src="/images/cover.png" width="100%" alt="cover"/>
### Abstract

Automatic Product Recommender **(APR)** is a very useful topic in modern technologies. In this repository I present an **APR** graphql API served locally for recommending products based on other customer's reviews. APR is very useful in modern technology because it helps product consumers to chose weather they should buy the product or not based on previous experience customer reviews.

### Folder structure

The folder structure of the server looks as follows:

```
C:.
├───app
│   └───__pycache__
├───model
│   └───static
├───resolvers
│   ├───mutations
│   └───queries
│       └───__pycache__
└───utils
    └───__pycache__
```

### Getting started

In this section we are going to show how you can use the APR server to make predictions locally.

First you are required to have `python` installed on your computer to be more specific `python` version 3 and `git`

Then you need to clone this repository by running the following command:

```shell
git clone https://github.com/CrispenGari/APR-dl.git
```

And then you navigate to the server folder of this repository by running the following command:

```shell
cd APR-dl/server
```

Next you are going to create a virtual environment `venv` by running the following command:

```shell
virtualenv venv
```

Then you need to activate the virtual environment by running the following command:

```shell
.\venv\Scripts\activate.bat
```

After activating the virtual environment you need to install the required packages by running the following command:

```shell
pip install -r requirements.txt
```

Then you are ready to start the server. To start the server you are going to run the following command:

```shell
cd api && python app.py
```

The above command will start the local server at default port of `3001` you can be able to make request to the server.

### GraphQL

With only this you will be able to test the `API` using the graphql playground that is served at the following endpoint:

```py
http://127.0.0.1:3001/graphql

# Or
http://localhost:3001/graphql
```

Open the above url in the browser and start making predictions.

### Example Prediction

In this section I will show how to make predictions to the apr-graphql-server. Note that this is an advanced way of doing it (using fragments) but if you are familiar with graphql you know the easy way of doing it. Send the following graphql request:

```
fragment ErrorFragment on Error{
  field
  message
}
fragment RatingFragment on Rating{
  rating
  probability
  stars
}
fragment RecommendFragment on Recommend{
  label
  class_
  probability
  emoji
}
fragment PredictionsFregmant on Predictions{
	recommend{
    ...RecommendFragment
  }
  rating{
    ...RatingFragment
  }
}
query APRPredictor($input: APRInput!){
  predictor(input: $input){
    error{
      ...ErrorFragment
    }
    ok
    predictions{
      ...PredictionsFregmant
    }
  }
}
```

With the following graphql variables:

```json
{
  "input": {
    "text_review": "Just piping in here ordered my usual size of small petite the slip that came with the dress is about a size negative zero it could hardly squeeze over my body and the dress itself is a bright pale aqua and it is a shift and because of the smocking it very strangely i think it very cheap and is ill fitting i would say if you are a person on which shift look awesome you might like this but mind the size of the slip in the range and also it is aqua",
    "text_review_upvote": 19
  }
}
```

You will get the similar response as follows:

```json
{
  "data": {
    "predictor": {
      "error": null,
      "ok": true,
      "predictions": {
        "rating": {
          "probability": 0.4749999940395355,
          "rating": 3,
          "stars": "⭐⭐⭐"
        },
        "recommend": {
          "class_": "NOT RECOMMENDED",
          "emoji": "👎",
          "label": 0,
          "probability": 0.769
        }
      }
    }
  }
}
```

### Model

The model that is being used here was build using tensorflow and keras `Subclassing` API and the architecture looks as follows:

```py
class APR(keras.Model):
  def __init__(self):
    super(APR, self).__init__()
    # layers for bidirectional
    forward_layer = keras.layers.GRU(
      128, return_sequences=True, dropout=.5,
      name="gru_forward_layer"
    )
    backward_layer = keras.layers.LSTM(
      128, return_sequences=True, dropout=.5,
      go_backwards=True, name="lstm_backward_layer"
    )
    self.embedding = keras.layers.Embedding(
          vocab_size, 100,
          input_length=max_words,
          weights=[embedding_matrix],
          trainable=True,
          name = "embedding_layer"
    )
    self.bidirectional = keras.layers.Bidirectional(
        forward_layer,
        backward_layer = backward_layer,
        name= "bidirectional_layer"
    )
    self.gru_layer = keras.layers.GRU(
              512, return_sequences=True,
              dropout=.5,
              name= "gru_layer"
      )
    self.lstm_layer = keras.layers.LSTM(
              512, return_sequences=True,
              dropout=.5,
              name="lstm_layer"
    )
    self.fc_1 = keras.layers.Dense(512, activation="relu", name="upvote_fc1")
    self.pooling_layer = keras.layers.GlobalAveragePooling1D(
          name="average_pooling_layer"
    )
    self.concatenate_layer = keras.layers.Concatenate(name="concatenate_layer_layer")

    self.dense_1 = keras.layers.Dense(64, activation='relu', name="dense_1")
    self.dropout_1 = keras.layers.Dropout(rate= .5, name="dropout_layer_0")
    self.dense_2 = keras.layers.Dense(512, activation='relu', name="dense_2")
    self.dropout_2 =  keras.layers.Dropout(rate= .5, name="dropout_layer_1")
    self.dense_3 = keras.layers.Dense(128, activation='relu', name="dense_3")
    self.dropout_3 = keras.layers.Dropout(rate= .5, name="dropout_layer_2")
    self.rating_output = keras.layers.Dense(5, activation='softmax', name="rating_output")
    self.recommend_output = keras.layers.Dense(1, activation='sigmoid', name="recommend_output")

  def call(self, inputs):
    text, upvote = inputs
    # Leaning the text features
    x_1 = self.embedding(text)
    x_1 = self.bidirectional(x_1)
    x_1 = self.gru_layer(x_1)
    x_1 = self.lstm_layer(x_1)
    x_1 = self.pooling_layer(x_1)

    # Learning the upvotes
    x_2 = self.fc_1(upvote)

    # concatenation
    x = self.concatenate_layer([x_1, x_2])

    # leaning combinned features
    x = self.dense_1(self.dropout_1(x))
    x = self.dense_2(self.dropout_2(x))
    x = self.dense_3(self.dropout_3(x))

    # outputs
    rating = self.rating_output(x)
    recommend = self.recommend_output(x)
    return rating, recommend
```

This models takes in two inputs which are:

```
texts: String variable for the review body.
upvotes: Positive Integer documenting the number of other customers who found this review positive.
```

And it outputs two outputs which are:

```
recommended:
    👍👎
    Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
rattings: ⭐⭐⭐⭐⭐
    Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
```

### Notebooks

There are two notebooks that were used in this project the one for data preparation and the other one for model training.

1. [Data Preparation](/notebooks/00_E_Commerce_Clothing_Reviews_Data_Prep.ipynb)
2. [Model Training](</notebooks/01_Automatic_Product_Recommender_(APR).ipynb>)
