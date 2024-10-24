from app import app

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test", methods=['GET'])
def jsonapi():
    return {
      "one": 1,
      "two": 2,
    }
