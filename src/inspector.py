from flask import Flask
import pymongo
import sys
app = Flask(__name__)

@app.route("/")
def hello():
    db = pymongo.Connection().stacko
    body = "Hello World: %d!" % db.train.count()
    return body

@app.route("/stuff/")
def get_stuff():
    c = pymongo.Connection().stacko.sample
    body = '<html><body>%s</html></body>' % '<br />'.join(
        ['<a href="/train/%s>%s</a>'
         % (r['PostId'], r['Title']) for r in c.find()])
    return body

@app.route("/train/<int:post_id>")
def get_stuff(post_id):
    c = pymongo.Connection().stacko.sample
    body = '<html><body>%s</html></body>' % '<br />'.join([r['Title'] for r in c.find()])
    return body

@app.route("/quit")
def quit_web():
    sys.exit()

if __name__ == "__main__":
    app.run(port=4242)
