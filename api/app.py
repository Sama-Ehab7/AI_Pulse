from flask import Flask

print("🔥 Server file is running")

app = Flask(__name__)

@app.route("/")
def home():
    return "Server is working!"

if __name__ == "__main__":
    print("🚀 Starting Flask...")
    app.run(debug=True)