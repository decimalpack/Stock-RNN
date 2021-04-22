import base64
import json

from flask import Flask, render_template, request

from nn import main as predict

app = Flask(__name__)

stock_list = [
    "MARUTI.NS", "BRITANNIA.NS", "BHARTIARTL.NS", "GRASIM.NS", "COALINDIA.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "ITC.NS", "RELIANCE.NS",
    "TITAN.NS", "TCS.NS", "BAJAJFINSV.NS", "TATASTEEL.NS", "BAJAJFINANCE.NS",
    "BAJAJ-AUTO.NS", "NTPC.NS", "LT.NS", "HEROMOTOCO.NS", "ICICIBANK.NS",
    "SHREECEM.NS", "TECHM.NS", "TATACONSUM.NS", "ONGC.NS", "NESTLEIND.NS",
    "CIPLA.NS", "ULTRACEMCO.NS", "HINDALCO.NS", "MM.NS", "WIPRO.NS", "BTC-INR"
]


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", stock_list=stock_list, data={})
    stock_name = request.form["stock_name"]
    end_date = request.form["end_date"]
    data = json.loads(predict(stock_name, end_date))
    with open("static/plot.png", "rb") as image_file:
        plot_img = base64.b64encode(image_file.read()).decode()
    return render_template("index.html",
                           stock_list=stock_list,
                           plot_img_b64=plot_img,
                           data=data)
