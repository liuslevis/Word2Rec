from flask import Flask,request,render_template,redirect,url_for
from flask.views import View
from cli import *

BASE_URL = '/word2rec'

app = Flask(__name__)

model = None
DEBUG = True
TRAIN_MODEL = False

if TRAIN_MODEL:
    prefs = read_prefs()
    model = train_model(prefs)
    save_model(model)
    model = load_model()
else:
    model = load_model()

def calc_redirect_url_likes_hates_removes(request, likes_str, hates_str):
    likes = []
    hates = []
    removes = []
    
    likes_param = request.args.get('like') 
    hates_param = request.args.get('hate')
    removes_param = request.args.get('remove')

    likes += likes_param.split(',') if likes_param else []
    hates += hates_param.split(',') if hates_param else []
    removes += removes_param.split(',') if removes_param else []

    likes += likes_str.split(',') if likes_str else []
    hates += hates_str.split(',') if hates_str else []

    likes = list(set(likes) - set(hates) - set(removes))
    hates = list(set(hates) - set(likes) - set(removes))

    url = ''
    if likes:
        url += '/like/' + ','.join(likes)
    if hates:
        url += '/hate/' + ','.join(hates)
    if likes_param or hates_param or removes_param:
        return url, likes, hates, removes
    else:
        return None, likes, hates, removes

@app.route(BASE_URL)
def index():
    return redirect(BASE_URL + '/vid/0')

# http://127.0.0.1:5000/vid/2000007/like/101,202/hate/303,404
@app.route(BASE_URL + '/vid/<vid>')
@app.route(BASE_URL + '/vid/<vid>/like/<likes_str>')
@app.route(BASE_URL + '/vid/<vid>/hate/<hates_str>')
@app.route(BASE_URL + '/vid/<vid>/like/<likes_str>/hate/<hates_str>')
@app.route(BASE_URL + '/vid/<vid>/hate/<hates_str>/like/<likes_str>')
def show_recommend(vid=None, likes_str=None, hates_str=None):
    url, likes, hates, removes = calc_redirect_url_likes_hates_removes(request, likes_str, hates_str)
    if url:
        return redirect(BASE_URL + '/vid/%s' % vid + url)

    recent_books = []
    if vid and vid != '0':
        recent_books = get_recent_books(vid)

    random_books = get_random_books(model)

    recommend_books = calc_recommend_books(model, likes, hates)

    return render_template('index.html', vid=vid, 
        recent_books=recent_books, 
        recommend_books=recommend_books,
        random_books=random_books,
        like_books=list(map(lambda x:get_book_info(x), likes)),
        hate_books=list(map(lambda x:get_book_info(x), hates)),
        base_url=BASE_URL
        )


if __name__ == '__main__':
    app.run(debug=DEBUG)

