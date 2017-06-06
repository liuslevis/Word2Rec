from flask import Flask,request,render_template,redirect,url_for
from flask.views import View
from cli import *

app = Flask(__name__)


def calc_redirect_url_likes_hates(likes_param, likes_str, hates_param, hates_str):
    likes = []
    hates = []
    likes += likes_param.split(',') if likes_param else []
    hates += hates_param.split(',') if hates_param else []
    likes += likes_str.split(',') if likes_str else []
    hates += hates_str.split(',') if hates_str else []
    url = ''
    if likes:
        url += '/like/' + ','.join(likes)
    if hates:
        url += '/hate/' + ','.join(hates)
    if likes_param or hates_param:
        return url, likes, hates
    else:
        return None, likes, hates

# http://127.0.0.1:5000/vid/2000007/like/101,202/hate/303,404
@app.route('/vid/<vid>')
@app.route('/vid/<vid>/like/<likes_str>')
@app.route('/vid/<vid>/hate/<hates_str>')
@app.route('/vid/<vid>/like/<likes_str>/hate/<hates_str>')
@app.route('/vid/<vid>/hate/<hates_str>/like/<likes_str>')
def show_recommend(vid=None, likes_str=None, hates_str=None):
    likes_param = request.args.get('like') 
    hates_param = request.args.get('hate')
    url, likes, hates = calc_redirect_url_likes_hates(likes_param, likes_str, hates_param, hates_str)
    if url:
        return redirect('vid/%s' % vid + url)

    model = load_model()
    
    recommend_books = []
    random_books = []
    recent_books = get_recent_books(vid)[:10]

    if vid and not likes:
        likes = list(map(lambda x:x['bookId'], recent_books))
    
    random_books = get_random_books(model)
    recommend_books = calc_recommend_books(model, likes, hates)

    return render_template('index.html', vid=vid, 
        recent_books=recent_books, 
        recommend_books=recommend_books,
        random_books=random_books,
        like_books=list(map(lambda x:get_book_info(x), likes)),
        hate_books=list(map(lambda x:get_book_info(x), hates)),
        )


if __name__ == '__main__':
    app.run(debug=True)