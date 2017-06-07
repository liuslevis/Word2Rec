from flask import Flask,request,render_template,redirect,url_for
from flask.views import View
from cli import *

model = load_model()
app = Flask(__name__)

# @param request {'like':'101,102', 'hate':'201,202', 'remove':'201'}
# @param like_str '103,104'
# @param hate_str '203,204'
# @return '/like/101,102,103,104/hate/202,203,204'
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

    url = ''
    if likes:
        url += '/like/' + ','.join(likes)
    if hates:
        url += '/hate/' + ','.join(hates)
    if likes_param or hates_param:
        return url, likes, hates, removes
    else:
        return None, likes, hates, removes

@app.route('/')
def index():
    return redirect('/vid/0')

# http://127.0.0.1:5000/vid/2000007/like/101,202/hate/303,404
@app.route('/vid/<vid>')
@app.route('/vid/<vid>/like/<likes_str>')
@app.route('/vid/<vid>/hate/<hates_str>')
@app.route('/vid/<vid>/like/<likes_str>/hate/<hates_str>')
@app.route('/vid/<vid>/hate/<hates_str>/like/<likes_str>')
def show_recommend(vid=None, likes_str=None, hates_str=None):
    url, likes, hates, removes = calc_redirect_url_likes_hates_removes(request, likes_str, hates_str)
    likes = list(set(likes) - set(hates) - set(removes))
    hates = list(set(hates) - set(likes) - set(removes))

    if url:
        return redirect('vid/%s' % vid + url)

    recent_books = []
    if vid and vid != '0':
        recent_books = get_recent_books(vid)[:10]
        if not likes:
            likes = list(map(lambda x:x['bookId'], recent_books))[:3]

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