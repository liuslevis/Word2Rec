from flask import Flask,request,render_template,redirect,url_for
from flask.views import View
from cli import *

app = Flask(__name__)

# http://127.0.0.1:5000/vid/2000007/like/101,202/hate/303,404
@app.route('/')
@app.route('/vid/<vid>')
@app.route('/vid/<vid>/like/<likes_str>')
@app.route('/vid/<vid>/hate/<hates_str>')
@app.route('/vid/<vid>/like/<likes_str>/hate/<hates_str>')
@app.route('/vid/<vid>/hate/<hates_str>/like/<likes_str>')
def show_recommend(vid=None, likes_str=None, hates_str=None):
    likes_param = request.form['like']
    hates_param = request.form['hate']
    print('debug 0', request.form)
    print('debug 1', hates_param)
    likes = likes_str.split(',') if likes_str else [] + likes_param.split(',') if likes_param else []
    hates = hates_str.split(',') if hates_str else [] + hates_param.split(',') if hates_param else []
    print('debug 2', hates_str)
    print('debug 3', hates)

    if likes_param and not hates_param:
        return redirect('/vid/%s/like/%s' % (vid, ','.join(likes)), code=302)
    elif not likes_param and hates_param:
        return redirect('/vid/%s/hate/%s' % (vid, ','.join(hates)), code=302)
    elif likes_param and hates_param:
        return redirect('/vid/%s/like/%s/hate/%s' % (vid, ','.join(likes), ','.join(hates)), code=302)


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