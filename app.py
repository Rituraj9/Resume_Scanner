import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle
import pandas as pd
import re
import nltk #For Stop Words i.e the,a,an
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to remove conjucation to make all words in present
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
from werkzeug.utils import secure_filename
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi

corpus = []

app = Flask(__name__)
UPLOAD_FOLDER = 'static//'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/resume')
def resume():
	return render_template('resume.html')

@app.route('/resume/resume_predict',methods=['POST'])
def pred():
	if request.method == 'POST':
		resume_df = pd.read_csv('UpdatedResumeDataSet.csv')
		for i in range(len(resume_df)):
			if resume_df['Category'][i]=='Data Science':
				resume_df['Category'][i]=0
			elif resume_df['Category'][i]=='HR':
				resume_df['Category'][i]=1
			elif resume_df['Category'][i]=='Advocate':
				resume_df['Category'][i]=2
			elif resume_df['Category'][i]=='Arts':
				resume_df['Category'][i]=3
			elif resume_df['Category'][i]=='Web Designing':
				resume_df['Category'][i]=4
			elif resume_df['Category'][i]=='Mechanical Engineer':
				resume_df['Category'][i]=5
			elif resume_df['Category'][i]=='Sales':
				resume_df['Category'][i]=6
			elif resume_df['Category'][i]=='Health and fitness':
				resume_df['Category'][i]=7
			elif resume_df['Category'][i]=='Civil Engineer':
				resume_df['Category'][i]=8
			elif resume_df['Category'][i]=='Java Developer':
				resume_df['Category'][i]=9
			elif resume_df['Category'][i]=='Business Analyst':
				resume_df['Category'][i]=10
			elif resume_df['Category'][i]=='SAP Developer':
				resume_df['Category'][i]=11
			elif resume_df['Category'][i]=='Automation Testing':
				resume_df['Category'][i]=12
			elif resume_df['Category'][i]=='Electrical Engineering':
				resume_df['Category'][i]=13
			elif resume_df['Category'][i]=='Operations Manager':
				resume_df['Category'][i]=14
			elif resume_df['Category'][i]=='Python Developer':
				resume_df['Category'][i]=15
			elif resume_df['Category'][i]=='DevOps Engineer':
				resume_df['Category'][i]=16
			elif resume_df['Category'][i]=='Network Security Engineer':
				resume_df['Category'][i]=17
			elif resume_df['Category'][i]=='PMO':
				resume_df['Category'][i]=18
			elif resume_df['Category'][i]=='Database':
				resume_df['Category'][i]=19
			elif resume_df['Category'][i]=='Hadoop':
				resume_df['Category'][i]=20
			elif resume_df['Category'][i]=='ETL Developer':
				resume_df['Category'][i]=21
			elif resume_df['Category'][i]=='DotNet Developer':
				resume_df['Category'][i]=22
			elif resume_df['Category'][i]=='Blockchain':
				resume_df['Category'][i]=23
			elif resume_df['Category'][i]=='Testing':
				resume_df['Category'][i]=24

		for i in range(0,len(resume_df)):
		    review = re.sub('[^a-zA-Z]',' ',resume_df['Resume'][i]) #cleaned all commaas,stops and all
		    review = review.lower() #convert all words to lowercase
		    review = review.split()
		    ps = PorterStemmer()
		    all_stopwords = stopwords.words('english')
		    all_stopwords.remove('not')
		    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
		    review = ' '.join(review)
		    corpus.append(review)

		from sklearn.feature_extraction.text import CountVectorizer
		cv = CountVectorizer(ngram_range=(1,1),max_features=1000) #take most frequent Words
		X = cv.fit_transform(corpus).toarray()
		y= resume_df['Category']
		import numpy as np
		y = np.array(y)
		y=y.astype('int')

		#from sklearn.model_selection import train_test_split

		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

		#from sklearn.naive_bayes import GaussianNB

		#nb = GaussianNB()

		#nb.fit(X_train,y_train)

		#pickle.dump(nb,open('sentiment_model.pkl','wb'))

		model=pickle.load(open('resume.pkl','rb'))
		#print(X_train.shape,y_train.shape)
		text = request.form['Resume']
		new_review = re.sub('[^a-zA-Z]', ' ', text)
		new_review = new_review.lower()
		new_review = new_review.split()
		ps = PorterStemmer()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
		new_review = ' '.join(new_review)
		new_corpus = [new_review]
		#print(new_corpus)
		#from sklearn.feature_extraction.text import CountVectorizer
		#cv = CountVectorizer(ngram_range=(1,1), min_df=1, vocabulary=model)
		#cv._validate_vocabulary()
		#print('loaded_vectorizer.get_feature_names(): {0}'.format(loaded_vectorizer.get_feature_names()))
		#from sklearn.feature_extraction.text import CountVectorizer
		#cv = CountVectorizer()
		new_X_test = cv.transform(new_corpus).toarray()
		#print(new_X_test)
		prediction = model.predict(new_X_test)
		#print(prediction)
		if prediction[0]==0:
			x='Data Science'
		elif prediction[0]==1:
			x='HR'
		elif prediction[0]==2:
			x='Advocate'
		elif prediction[0]==3:
			x='Arts'
		elif prediction[0]==4:
			x='Web Designing'
		elif prediction[0]==5:
			x='Mechanical Engineer'
		elif prediction[0]==6:
			x='Sales'
		elif prediction[0]==7:
			x='Health and fitness'
		elif prediction[0]==8:
			x='Civil Engineer'
		elif prediction[0]==9:
			x='Java Developer'
		elif prediction[0]==10:
			x='Business Analyst'
		elif prediction[0]==11:
			x='SAP Developer'
		elif prediction[0]==12:
			x='Automation Testing'
		elif prediction[0]==13:
			x='Electrical Engineering'
		elif prediction[0]==14:
			x='Operations Manager'
		elif prediction[0]==15:
			x='Python Developer'
		elif prediction[0]==16:
			x='DevOps Engineer'
		elif prediction[0]==17:
			x='Network Security Engineer'
		elif prediction[0]==18:
			x='PMO'
		elif prediction[0]==19:
			x='Database'
		elif prediction[0]==20:
			x='Hadoop'
		elif prediction[0]==21:
			x='ETL Developer'
		elif prediction[0]==22:
			x='DotNet Developer'
		elif prediction[0]==23:
			x='Blockchain'
		elif prediction[0]==24:
			x='Testing'

		x="Your Resume is predicted to be mostly towards "+x + "."
		return render_template('resume.html',prediction_text='{}'.format(x))

	#return render_template("home.html")

@app.route('/upload')
def upload_file():
   return render_template('resume_upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		tempdir = f.filename

		#pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR/tesseract.exe"
		pytesseract.pytesseract.tesseract_cmd = "./.apt/usr/bin/tesseract"
		#print(tempdir)
		#with open(tempdir, 'r+') as f:
		#	with WImage(file = f) as img:
		#		print('Opened large image')
		p = wi(filename=tempdir)
		#print(p)
		pdfImg = p.convert('jpeg')
		pdfImg.crop(width=100,height=100)
		#print(pdfImg)
		imgb=[]
		for img in pdfImg.sequence:
			page = wi(image=img)
			imgb.append(page.make_blob('jpeg'))

		#print(imgb)
		#print("hi")
		extract_t=[]
		for i in imgb:
			im = Image.open(io.BytesIO(i))
			text=pytesseract.image_to_string(im,lang='eng')
			extract_t.append(text)

		#print(extract_t[0])
		resume_df = pd.read_csv('UpdatedResumeDataSet.csv')
		for i in range(len(resume_df)):
			if resume_df['Category'][i]=='Data Science':
				resume_df['Category'][i]=0
			elif resume_df['Category'][i]=='HR':
				resume_df['Category'][i]=1
			elif resume_df['Category'][i]=='Advocate':
				resume_df['Category'][i]=2
			elif resume_df['Category'][i]=='Arts':
				resume_df['Category'][i]=3
			elif resume_df['Category'][i]=='Web Designing':
				resume_df['Category'][i]=4
			elif resume_df['Category'][i]=='Mechanical Engineer':
				resume_df['Category'][i]=5
			elif resume_df['Category'][i]=='Sales':
				resume_df['Category'][i]=6
			elif resume_df['Category'][i]=='Health and fitness':
				resume_df['Category'][i]=7
			elif resume_df['Category'][i]=='Civil Engineer':
				resume_df['Category'][i]=8
			elif resume_df['Category'][i]=='Java Developer':
				resume_df['Category'][i]=9
			elif resume_df['Category'][i]=='Business Analyst':
				resume_df['Category'][i]=10
			elif resume_df['Category'][i]=='SAP Developer':
				resume_df['Category'][i]=11
			elif resume_df['Category'][i]=='Automation Testing':
				resume_df['Category'][i]=12
			elif resume_df['Category'][i]=='Electrical Engineering':
				resume_df['Category'][i]=13
			elif resume_df['Category'][i]=='Operations Manager':
				resume_df['Category'][i]=14
			elif resume_df['Category'][i]=='Python Developer':
				resume_df['Category'][i]=15
			elif resume_df['Category'][i]=='DevOps Engineer':
				resume_df['Category'][i]=16
			elif resume_df['Category'][i]=='Network Security Engineer':
				resume_df['Category'][i]=17
			elif resume_df['Category'][i]=='PMO':
				resume_df['Category'][i]=18
			elif resume_df['Category'][i]=='Database':
				resume_df['Category'][i]=19
			elif resume_df['Category'][i]=='Hadoop':
				resume_df['Category'][i]=20
			elif resume_df['Category'][i]=='ETL Developer':
				resume_df['Category'][i]=21
			elif resume_df['Category'][i]=='DotNet Developer':
				resume_df['Category'][i]=22
			elif resume_df['Category'][i]=='Blockchain':
				resume_df['Category'][i]=23
			elif resume_df['Category'][i]=='Testing':
				resume_df['Category'][i]=24

		for i in range(0,len(resume_df)):
		    review = re.sub('[^a-zA-Z]',' ',resume_df['Resume'][i]) #cleaned all commaas,stops and all
		    review = review.lower() #convert all words to lowercase
		    review = review.split()
		    ps = PorterStemmer()
		    all_stopwords = stopwords.words('english')
		    all_stopwords.remove('not')
		    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
		    review = ' '.join(review)
		    corpus.append(review)

		from sklearn.feature_extraction.text import CountVectorizer
		cv = CountVectorizer(ngram_range=(1,1),max_features=1000) #take most frequent Words
		X = cv.fit_transform(corpus).toarray()
		y= resume_df['Category']
		import numpy as np
		y = np.array(y)
		y=y.astype('int')

		#from sklearn.model_selection import train_test_split

		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

		#from sklearn.naive_bayes import GaussianNB

		#nb = GaussianNB()

		#nb.fit(X_train,y_train)

		#pickle.dump(nb,open('sentiment_model.pkl','wb'))

		model=pickle.load(open('resume.pkl','rb'))
		m = re.sub('[^a-zA-Z]',' ',extract_t[0])
		m = m.lower()
		m = m.split()
		ps=PorterStemmer()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		m = [ps.stem(word) for word in m if not word in set(all_stopwords)]
		m = ' '.join(m)
		corpus_t=[m]
		#print(corpus_t)
		test = cv.transform(corpus_t).toarray()
		prediction = model.predict(test)
		print(prediction)

		if prediction[0]==0:
			x='Data Science'
		elif prediction[0]==1:
			x='HR'
		elif prediction[0]==2:
			x='Advocate'
		elif prediction[0]==3:
			x='Arts'
		elif prediction[0]==4:
			x='Web Designing'
		elif prediction[0]==5:
			x='Mechanical Engineer'
		elif prediction[0]==6:
			x='Sales'
		elif prediction[0]==7:
			x='Health and fitness'
		elif prediction[0]==8:
			x='Civil Engineer'
		elif prediction[0]==9:
			x='Java Developer'
		elif prediction[0]==10:
			x='Business Analyst'
		elif prediction[0]==11:
			x='SAP Developer'
		elif prediction[0]==12:
			x='Automation Testing'
		elif prediction[0]==13:
			x='Electrical Engineering'
		elif prediction[0]==14:
			x='Operations Manager'
		elif prediction[0]==15:
			x='Python Developer'
		elif prediction[0]==16:
			x='DevOps Engineer'
		elif prediction[0]==17:
			x='Network Security Engineer'
		elif prediction[0]==18:
			x='PMO'
		elif prediction[0]==19:
			x='Database'
		elif prediction[0]==20:
			x='Hadoop'
		elif prediction[0]==21:
			x='ETL Developer'
		elif prediction[0]==22:
			x='DotNet Developer'
		elif prediction[0]==23:
			x='Blockchain'
		elif prediction[0]==24:
			x='Testing'

		x="Your Resume is predicted to be mostly towards "+x + "."
		return render_template('resume_upload.html',prediction_text='{}'.format(x))
		#return 'file uploaded successfully'

if __name__ == "__main__":
	app.run(debug=True)