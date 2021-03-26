from flask_wtf import FlaskForm
from wtforms import SubmitField,TextAreaField

class SentimentForm(FlaskForm):
    Resume = TextAreaField('Resume',validators=[DataRequired()])
    submit = SubmitField('Predict')