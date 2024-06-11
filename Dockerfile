FROM python:3.9
COPY . /senttrans
WORKDIR /senttrans
RUN pip install -r requirements.txt
CMD ["python", "main.py"]

# FROM python:3.9
# COPY . /senttrans
# WORKDIR /senttrans
# RUN pip install -r requirements.txt
# EXPOSE 5000
# CMD gunicorn -w 4 -b 0.0.0.0:5000 app:app
# # CMD ["python", "app.py"]


# docker build -t sent-trans-app .



