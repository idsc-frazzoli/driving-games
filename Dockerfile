FROM python:3.7

WORKDIR /cities
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup.py develop --no-deps
RUN zc-demo --help
CMD ["zc-demo"]
