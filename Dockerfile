FROM python:3.7

WORKDIR /driving_games
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup.py develop --no-deps
RUN dg-demo --help
CMD ["dg-demo"]
