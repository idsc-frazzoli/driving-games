FROM alezana/dg_base:3.10

# Install Driving Games
WORKDIR /driving_games
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN find .

ENV DISABLE_CONTRACTS=1
#ENV COLUMNS=130

RUN pipdeptree
RUN python setup.py develop --no-deps

RUN dg-demo --help

RUN crash-exp --help

CMD ["dg-demo"]
