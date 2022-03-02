FROM alezana/dg_base:3.10

# Install Driving Games
WORKDIR /driving_games
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/idsc-frazzoli/dg-commons.git@dev-az014

#COPY requirements-extra.txt .
#RUN pip install -r requirements-extra.txt

COPY . .

RUN find .

ENV DISABLE_CONTRACTS=1

#RUN pipdeptree
RUN python setup.py develop --no-deps

RUN dg-demo --help

RUN crash-exp --help

CMD ["dg-demo"]
