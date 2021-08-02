FROM python:3.8

# Copy dependencies file
COPY dependencies-apt.txt /tmp/

# Install prerequisites for commonroad
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' /tmp/dependencies-apt.txt | uniq) \
    && rm -rf /var/lib/apt/lists/*

# Install commonroad
RUN git clone https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker.git
RUN cd commonroad-drivability-checker && bash build.sh -e /usr/local -v 3.8 --cgal --serializer -i -j 3 --no-root

# Install Driving Games
WORKDIR /driving_games
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN find .

ENV DISABLE_CONTRACTS=1
#ENV COLUMNS=130

RUN pipdeptree
RUN python setup.py develop --no-deps
RUN dg-demo --help
CMD ["dg-demo"]
