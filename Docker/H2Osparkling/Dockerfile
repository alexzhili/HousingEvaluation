FROM spark-master:3.3

ENV MASTER "local[*]" 
ENV SPARK_HOME='/spark-3.3.0-bin-hadoop3' 

ADD sparkling-water.zip /sparkling-water.zip 

RUN unzip /sparkling-water.zip && \
    rm /sparkling-water.zip 

CMD /sparkling-water-3.38.0.2-1-3.3/bin/run-sparkling.sh