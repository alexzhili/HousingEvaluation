FROM spark-master:3.3

ENV SPARK_HOME='/spark-3.3.0-bin-hadoop3' 
ENV SPARK_MASTER=spark://spark-master:7077
EXPOSE 8081

CMD /spark-3.3.0-bin-hadoop3/sbin/start-worker.sh $SPARK_MASTER && bash
