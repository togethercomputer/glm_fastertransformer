FROM xzyaoi/glm_fastertransformer:0.0.3

RUN mkdir /app
ADD . /app

RUN pip install -r /app/glm/requirements.txt
CMD ["bash", "/app/serve.sh"]
EXPOSE 5000