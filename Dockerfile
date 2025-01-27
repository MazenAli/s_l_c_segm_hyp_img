FROM tensorflow/tensorflow:2.8.0

WORKDIR /app
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash"]
