FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY Python_Model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Python_Model/ .

# Varsayilan: test_verileri/hastalikli_ornekler icindeki ornekleri analiz et,
# sonuclari /app/sonuclar klasorune yaz (volume ile disari alinabilir).
ENTRYPOINT ["python", "analiz.py"]
CMD ["--girdi", "/app/test_verileri/hastalikli_ornekler", "--cikti", "/app/sonuclar"]
