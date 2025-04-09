FROM python:3.12

WORKDIR /
COPY . /
# RUN  sudo apt update && sudo apt install llvm-dev
# RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install numba
RUN pip install geopy


CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]