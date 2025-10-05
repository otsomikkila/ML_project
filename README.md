python venv ohje:

´´´
python3 -m venv venv
´´´
saattaa tarvita:

´´´
sudo apt install python3-venv
´´´

aktivoi ympäristö:

´´´
source venv/bin/activate
´´´

lataa pip paketit:
´´´
pip install numpy matplotlib opencv-python scikit-learn
´´´


edistys tähän mennessä:

naiivi knn tunnistus toimii, mutta tarkkuus vain 33%
pienentämällä tarkasteltavaa aluetta päästiin 36.6%

canny edge detection ja hog mukana:

PCA ei toiminut, teki asioista vaan hitaampia