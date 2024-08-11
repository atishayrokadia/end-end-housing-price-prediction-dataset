## End to end data housing price prediction model


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/atishayrokadia/end-end-housing-price-prediction-dataset
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlProject python=3.8 -y
```

```bash
conda activate mlProject
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
```
Checking  parameter corelation with price
here are the results.
price               1.000000
area                0.535997
bedrooms            0.366494
bathrooms           0.517545
stories             0.420712
mainroad            0.296898
guestroom           0.255517
basement            0.187057
hotwaterheating     0.093073
airconditioning     0.452954
parking             0.384394
prefarea            0.329777
furnishingstatus   -0.304721
We have selected parameters with an absolute value greater than abs(0.3)
```
```bash
# Finally run the following command
python app.py
```

```bash
for host 
python webapp.py
```



