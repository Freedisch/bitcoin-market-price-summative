# Bitcoin Price Prediction

## Project Overview
This project develops a machine learning pipeline for predicting Bitcoin prices using a Multi-layer Perceptron (MLP) model. The project encompasses the entire workflow including data preprocessing, model training, model evaluation, and deployment using Flask API on a cloud platform.


## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
What things you need to install the software and how to install them:
```bash
python==3.8
numpy
pandas
tensorflow
keras
flask
scikit-learn
matplotlib
```

### Installing
A step-by-step series of examples that tell you how to get a development environment running.

1. Clone the repository:
   ```bash
   git clone https://github.com/freedisch/bitcoin-market-price-summative.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the project directory:
   ```bash
   cd bitcoin-market-price-summative
   ```

## Running the Notebook
Explain how to run the Jupyter notebook for developing and testing the model:
```bash
jupyter notebook notebook/bitcoin-market-price-summative.ipynb
```

## Deployment
Additional notes on how to deploy this on a live system using Flask and Docker:
1. Build the Docker container:
   ```bash
   docker build -t bitcoin-prediction .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 bitcoin-prediction
   ```

3. Access the API at:
   ```plaintext
   http://localhost:5000/predict
   ```

## Built With
* [Flask](http://flask.palletsprojects.com/) - The web framework used for the API
* [TensorFlow](https://www.tensorflow.org/) - ML framework for building the neural network
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis

## Contributing
Please read [CONTRIBUTING.md](https://github.com/freedisch/bitcoin-market-price-summative/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
* **Thibaut Batale** - *Initial work* - [Freedisch](https://github.com/freedisch)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
