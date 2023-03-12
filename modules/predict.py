import dill
import os
import pandas as pd
from datetime import datetime

def load_model():
    with open('data/models/cars_pipe_202303120909.pkl', 'rb') as file:
        model = dill.load(file)

    return model

def predict_all_data(model):
    car_id = []
    pred = []

    for root, dirs, files in os.walk("data/test"):
        for filename in files:
            test_data = pd.read_json('data/test/' + filename, typ='Series')
            df = pd.DataFrame(test_data).transpose()

            pred_test = model.predict(df)

            car_id.append(df.id[0])
            pred.append(pred_test[0])

    res_df = pd.DataFrame({'car_id': pd.Series(car_id), 'pred': pd.Series(pred)})

    return res_df

def save_pred(res_pred):
    res_pred.to_csv(f'data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

def predict():
    model = load_model()

    res_df = predict_all_data(model)

    save_pred(res_df)

    pass

if __name__ == '__main__':
    predict()
