from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

models = joblib.load('sales_forecasting_arima_models.pkl')

df_inventory = pd.read_csv('inventory_data.csv')
df_inventory['item id'] = df_inventory['item name'].apply(lambda x: int(x.split()[1]))

@app.route('/')
def index():
    return redirect(url_for('cashier'))

@app.route('/cashier', methods=['GET', 'POST'])
def cashier():
    sold_items = []
    
    if request.method == 'POST':
        item_id = int(request.form['item_id'])
        quantity_sold = int(request.form['quantity_sold'])
        if item_id not in df_inventory['item id'].values:
            return render_template('cashier.html', 
                                   inventory=df_inventory, 
                                   sold_items=sold_items,
                                   error="Item ID not found.")
        
        current_stock = df_inventory.loc[df_inventory['item id'] == item_id, 'stock quantity'].values[0]
        if quantity_sold > current_stock:
            return render_template('cashier.html', 
                                   inventory=df_inventory, 
                                   sold_items=sold_items,
                                   error="Quantity sold exceeds available stock.")
        
        df_inventory.loc[df_inventory['item id'] == item_id, 'stock quantity'] -= quantity_sold
        
        df_inventory.to_csv('inventory_data.csv', index=False)
        
        item_details = df_inventory.loc[df_inventory['item id'] == item_id].iloc[0]
        sold_items.append({
            'item_id': item_id,
            'item_name': item_details['item name'],
            'quantity_sold': quantity_sold,
            'price': item_details['item price'],
            'total_price': quantity_sold * item_details['item price']
        })
        
        return render_template('cashier.html', 
                               inventory=df_inventory,
                               sold_items=sold_items)

    return render_template('cashier.html', inventory=df_inventory, sold_items=sold_items)


@app.route('/inventory')
def inventory():
    forecast_steps = 7
    forecast_results = []

    for item_id, model_fit in models.items():
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=datetime.now(), periods=forecast_steps, freq='D')
        forecast_df = pd.DataFrame({
            'item id': item_id,
            'ds': forecast_index,
            'forecasted_quantity': forecast.predicted_mean
        })
        forecast_results.append(forecast_df)

    forecast_all = pd.concat(forecast_results)

    forecast_merged = forecast_all.merge(df_inventory[['item id', 'stock quantity']], on='item id', how='left')

    summary = forecast_merged.groupby('item id').agg({
        'forecasted_quantity': 'sum',
        'stock quantity': 'max'
    }).reset_index()

    insufficient_inventory = []
    for index, row in summary.iterrows():
        item_id = row['item id']
        total_forecasted_demand = row['forecasted_quantity']
        inventory_stock = row['stock quantity']
        
        if inventory_stock < total_forecasted_demand:
            insufficient_inventory.append({
                'item id': item_id,
                'forecasted_demand': total_forecasted_demand,
                'available_stock': inventory_stock
            })
    inventory_temp = pd.read_csv('inventory_data.csv')
    return render_template('inventory.html', inventory=inventory_temp, insufficient_inventory=insufficient_inventory)

if __name__ == '__main__':
    app.run(debug=True)
