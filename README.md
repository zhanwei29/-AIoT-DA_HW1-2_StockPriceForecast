# -AIoT-DA_HW1-2_StockPriceForecast
用Prophet根據2330.csv進行股價預測分析並產生未來5個月的股價預測及信心區間

-------------------------------------------------------------------
Prompt1:
根據檔案2330-training.csv，用python去 solve "auto regression" problem，要照著CRISP-datamining(DM)的步驟，1.)將檔案2330-training.csv中的A column 當作附圖中的x軸並標示年分，2.)用檔案2330-training.csv中的B到G column各別代表的x1到x6當作"auto regression" problem中的輸入，並預測出y軸，也就是附圖的當日收盤價，做出像附圖一樣的圖，並要包含 1. feature selection 2. model evaluation 3. web implementation deployment 4. 預測未來12個月分別的收盤價並產生預測範圍的漏斗圖 (websim) 

Prompt2:
為甚麼我的預測圖中沒有2023年份的資料
為甚麼我的未來預測曲線(紅線)的部分是一條平行線，並沒有顯示該有的曲折線

Prompt3:
跑出來的圖示這樣的，並沒有漏斗圖，而且未來預測也在0元


Prompt4:
help me with using Python to do a auto regression problem to predict the stock price here is my code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 讀取資料集
data = pd.read_csv('你的資料集.csv')

# 移除數字中的逗號，並轉換為浮點數
data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']] = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].replace({',': ''}, regex=True).astype(float)

# 確保日期欄位為 datetime 格式
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# 定義自變量和應變量
X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y = data['x1']  # 假設 'x1' 是目標變量，根據你的需求可以調整

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測並評估模型
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

# 繪製結果圖
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['x1'], color='blue', label='Historical Prices')  # 繪製歷史資料

# 預測未來的12個月數據（假設每個月有一個觀測值）
future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.DateOffset(months=1), periods=12, freq='M')
future_X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].iloc[-1].values.reshape(1, -1)  # 使用最新資料進行預測
future_predictions = []

# 進行未來12個月的預測
for _ in range(12):
    future_pred = model.predict(future_X)
    future_predictions.append(future_pred[0])
    # 這裡假設你會根據未來預測的值更新 x1 並進行下一步預測，具體邏輯可以根據需求修改
    future_X[0][0] = future_pred[0]  # 假設 x1 是你想要預測的未來數值

# 繪製未來預測
plt.plot(future_dates, future_predictions, color='red', label='Future Predictions')

# 圖形標題和標籤
plt.title('Future Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# 顯示圖表
plt.show()

Prompt5:
i want to change the prediction part, i want the gray area become the upper trend prediction

Prompt6:
this is the result, but i want it looks like the second picture

can i have the upper trend like the second picture on my last question,which the upper trend is cover by the area of two prediction line,one is the stock price rise line at the top and the second line is the downward stock price line at the bottom

Prompt7:
i want to change my question, i want 2 line after the historical line : 1. the upward trend 2. the downward trend

Prompt8:
why the two line is straight , i want this two line be the prediction lines

Prompt9:
this is the result, i wand the green line looks like the tail part in second picture which green line(upward trend line) is the upper bound of the blue part of the tail part in second picture , and the red line(downward trend line) is the lower bound of the blue part of the tail part in second picture

Prompt10:
i want the green、black、red line start at the same start point after the historical line which is the initialization forecast

Prompt11:
what's the difference between historical and actual price

i want to delete the forecast line( orange line) and change the future predicting line( black line with dots) into black line without dots

Prompt12:
Can we try a different forecasting model?
Let's try the Facebook Prophet model.

Prompt13:
can you line all the black dots and i want to forecast 5 months after the data

Prompt14:
why the forecast part looks like the same

i want the uncertainty interval has the feature which the closer the time is to now, the prediction interval will be more convergent; conversely, the further the time is from now, the larger the prediction interval will be, indicating that the uncertainty will be higher.

Prompt15:
i want the uncertainty looks like the funnel like this picture

Prompt16:
this is my result, why the uncertainty interval doesn't have this feature :The closer the time is to now, the prediction interval will be more convergent; conversely, the further the time is from now, the larger the prediction interval will be, indicating that the uncertainty will be higher.

Prompt17:
model = Prophet(interval_width=0.95, changepoint_prior_scale=0.5, uncertainty_samples=10) what's every parameters in the code

final result
