from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from prophet import Prophet
import os

# 使用非 GUI 的後端，以便能夠在 Flask 中生成圖片
matplotlib.use('Agg')

app = Flask(__name__, static_url_path='/static')  # 確保 Flask 正確提供 static 文件

# 設定靜態檔案目錄
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 建立 static/uploads 資料夾
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print(f"Current working directory: {os.getcwd()}")  # 打印工作目錄檢查路徑

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 檢查是否有上傳檔案
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # 儲存上傳的檔案
        if file:
            filename = file.filename  # 保存檔案名稱以在網頁上顯示
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 處理檔案並做預測
            data = pd.read_csv(file_path)
            
            # 轉換 'Date' 成 datetime 格式
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

            # 清理數據：將數值型欄位的逗號移除並轉換為浮點數
            for col in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
                data[col] = data[col].replace({',': ''}, regex=True).astype(float)

            # 準備 Prophet 模型所需的數據
            data_prophet = pd.DataFrame()
            data_prophet['ds'] = data['Date']
            data_prophet['y'] = data['x1']

            # 初始化 Prophet 模型
            model = Prophet(interval_width=0.95, changepoint_prior_scale=0.5, uncertainty_samples=500)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            # 訓練模型
            model.fit(data_prophet)

            # 預測未來 60 天的數據
            future_dates = model.make_future_dataframe(periods=60)
            forecast = model.predict(future_dates)

            # 畫出預測結果
            fig, ax = plt.subplots(figsize=(10, 6))
            model.plot(forecast, ax=ax)
            ax.set_title('Stock Price Forecast with Enhanced Uncertainty Widening')
            ax.plot(data_prophet['ds'], data_prophet['y'], color='black', lw=2, label='Actual Data')

            # 添加標註
            ax.axvline(x=data_prophet['ds'].iloc[-1], color='red', linestyle='--')
            ax.text(data_prophet['ds'].iloc[-1], forecast['yhat'].iloc[-1] + 100, 'Forecast Initialization', color='red')
            ax.annotate('Upward Trend', xy=(future_dates.iloc[20], forecast['yhat'].iloc[20]),
                        xytext=(future_dates.iloc[20], forecast['yhat'].iloc[20] + 100),
                        arrowprops=dict(facecolor='green', shrink=0.05))
            ax.legend()

            # 儲存圖表並回傳結果
            try:
                plot_path = os.path.join(app.root_path, 'static/uploads', 'forecast_plot.png')  # 使用 app.root_path 確保是相對於 Flask 根目錄
                plt.tight_layout()
                plt.savefig(plot_path)
                print(f"Image saved at: {plot_path}")  # 確認圖片是否正確保存
            except Exception as e:
                print(f"Failed to save image: {e}")
            finally:
                plt.close()

            # 渲染模板並將檔案名稱和圖片路徑傳給前端
            return render_template('index.html', plot_url='/static/uploads/forecast_plot.png', uploaded=True, filename=filename)

    return render_template('index.html', uploaded=False)

if __name__ == "__main__":
    app.run(debug=True)
