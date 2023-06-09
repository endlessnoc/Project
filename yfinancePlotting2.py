 
import yfinance as yf
import matplotlib.pyplot as plt
start_date = '2013-05-16'
end_date = '2023-05-16'

#引入聯發科資料
mt = yf.Ticker("2454.TW")
mt_history = mt.history(start=start_date, end=end_date)
mt_return = mt_history['Close'][-1] / mt_history['Close'][0]

#引入台灣大盤指數資料
tw = yf.Ticker("^TWII")
tw_history = tw.history(start=start_date, end=end_date)
tw_return = tw_history['Close'][-1] / tw_history['Close'][0]

# 計算兩者報酬率
mt_return_rate = mt_return / 100
tw_return_rate = tw_return / 100

#設置Plotting相關設置和透明背景並存檔
plt.plot(mt_history.index, mt_history['Close'] / mt_history['Close'][0], label="MediaTek")
plt.plot(tw_history.index, tw_history['Close'] / tw_history['Close'][0], label="TAIEX")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Cumulative Return ")
plt.title("Cumulative Return in 10 years")
plt.savefig("return_rate.png", transparent=True)

