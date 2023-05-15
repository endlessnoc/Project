import yfinance as yf
import matplotlib.pyplot as plt
start_date = '2022-05-15'
end_date = '2023-05-15'

#載入資料
mt = yf.Ticker('2454.TW')
twii = yf.Ticker('^TWII')

mt_data = mt.history(start=start_date, end=end_date)
twii_data = twii.history(start=start_date, end=end_date)

#設定Plotting相關設置
fig, ax1 = plt.subplots()
ax1.plot(mt_data['Close'], color='blue', label='MediaTek')
ax1.set_ylabel('MediaTek')

ax2 = ax1.twinx()
ax2.plot(twii_data['Close'], color='red', label='TWII')
ax2.set_ylabel('TWII Index')


ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

#設定成透明背景
fig.patch.set_alpha(0)
#存檔
plt.savefig('stock_prices.png', transparent=True)

