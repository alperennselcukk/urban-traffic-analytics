"""
UrbanTraffic-IST: Şehir İçi Trafik Gecikme ve Tahmin Analizi
Geliştirici: Alperen Selçuk
Teknolojiler: Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.gridspec as gridspec
np.random.seed(42)
n_samples = 1200

data = {
    'Mesafe_km': np.round(np.random.uniform(5, 45, n_samples), 1),
    'Saat': np.random.randint(0, 24, n_samples),
    'Hava_Durumu': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]), # 0: Açık, 1: Yağmurlu, 2: Fırtına
    'Kaza_Etkisi': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Yol_Calismasi': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'Hafta_Ici': np.random.choice([1, 0], n_samples, p=[0.71, 0.29])
}

df = pd.DataFrame(data)

# Temel süre hesaplama
base_time = df['Mesafe_km'] * 1.5

# Zirve saatler (07-09 ve 17-19)
is_rush = df['Saat'].isin([7, 8, 9, 17, 18, 19])

# DÜZELTİLEN KISIM: Parantezli ve temiz çarpan hesabı
rush_multiplier = 1.0 + (is_rush & (df['Hafta_Ici'] == 1)) * 1.8 + ((~is_rush) & (df['Hafta_Ici'] == 1)) * 0.4

# Etkiler
weather_delay = df['Hava_Durumu'] * 12
accident_delay = df['Kaza_Etkisi'] * 25
roadwork_delay = df['Yol_Calismasi'] * 15

# Toplam Gecikme (Dakika)
df['Toplam_Gecikme_dk'] = np.round(
    (base_time * rush_multiplier) + weather_delay + accident_delay + roadwork_delay + np.random.normal(5, 3, n_samples),
    1
).clip(lower=2)

print("--- TRAFİK VERİ SETİ BAŞARIYLA OLUŞTU ---")
print(df.head())

X = df.drop(columns=['Toplam_Gecikme_dk'])
y = df['Toplam_Gecikme_dk']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

multi_lr = LinearRegression()
multi_lr.fit(X_train_scaled,y_train)
y_pred_multi = multi_lr.predict(X_test_scaled)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled,y_train)
y_pred_knn = knn.predict(X_test_scaled)

r2_multi = r2_score(y_test,y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test,y_pred_multi))
mae_multi = mean_absolute_error(y_test,y_pred_multi)

r2_knn = r2_score(y_test,y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test,y_pred_knn))
mae_knn = mean_absolute_error(y_test,y_pred_knn)

comparison_df = pd.DataFrame({
    'Model': ['Çoklu Lineer Regresyon', 'KNN Regressor (k=5)'],
    'R2 Score': [r2_multi, r2_knn],
    'RMSE (Dk)': [rmse_multi, rmse_knn],
    'MAE (Dk)': [mae_multi, mae_knn]
})
print(comparison_df)


BG_COLOR = '#F8FAFC'
CARD_COLOR = '#FFFFFF'
PRIMARY_TEXT = '#0F172A'
SECONDARY_TEXT = '#475569'

fig = plt.figure(figsize=(16, 11), facecolor=BG_COLOR)
gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.4, 1, 1], hspace=0.35, wspace=0.25)

# --- ÜST BİLGİ & KPI KARTLARI ---
# KPI 1: Toplam Senaryo
ax_kpi1 = fig.add_subplot(gs[0, 0])
ax_kpi1.set_facecolor(CARD_COLOR)
ax_kpi1.text(0.5, 0.65, '1,200', fontsize=22, fontweight='bold', color='#2563EB', ha='center', va='center')
ax_kpi1.text(0.5, 0.25, 'Toplam Trafik Senaryosu', fontsize=10, color=SECONDARY_TEXT, ha='center', va='center')
ax_kpi1.axis('off')

# KPI 2: En Yoğun Saat
peak_hour = df.groupby('Saat')['Toplam_Gecikme_dk'].mean().idxmax()
ax_kpi2 = fig.add_subplot(gs[0, 1])
ax_kpi2.set_facecolor(CARD_COLOR)
ax_kpi2.text(0.5, 0.65, f'{peak_hour}:00', fontsize=22, fontweight='bold', color='#D97706', ha='center', va='center')
ax_kpi2.text(0.5, 0.25, 'Zirve Trafik Saati (Rush)', fontsize=10, color=SECONDARY_TEXT, ha='center', va='center')
ax_kpi2.axis('off')

# KPI 3: KNN R2 Başarısı
ax_kpi3 = fig.add_subplot(gs[0, 2])
ax_kpi3.set_facecolor(CARD_COLOR)
ax_kpi3.text(0.5, 0.65, f'%{round(r2_knn*100, 1)}', fontsize=22, fontweight='bold', color='#059669', ha='center', va='center')
ax_kpi3.text(0.5, 0.25, 'KNN Model Doğruluğu (R²)', fontsize=10, color=SECONDARY_TEXT, ha='center', va='center')
ax_kpi3.axis('off')


# --- GRAFİK 1: DONUT CHART (Hava Durumu Dağılımı) ---
ax_donut = fig.add_subplot(gs[1, 0])
ax_donut.set_facecolor(CARD_COLOR)
weather_counts = df['Hava_Durumu'].value_counts()
weather_labels = ['Açık', 'Yağmurlu', 'Fırtına']
colors_donut = ['#38BDF8', '#818CF8', '#C084FC']

wedges, texts, autotexts = ax_donut.pie(
    weather_counts, labels=weather_labels, autopct='%1.1f%%',
    startangle=140, colors=colors_donut, pctdistance=0.75,
    textprops=dict(color=PRIMARY_TEXT, fontweight='bold')
)
# Donut deliği açma
centre_circle = plt.Circle((0,0), 0.55, fc=CARD_COLOR)
ax_donut.add_artist(centre_circle)
ax_donut.set_title('Hava Koşulları Dağılımı', fontsize=11, fontweight='bold', color=PRIMARY_TEXT, pad=10)


# --- GRAFİK 2: MESAFE VS GECİKME TRENDİ (Scatter + Trend Line) ---
ax_scatter = fig.add_subplot(gs[1, 1:])
ax_scatter.set_facecolor(CARD_COLOR)
sns.regplot(data=df, x='Mesafe_km', y='Toplam_Gecikme_dk', ax=ax_scatter,
            color='#0284C7', scatter_kws={'alpha':0.4, 's':25}, line_kws={'color':'#DC2626', 'linewidth':2})
ax_scatter.set_title('Mesafe (km) ile Gecikme Süresi İlişkisi ve Eğilim Çizgisi', fontsize=11, fontweight='bold', color=PRIMARY_TEXT)
ax_scatter.set_xlabel('Yolculuk Mesafesi (km)', color=SECONDARY_TEXT)
ax_scatter.set_ylabel('Gecikme (Dk)', color=SECONDARY_TEXT)
ax_scatter.grid(True, linestyle='--', alpha=0.5)


# --- GRAFİK 3: KEMAN GRAFİĞİ (Violin Plot - Hataların Yoğunlaşması) ---
ax_violin = fig.add_subplot(gs[2, :2])
ax_violin.set_facecolor(CARD_COLOR)
residuals_knn = y_test - y_pred_knn
residuals_multi = y_test - y_pred_multi

res_df = pd.DataFrame({
    'Model': ['Çoklu Lineer']*len(residuals_multi) + ['KNN (k=5)']*len(residuals_knn),
    'Hata': np.concatenate([residuals_multi, residuals_knn])
})

sns.violinplot(data=res_df, x='Model', y='Hata', ax=ax_violin, palette=['#F87171', '#34D399'], inner='quartile')
ax_violin.axhline(0, color='#0284C7', linestyle='--', linewidth=1.5)
ax_violin.set_title('Model Tahmin Hatalarının Yoğunluk Dağılımı (Violin Plot)', fontsize=11, fontweight='bold', color=PRIMARY_TEXT)
ax_violin.set_xlabel('')
ax_violin.set_ylabel('Hata Miktarı (Dk)', color=SECONDARY_TEXT)
ax_violin.grid(True, linestyle='--', alpha=0.5)


# --- GRAFİK 4: MODEL MAE KARŞILAŞTIRMASI ---
ax_mae = fig.add_subplot(gs[2, 2])
ax_mae.set_facecolor(CARD_COLOR)
bars = ax_mae.bar(['Lineer', 'KNN'], [mae_multi, mae_knn], color=['#EF4444', '#10B981'], width=0.45)
ax_mae.set_title('Ortalama Hata (MAE - Dk)', fontsize=11, fontweight='bold', color=PRIMARY_TEXT)
ax_mae.grid(True, linestyle='--', axis='y', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax_mae.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{round(yval, 2)}', 
                ha='center', va='bottom', fontweight='bold', color=PRIMARY_TEXT)

fig.suptitle('UrbanTraffic-IST  |  Gelişmiş Analitik Rapor ve Model Performansı', 
             fontsize=16, fontweight='bold', color=PRIMARY_TEXT, y=0.98)

plt.show()




