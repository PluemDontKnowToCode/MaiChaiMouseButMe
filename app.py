from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np

app = Flask(__name__)

# --- Global Variables ---
features = [
    'DPI',
    'Polling rate (Hz)',
    'Weight (grams)',
    'Length (mm)',
    'Width (mm)',
    'Height (mm)',
    'Side buttons'
]

# โหลดข้อมูลเริ่มต้น
df = pd.read_csv('2025_09_06_MousedB.csv', sep=';')

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/process', methods=['POST'])
def process_mouse():
    name = request.form.get('name')
    selected_features = request.form.getlist('features')

    # ใช้ข้อมูล df ที่โหลดไว้ทำการวิเคราะห์
    existing_features = [f for f in selected_features if f in df.columns]
    df_features = df[existing_features].copy()

    # จัดการ Missing Values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=existing_features)

    # สเกลข้อมูล
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=existing_features)

    # ทำ PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)

    df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df_pca['Model'] = df['Model']
    df_pca['Brand'] = df['Brand']

    # สมมติว่าผู้ใช้มีเมาส์ในคลัง (test)
    my_mouse_collection = ['Viper Mini Signature Edition', 'MAD R Major', 'Susanto-X', 'Y2 Pro']
    df_my_collection = df_pca[df_pca['Model'].isin(my_mouse_collection)]

    if df_my_collection.empty:
        return jsonify({'result': f'ไม่พบข้อมูลเมาส์ในคลังสำหรับคำนวณโปรไฟล์'})

    # คำนวณโปรไฟล์ในอุดมคติ
    ideal_mouse_profile = df_my_collection[['PC1', 'PC2']].mean().values

    # คำนวณระยะห่างจากเมาส์แต่ละตัว
    distances = np.linalg.norm(df_pca[['PC1', 'PC2']].values - ideal_mouse_profile, axis=1)
    df_pca['Distance_to_Ideal'] = distances

    # เมาส์ที่ใกล้เคียงที่สุด
    recommendations = df_pca.sort_values('Distance_to_Ideal', ascending=True).head(5)
    result = recommendations[['Model', 'Brand', 'Distance_to_Ideal']].to_dict(orient='records')

    return jsonify({
        'name': name,
        'selected_features': selected_features,
        'recommendations': result
    })


if __name__ == '__main__':
    app.run(debug=True)
