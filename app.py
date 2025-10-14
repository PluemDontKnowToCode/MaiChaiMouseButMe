from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np
import json

app = Flask(__name__)

# โหลด dataset และทำ copy สำหรับแก้ไข
df = pd.read_csv('2025_09_06_MousedB.csv', sep=';')
df_copy = df.copy()

# หน้าแรก
@app.route('/')
def index():
    features = ['DPI', 'Polling rate (Hz)', 'Weight (grams)',
                'Length (mm)', 'Width (mm)', 'Height (mm)', 'Side buttons']
    return render_template('index.html', features=features)

# Search suggestions
@app.route('/search')
def search_mouse():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify({'error':'⚠️ Please enter a search term'})
    mask = (df_copy['Model'].str.lower().str.contains(query, na=False) |
            df_copy['Brand'].str.lower().str.contains(query, na=False))
    results = df_copy[mask]
    if results.empty:
        return jsonify({'message':'❌ No matching mouse found'})
    return jsonify(results[['Model','Brand']].to_dict   (orient='records'))

# Add new mouse
@app.route('/add', methods=['POST'])
def add_mouse():
    global df_copy
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error':'No data received'}), 400

    required_cols = ['Model','Brand','DPI','Polling rate (Hz)',
                     'Weight (grams)','Length (mm)','Width (mm)',
                     'Height (mm)','Side buttons']

    for col in required_cols:
        if col not in data or str(data[col]).strip()=='':
            return jsonify({'error':f'Missing field: {col}'}), 400

    numeric_cols = ['DPI','Polling rate (Hz)','Weight (grams)',
                    'Length (mm)','Width (mm)','Height (mm)','Side buttons']
    for col in numeric_cols:
        try: data[col]=float(data[col])
        except ValueError:
            return jsonify({'error':f'Invalid number format for {col}'}), 400

    new_row = pd.DataFrame([data])
    df_copy = pd.concat([df_copy,new_row], ignore_index=True)
    df_copy.to_csv('mouse_data_copy.csv', sep=';', index=False)
    return jsonify({'success':f"✅ Added '{data['Model']}' to dataset copy!"})

# Process PCA + Top 5 recommendations
@app.route('/process', methods=['POST'])
def process_mouse():
    name = request.form.get('name','')
    selected_features = request.form.getlist('features')
    existing_features = [f for f in selected_features if f in df_copy.columns]
    if not existing_features:
        return jsonify({'result':'⚠️ ไม่ได้เลือกคุณสมบัติใดๆ'})

    df_features = df_copy[existing_features].copy()
    df_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df_features),
                              columns=existing_features)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_imputed),
                             columns=existing_features)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pcs, columns=['PC1','PC2'])
    df_pca['Model'] = df_copy['Model']
    df_pca['Brand'] = df_copy['Brand']

    # ใช้ input ของผู้ใช้แทน collection แบบคงที่
    my_mouse_input = request.form.get('my_mouse_collection', '')
    my_mouse_collection = [x.strip() for x in my_mouse_input.split(',') if x.strip()]

    if not my_mouse_collection:
        return jsonify({'result':'⚠️ ไม่พบเมาส์ในคลังสำหรับคำนวณโปรไฟล์'})

    df_my = df_pca[df_pca['Model'].isin(my_mouse_collection)]
    if df_my.empty:
        return jsonify({'result':'⚠️ ไม่พบเมาส์ที่คุณเลือกใน dataset'})

    ideal_profile = df_my[['PC1','PC2']].mean().values
    df_pca['Distance_to_Ideal'] = np.linalg.norm(df_pca[['PC1','PC2']].values - ideal_profile, axis=1)

    recommendations = df_pca.sort_values('Distance_to_Ideal').head(5)
    result = recommendations[['Model','Brand','Distance_to_Ideal']].to_dict(orient='records')

    return jsonify({'name':name,'selected_features':selected_features,'recommendations':result})


# PCA details page
@app.route('/pca-details-page')
def pca_details_page():
    
    my_mouse_input = request.args.get('my_mouse_collection','')
    features = request.args.get('features','')
    selected_features = [f for f in features.split(',') if f in df_copy.columns]
    my_mouse_collection = [x.strip() for x in my_mouse_input.split(',') if x.strip()]

    if not selected_features:
        return "<p>⚠️ No features selected.</p>"

    df_features = df_copy[selected_features].copy()
    df_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df_features),
                              columns=selected_features)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_imputed),
                             columns=selected_features)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pcs, columns=['PC1','PC2'])
    df_pca['Model']=df_copy['Model']
    df_pca['Brand']=df_copy['Brand']

   

    df_my = df_pca[df_pca['Model'].isin(my_mouse_collection)]
    ideal_profile=df_my[['PC1','PC2']].mean().values
    df_pca['Distance_to_Ideal']=np.linalg.norm(df_pca[['PC1','PC2']].values - ideal_profile, axis=1)


    table_data = df_pca.sort_values('Distance_to_Ideal').to_dict(orient='records')
    ideal_profile_data = ideal_profile.tolist()

    data_by_brand = {}
    for brand, group in df_pca.groupby('Brand'):
        data_by_brand[brand] = group[['PC1', 'PC2', 'Model']].to_dict(orient='records')

    return render_template(
       'plot.html', 
        table_rows=table_data,
        data_by_brand_json=data_by_brand, 
        ideal_profile_json=ideal_profile_data) 



if __name__=='__main__':
    app.run(debug=True)
