def clean_mouse_data(df, features):
    """
    Clean mouse data by:
    1. Removing rows for models that have both missing and non-missing data, keeping only the complete row.
    2. Combining brands for identical models with same features but different brands.
    Returns cleaned df_features and df_copy_clean.
    """
    df_features = df[features].copy()
    missing_mask = df_features.isnull().any(axis=1)
    models_with_missing = set(df.loc[missing_mask, 'Model'])
    complete_mask = ~df_features.isnull().any(axis=1)
    models_with_complete = set(df.loc[complete_mask, 'Model'])
    duplicate_models = models_with_missing & models_with_complete
    drop_idx = df.index[missing_mask & df['Model'].isin(duplicate_models)]
    df_features = df_features.drop(index=drop_idx)
    df_clean = df.drop(index=drop_idx)
    df_full = df_clean[['Model', 'Brand'] + features].copy()
    grouped = df_full.groupby(['Model'] + features, dropna=False)
    combined_rows = []
    for _, group in grouped:
        brands = ', '.join(sorted(set(group['Brand'])))
        row = group.iloc[0].copy()
        row['Brand'] = brands
        combined_rows.append(row)
    df_combined = pd.DataFrame(combined_rows)
    df_features = df_combined[features].copy()
    df_clean = df_combined[['Model', 'Brand'] + features].copy()
    return df_features, df_clean
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

# -------------------------------------------------------------------- clean data ------------------------------------------------------

# # feature ที่จะใช้ 
# features = [
#     'DPI',
#     'Polling rate (Hz)',
#     'Weight (grams)',
#     'Length (mm)',
#     'Width (mm)',
#     'Height (mm)',
#     'Side buttons'
# ]
# existing_features = [f for f in features if f in df.columns]

# valid_counts = df_features.notna().sum()
# total_counts = len(df_features)
# percent_valid = (valid_counts / total_counts * 100).round(2)
# percent_missing = (100 - percent_valid).round(2)


# stats = pd.DataFrame({
#     'Mean': df_copy.mean(),
#     'Max': df_copy.max(),
#     'Min': df_copy.min(),
#     'StdDev': df_copy.std(),
#     'Mode': df_copy.mode().iloc[0],
#     '%Valid': percent_valid,
#     '%Missing': percent_missing
# }).round(2)


# -----------------------------------------------------------------------------------------------------------------------------------------------------


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
        return jsonify({'message':'No matching mouse found'})
    # Return all details needed for the detail popup
    detail_cols = ['Model','Brand','DPI','Polling rate (Hz)','Weight (grams)','Length (mm)','Width (mm)','Height (mm)','Side buttons']
    return jsonify(results[detail_cols].to_dict(orient='records'))

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
    return jsonify({'success':f" Added '{data['Model']}' to dataset"})

# Process PCA + Top 5 recommendations
@app.route('/process', methods=['POST'])
def process_mouse():
    name = request.form.get('name','')
    selected_features = request.form.getlist('features')
    existing_features = [f for f in selected_features if f in df_copy.columns]
    if not existing_features:
        return jsonify({'result':'⚠️ ไม่ได้เลือกคุณสมบัติใดๆ'})

    # น่าจะต้องเอาอก เพราะจะ clean data ข้างบน -------------------------------------------------------------------------------------------
    # Clean and combine brands
    df_features, df_copy_clean = clean_mouse_data(df_copy, existing_features)

    # --- Impute and scale ---
    df_imputed = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df_features),
                              columns=existing_features)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_imputed),
                             columns=existing_features)
    


    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pcs, columns=['PC1','PC2'])
    df_pca['Model'] = df_copy_clean['Model']
    df_pca['Brand'] = df_copy_clean['Brand']

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

    df_recommend = df_pca[~df_pca['Model'].isin(my_mouse_collection)]
    recommendations = df_recommend.sort_values('Distance_to_Ideal').head(5)
    result = recommendations[['Model','Brand','Distance_to_Ideal']].to_dict(orient='records')

    return jsonify({'name':name,'selected_features':selected_features,'recommendations':result})

@app.route('/pca-details-page')
def pca_details_page():
    
    my_mouse_input = request.args.get('my_mouse_collection', '')
    features = request.args.get('features', '')

    selected_features = [f for f in features.split(',') if f in df_copy.columns]
    my_mouse_collection = [x.strip() for x in my_mouse_input.split(',') if x.strip()]

    if not selected_features:
        return render_template('plot.html', error="⚠️ No features selected.")

    debug_steps = []
    # Step 1: Select columns and clean data
    df_features, df_copy_clean = clean_mouse_data(df_copy, selected_features)
    # ...existing code...
    valid_counts = df_features.notna().sum()
    total_counts = len(df_features)
    percent_valid = (valid_counts / total_counts * 100).round(2)
    percent_missing = (100 - percent_valid).round(2)
    stats = pd.DataFrame({
        'Mean': df_features.mean(),
        'Max': df_features.max(),
        'Min': df_features.min(),
        'StdDev': df_features.std(),
        'Mode': df_features.mode().iloc[0],
        '%Valid': percent_valid,
        '%Missing': percent_missing
    }).round(2)
    stats_html = "<div class='card' style='margin-bottom:18px;'>"
    stats_html += "<div class='pca-topic'>Feature Statistics & Missing Value Summary</div>"
    stats_html += stats.to_html(classes='pca-detail')
    stats_html += "</div>"
    debug_steps.append({
        'topic': 'Step 1: Selected Features',
        'detail': str(pd.concat([df_copy[['Model']], df_features], axis=1).head())
    })

    # --------------------------------------------------------------------------------------------------------------------
    # Step 2: Impute Missing Values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=selected_features)
    debug_steps.append({
        'topic': 'Step 2: After Imputation (mean)',
        'detail': str(pd.concat([df_copy[['Model']], df_imputed], axis=1).head())
    })
    # Step 3: Standardize
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=selected_features)
    debug_steps.append({
        'topic': 'Step 3: After Standardization',
        'detail': str(pd.concat([df_copy[['Model']], df_scaled], axis=1).head())
    })
    # --------------------------------------------------------------------------------------------------------------------



    # Step 4: Center Data (Mean Shifting)
    X = df_imputed.values
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    debug_steps.append({
        'topic': 'Step 4: Center Data (Mean Shifting)',
        'detail': f"X_centered = X - mu\nmu = {mu}\nX_centered (first 5 rows):\n" + str(pd.DataFrame(X_centered[:5], columns=selected_features, index=df_copy['Model'][:5]).to_string())
    })

    # Step 5: Covariance Matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    debug_steps.append({
        'topic': 'Step 5: Covariance Matrix',
        'detail': f"C = (1/(n-1)) X_centered^T X_centered\nCovariance Matrix:\n" + str(np.round(cov_matrix, 4))
    })

    # Step 6: Eigenvalues & Eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    debug_steps.append({
        'topic': 'Step 6: Eigenvalues & Eigenvectors',
        'detail': f"np.linalg.eig(C)\nEigenvalues:\n{np.round(eigvals, 4)}\nEigenvectors (columns):\n{np.round(eigvecs, 4)}"
    })

    # Step 7: Sort Eigenvalues/Eigenvectors
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    debug_steps.append({
        'topic': 'Step 7: Sort Eigenvalues/Eigenvectors',
        'detail': f"Sorted Eigenvalues:\n{np.round(eigvals_sorted, 4)}\nSorted Eigenvectors:\n{np.round(eigvecs_sorted, 4)}"
    })

    # Step 8: PCA Transform (Project data)
    W_top = eigvecs_sorted[:, :2]
    Z = np.dot(X_centered, W_top)
    debug_steps.append({
        'topic': 'Step 8: PCA Transform',
        'detail': f"Z = X_centered . W_top\nW_top (first 2 eigenvectors):\n{np.round(W_top, 4)}\nZ (first 5 rows):\n" + str(pd.DataFrame(Z[:5], columns=['PC1','PC2'], index=df_copy['Model'][:5]).to_string())
    })

    # Step 9: Fit sklearn PCA for comparison and downstream
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    df_pca['Model'] = df_copy['Model']
    df_pca['Brand'] = df_copy['Brand']
    debug_steps.append({
        'topic': 'Step 9: Sklearn PCA Components (PC1, PC2)',
        'detail': str(df_pca.head())
    })
    

    # Step 10: Ideal Profile
    df_my = df_pca[df_pca['Model'].isin(my_mouse_collection)]
    if df_my.empty:
        return render_template('plot.html', error="⚠️ No matching models found in collection.")
    ideal_profile = df_my[['PC1', 'PC2']].mean().values
    debug_steps.append({
        'topic': 'Step 10: Ideal Profile (mean of selected)',
        'detail': str(ideal_profile)
    })

    # Step 11: Euclidean Distance
    df_pca['Distance_to_Ideal'] = np.linalg.norm(
        df_pca[['PC1', 'PC2']].values - ideal_profile, axis=1
    )
    debug_steps.append({
        'topic': 'Step 11: Distance to Ideal (Euclidean)',
        'detail': str(df_pca[['Model', 'Distance_to_Ideal']].head())
    })

    # Step 12: Prepare Data
    table_data = df_pca.sort_values('Distance_to_Ideal').to_dict(orient='records')
    ideal_profile_data = ideal_profile.tolist()
    data_by_brand = {}
    for brand, group in df_pca.groupby('Brand'):
        data_by_brand[brand] = group[['PC1', 'PC2', 'Model']].to_dict(orient='records')

    # Step-by-step explanation (Thai & English)
    debug_html = "<br><br>".join([
        f'<div class="pca-topic">{step["topic"]}</div><pre class="pca-detail">{step["detail"]}</pre>'
        for step in debug_steps
    ])
    return render_template(
        'plot.html',
        stats_html=stats_html,
        debug_html=debug_html,
        table_rows=table_data,
        data_by_brand_json=data_by_brand,
        ideal_profile_json=ideal_profile_data
    )
if __name__=='__main__':
    app.run(debug=True)
