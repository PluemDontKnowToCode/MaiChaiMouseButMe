let myCollection = [];
const nameInput = document.getElementById('mouseName');
const suggestionsBox = document.getElementById('suggestions');
const collectionDiv = document.getElementById('myCollection');
const form = document.getElementById('mouseForm');
const resultDiv = document.getElementById('result');

const addForm = document.getElementById('addForm');
const addResult = document.getElementById('addResult');

// Render My Collection
function renderCollection(){
    collectionDiv.innerHTML = '';
    myCollection.forEach((m,i)=>{
        const div = document.createElement('div');
        div.textContent = m + ' ';
        const btn = document.createElement('button');
        btn.textContent = 'Delete';
        btn.onclick = () => {
            myCollection.splice(i,1);
            renderCollection();
        };
        div.appendChild(btn);
        collectionDiv.appendChild(div);
    });
}
//Add new
addForm.onsubmit = async (e) => {
    e.preventDefault();
    addResult.textContent = '';

    // Collect form data
    const formData = {};
    addForm.querySelectorAll('input').forEach(input => {
        formData[input.name] = input.value;
    });

    // Send to backend
    const res = await fetch('/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(formData)
    });
    const data = await res.json();

    if (data.success) {
        addResult.textContent = data.success;
        addResult.style.color = 'lime';
        addForm.reset();
    } else if (data.error) {
        addResult.textContent = data.error;
        addResult.style.color = 'red';
    }
};
nameInput.addEventListener('input', async () => {
    const query = nameInput.value.trim();
    if(query.length < 2){ suggestionsBox.style.display='none'; return; }

    const res = await fetch(`/search?query=${encodeURIComponent(query)}`);
    const data = await res.json();

    suggestionsBox.innerHTML = '';
    if(Array.isArray(data) && data.length > 0){
        // ปรับความกว้างตรงกับ input
        suggestionsBox.style.width = nameInput.offsetWidth + 'px';

        data.forEach(item=>{
            const div = document.createElement('div');
            // Left: Model (Brand)
            const infoSpan = document.createElement('span');
            infoSpan.textContent = `${item.Model} (${item.Brand})`;
            div.appendChild(infoSpan);

            // Add button
            const addBtn = document.createElement('button');
            addBtn.textContent = 'Add';
            addBtn.onclick = (e) => {
                e.stopPropagation();
                if(!myCollection.includes(item.Model)){
                    myCollection.push(item.Model);
                    renderCollection();
                }
            };
            div.appendChild(addBtn);

            // Detail button
            const detailBtn = document.createElement('button');
            detailBtn.textContent = 'Detail';
            detailBtn.style.marginLeft = '8px';
            detailBtn.onclick = async (e) => {
                e.stopPropagation();
                // Build decorated modal
                let detailHtml = `<div id='mouseDetailModal' style='position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.45);z-index:9999;display:flex;align-items:center;justify-content:center;'>`;
                detailHtml += `<div style='background:#181a1b;border-radius:18px;box-shadow:0 8px 32px #00ffea55;padding:32px 38px;min-width:320px;max-width:90vw;color:#00ffea;font-family:Orbitron,Segoe UI,Arial,sans-serif;position:relative;'>`;
                detailHtml += `<button id='closeDetailModal' style='position:absolute;top:12px;right:18px;background:#ff0057;color:#fff;border:none;border-radius:6px;padding:4px 12px;font-size:1em;cursor:pointer;'>✖</button>`;
                detailHtml += `<h2 style='margin-top:0;margin-bottom:18px;color:#00ffea;text-shadow:0 2px 8px #222;'>${item.Model}</h2>`;
                detailHtml += `<h3 style='margin:0 0 18px 0;color:#fff;font-weight:normal;'>Brand: <span style='color:#00ffea;'>${item.Brand}</span></h3>`;
                detailHtml += `<table style='width:100%;border-collapse:collapse;background:transparent;color:#fff;'>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>DPI</td><td style='padding:8px 12px;color:#00ffea;'>${item.DPI ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Polling rate (Hz)</td><td style='padding:8px 12px;color:#00ffea;'>${item['Polling rate (Hz)'] ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Weight (grams)</td><td style='padding:8px 12px;color:#00ffea;'>${item['Weight (grams)'] ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Length (mm)</td><td style='padding:8px 12px;color:#00ffea;'>${item['Length (mm)'] ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Width (mm)</td><td style='padding:8px 12px;color:#00ffea;'>${item['Width (mm)'] ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Height (mm)</td><td style='padding:8px 12px;color:#00ffea;'>${item['Height (mm)'] ?? '-'}</td></tr>`;
                detailHtml += `<tr><td style='padding:8px 12px;'>Side buttons</td><td style='padding:8px 12px;color:#00ffea;'>${item['Side buttons'] ?? '-'}</td></tr>`;
                detailHtml += `</table>`;
                detailHtml += `</div></div>`;
                // Insert modal into DOM
                const modalDiv = document.createElement('div');
                modalDiv.innerHTML = detailHtml;
                document.body.appendChild(modalDiv);
                // Close handler
                modalDiv.querySelector('#closeDetailModal').onclick = () => {
                    document.body.removeChild(modalDiv);
                };
                // Also close on background click
                modalDiv.querySelector('#mouseDetailModal').onclick = (ev) => {
                    if(ev.target === modalDiv.querySelector('#mouseDetailModal')){
                        document.body.removeChild(modalDiv);
                    }
                };
            };
            div.appendChild(detailBtn);

            suggestionsBox.appendChild(div);

            div.onclick = () => { nameInput.value = item.Model; suggestionsBox.style.display='none'; };
        });
        suggestionsBox.style.display = 'block';
    } else {
        suggestionsBox.style.display='none';
    }
});


document.addEventListener('click', (e) => { 
    if(!suggestionsBox.contains(e.target) && e.target !== nameInput){ 
        suggestionsBox.style.display='none'; 
    } 
});

// Process Mouse
form.onsubmit = async (e) => {
    e.preventDefault();

    if(myCollection.length === 0){
        alert('Please add at least one mouse to My Collection');
        return;
    }

    // เก็บ features ที่เลือก
    const selected_features = [];
    form.querySelectorAll('input[name="features"]:checked').forEach(f => selected_features.push(f.value));
    if(selected_features.length === 0){
        alert('Please select at least one feature');
        return;
    }

    // เตรียมข้อมูลส่งไป backend
    const formData = new FormData();
    formData.append('name', nameInput.value);
    formData.append('my_mouse_collection', myCollection.join(',')); // ส่งเป็น comma-separated
    selected_features.forEach(f => formData.append('features', f));

    // เรียก backend
    const res = await fetch('/process', {method:'POST', body: formData});
    const data = await res.json();

    if(data.result){ 
        resultDiv.innerHTML = `<p>${data.result}</p>`; 
        return; 
    }

    // แสดงผลลัพธ์
    // หลังแสดงผล Top 5 Recommendations
let html = `<div class="card"><h3>Result</h3>`;
html += `<p><b>Selected Features:</b> ${data.selected_features.join(', ')}</p>`;
html += `<h4>Top 5 Recommended Mice</h4><table><tr><th>#</th><th>Model</th><th>Brand</th><th>Distance</th></tr>`;
data.recommendations.forEach((r,i)=>{
    html += `<tr><td>${i+1}</td><td>${r.Model}</td><td>${r.Brand}</td><td>${r.Distance_to_Ideal.toFixed(4)}</td></tr>`;
});
html += `</table>`;

// ปุ่ม View PCA Details
html += `<button id="viewPCA" style="margin-top:10px;background:#17a2b8;color:white;">🔎 View PCA Details</button>`;
html += `</div>`;
resultDiv.innerHTML = html;

// กดปุ่มเปิดหน้า PCA Details
document.getElementById('viewPCA').onclick = () => {
    const features = [];
    form.querySelectorAll('input[name="features"]:checked').forEach(f => features.push(f.value));
    const featuresStr = encodeURIComponent(features.join(','));
    const collectionStr = encodeURIComponent(myCollection.join(','));
    window.open(`/pca-details-page?name=${encodeURIComponent(nameInput.value)}&features=${featuresStr}&my_mouse_collection=${collectionStr}`, '_blank');
};


};