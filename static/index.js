let myCollection = [];
const nameInput = document.getElementById('mouseName');
const suggestionsBox = document.getElementById('suggestions');
const collectionDiv = document.getElementById('myCollection');
const form = document.getElementById('mouseForm');
const resultDiv = document.getElementById('result');

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
            div.textContent = `${item.Model} (${item.Brand})`;

            const btn = document.createElement('button');
            btn.textContent = 'Add';
            btn.onclick = (e) => {
                e.stopPropagation();
                if(!myCollection.includes(item.Model)){
                    myCollection.push(item.Model);
                    renderCollection();
                }
            };
            div.appendChild(btn);
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