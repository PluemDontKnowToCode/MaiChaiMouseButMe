// static/plot.js

// รอให้หน้าเว็บโหลดเสร็จก่อนจึงเริ่มทำงาน
document.addEventListener('DOMContentLoaded', function () {

    // --- 1. อ่านข้อมูลจาก "สะพาน" ใน HTML ---
    const dataByBrandElement = document.getElementById('data-by-brand');
    const idealProfileElement = document.getElementById('ideal-profile');

    if (!dataByBrandElement || !idealProfileElement) {
        console.error("Data script tags not found in HTML!");
        return;
    }

    // แปลงข้อมูลจาก JSON string ให้เป็น JavaScript object
    const dataByBrand = JSON.parse(dataByBrandElement.textContent);
    const idealProfile = JSON.parse(idealProfileElement.textContent);

    
    // --- 2. ส่วนของการวาดกราฟที่ถูกต้อง (ไม่มีโค้ด {{...}} เลย) ---
    function stringToColor(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        const hue = (hash * 137.508) % 360;
        return `hsl(${hue}, 80%, 65%)`;
    }

    const data = []; // Array สำหรับเก็บ trace ของแต่ละยี่ห้อ

    // --- วนลูปเพื่อสร้างกราฟทีละยี่ห้อ (นี่คือหัวใจหลัก) ---
    for (const brand in dataByBrand) {
        const miceForBrand = dataByBrand[brand];
        const trace = {
            x: miceForBrand.map(mouse => mouse.PC1),
            y: miceForBrand.map(mouse => mouse.PC2),
            text: miceForBrand.map(mouse => mouse.Model),
            mode: 'markers',
            type: 'scatter',
            name: brand, // ตั้งชื่อ Legend ตามยี่ห้อ
            marker: {
                size: 11,
                color: stringToColor(brand) // สร้างสีตามยี่ห้อ
            }
        };
        data.push(trace);
    }

    // เพิ่มดาวสีแดง
    const traceIdealProfile = {
        x: [idealProfile[0]],
        y: [idealProfile[1]],
        mode: 'markers',
        type: 'scatter',
        name: 'Ideal Profile',
        marker: {
            symbol: 'star', color: 'red', size: 20,
            line: { color: 'black', width: 2 }
        }
    };
    data.push(traceIdealProfile);

    const layout = {
        title: 'การกระจายของเมาส์ในแกน PCA (แยกตามยี่ห้อ)',
        xaxis: { title: 'แกนคุณลักษณะที่ 1 (PC1)' },
        yaxis: { title: 'แกนคุณลักษณะที่ 2 (PC2)' },
        hovermode: 'closest'
    };

    Plotly.newPlot('myDiv', data, layout);
});