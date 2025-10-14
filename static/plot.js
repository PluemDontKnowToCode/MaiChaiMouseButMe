function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = (hash * 137.508) % 360;
    return `hsl(${hue}, 80%, 65%)`;
}



const flatMiceData = Object.values(dataByBrand).flat();
const sortedMice = [...flatMiceData].sort((a, b) => a.Distance_to_Ideal - b.Distance_to_Ideal);
const topMice = sortedMice.slice(0, 20);

const traceAllMice = {
    x: flatMiceData.map(mouse => mouse.PC1),
    y: flatMiceData.map(mouse => mouse.PC2),
    text: flatMiceData.map(mouse => `${mouse.Model} (${mouse.Brand})`), // ข้อความเมื่อเอาเมาส์ไปชี้
    mode: 'markers',
    type: 'scatter',
    name: 'All Mice',
    marker: { 
        size: 10, 
        color: 'rgba(0, 123, 255, 0.7)' 
    }
};


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
const data = [traceAllMice, traceIdealProfile];


const layout = {
    title: 'การกระจายของเมาส์ในแกน PCA (แยกตามยี่ห้อ)',
    xaxis: { title: 'แกนคุณลักษณะที่ 1 (PC1)' },
    yaxis: { title: 'แกนคุณลักษณะที่ 2 (PC2)' },
    hovermode: 'closest'
};

Plotly.newPlot('myDiv', data, layout);