function stringToColor(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                hash = str.charCodeAt(i) + ((hash << 5) - hash);
            }
            const hue = (hash * 137.508) % 360;
            return `hsl(${hue}, 80%, 65%)`; 
        }

        const dataByBrand = {{ all_mice_json | tojson }};
        const idealProfile = {{ ideal_profile_json | tojson }};

        const data = [];

        for (const brand in dataByBrand) {
            const miceForBrand = dataByBrand[brand];
            
            const trace = {
                x: miceForBrand.map(mouse => mouse.PC1),
                y: miceForBrand.map(mouse => mouse.PC2),
                text: miceForBrand.map(mouse => mouse.Model),
                mode: 'markers',
                type: 'scatter',
                name: brand, 
                marker: {
                    size: 11,
                    color: stringToColor(brand) 
                }
            };
            data.push(trace);
        }

        const traceIdealProfile = {
            x: [idealProfile[0]],
            y: [idealProfile[1]],
            mode: 'markers', type: 'scatter', name: 'Ideal Profile',
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