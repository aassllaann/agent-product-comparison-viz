let request = require('request').defaults({ 'proxy': "http://127.0.0.1:7890" });
let Queue = require('promise-queue');
let fs = require('fs');
let path = require('path');
let queue = new Queue(10); //最多同时10线程采集

request('https://www.dxomark.com/daksensor/ajax/jsontested', //获得相机列表
  (error, response, body) => {
    if (error) { console.error("获取列表失败:", error); return; }
    
    let cameraList = JSON.parse(body).data;
    let finishedCount = 0;

    // 1. 过滤符合年份条件的数据
    let filteredList = cameraList.filter(cameraMeta => {
      let year = parseInt(cameraMeta.year); 
      return year >= 2015 && year <= 2025;
    });

    console.log(`符合条件的相机共: ${filteredList.length} 款`);

    // 2. 遍历过滤后的列表
    filteredList.forEach(cameraMeta => {
      let camera = Object.assign({}, cameraMeta);
      let link = `https://www.dxomark.com${camera.link}---Specifications`; 

      queue.add(() => new Promise(res => { 
        let doRequest = () => {
          request(link, (error, response, body) => {
            if (error) { 
              console.log("retrying.." + camera.name);
              doRequest();
              return;
            }
            
            let specMatcherRegexp = /descriptifgauche.+?>([\s\S]+?)<\/td>[\s\S]+?descriptif_data.+?>([\s\S]+?)<\/td>/img;
            let match = specMatcherRegexp.exec(body);
            while (match) {  
              camera[match[1]] = match[2];
              match = specMatcherRegexp.exec(body);
            }

            // 确保目录存在
            if (!fs.existsSync('./scraped')) fs.mkdirSync('./scraped');

            fs.writeFileSync(path.join('./scraped', camera.name.replace(/\//g, '-') + '.txt'), JSON.stringify(camera, null, 4), { encoding: "UTF8" });
            
            finishedCount++;
            console.log(`Finished ${finishedCount}/${filteredList.length}: ${camera.name}`);
            res();
          })
        };
        doRequest();
      }));
    });
  });