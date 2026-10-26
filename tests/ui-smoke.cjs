const { JSDOM } = require('jsdom');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const { join } = require('node:path');
const assert = require('node:assert/strict');
const code = readFileSync(join(__dirname, '..', 'web', 'serverDownload.js'), 'utf8').replace(/^import .*;\r?\n/gm, '');
const model = {name:'model.safetensors',directory:'checkpoints',url:'https://huggingface.co/example/model/resolve/main/model.safetensors',isMissing:true};
const paths = {checkpoints:['/opt/ComfyUI/models/checkpoints']};
async function check(kind) {
 const panel = kind === 'panel';
 const markup = panel
   ? '<section data-testid="error-group-missing-model"><div data-testid="missing-model-importable-rows"></div><div data-testid="missing-model-actions"><button id="native">Download</button></div></section>'
   : '<section role="dialog"><h2>Missing models</h2><div class="scrollbar-custom overflow-y-auto rounded-lg"><div><span class="rounded-full">CHECKPOINT</span></div></div><div class="justify-end gap"><button id="native">Download</button></div></section>';
 const dom = new JSDOM('<div id="vue-app"></div>'+markup,{url:'http://localhost',runScripts:'outside-only'});
 const {window} = dom;
 const calls = [];
 const stores = new Map(panel
  ? [['missingModel',{missingModelCandidates:[model],folderPaths:paths}]]
  : [['dialog',{dialogStack:[{key:'global-missing-models-warning',contentProps:{missingModels:[model],paths}}]}]]);
 window.document.getElementById('vue-app').__vue_app__={config:{globalProperties:{$pinia:{_s:stores}}}};
 window.app = {registerExtension(){},graph:null};
 window.api = {
   addEventListener(){},
   getFolderPaths:async()=>paths,
   fetchApi:async(path,options={})=>{
     calls.push({path,options});
     return {ok:true,json:async()=>path.endsWith('folder_paths') ? paths : {success:true}};
   }
 };
 window.setTimeout=()=>0;
 window.setInterval=()=>0;
 vm.runInContext(code,dom.getInternalVMContext());
 await window.injectServerDownloadButtons();
 const button=window.document.querySelector('.server-download-all-btn');
 assert.ok(button,`${kind}: server download button exists`);
 assert.equal(button.disabled,false,`${kind}: button enabled`);
 assert.match(button.textContent,/Download to Pod \(1\)/);
 if(panel) assert.equal(button.parentElement.dataset.testid,'missing-model-actions');
 await window.injectServerDownloadButtons();
 assert.equal(window.document.querySelectorAll('.server-download-all-btn').length,1);
 await button.onclick({stopPropagation(){}});
 const post=calls.find(c=>c.path==='/server_download/start');
 assert.ok(post,`${kind}: click sends download to server`);
 assert.equal(post.options.method,'POST');
 assert.deepEqual(JSON.parse(post.options.body),{url:model.url,save_path:'checkpoints',filename:model.name});
 assert.ok(window.document.getElementById('native'),`${kind}: native button preserved`);
 console.log(`PASS ${kind}: one enabled Download to Pod button; click posts correct model to server`);
 dom.window.close();
}
(async()=>{await check('panel');await check('dialog')})().catch(e=>{console.error(e);process.exitCode=1});
