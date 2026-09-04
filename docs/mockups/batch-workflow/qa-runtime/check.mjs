import fs from 'node:fs/promises';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {parseHTML} from 'linkedom';

const source=await fs.readFile(new URL('../vipp-batch-workflow-comparison.html',import.meta.url),'utf8');
const script=source.match(/<script>([\s\S]*?)<\/script>/)[1];
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const outcomes=[];

function fixture(platform='Win32') {
  const {window}=parseHTML(source),document=window.document;
  Object.defineProperty(window.HTMLSelectElement.prototype,'value',{
    get(){return [...this.options].find(o=>o.hasAttribute('selected'))?.getAttribute('value')??this.options[0]?.getAttribute('value')??this.options[0]?.textContent??'';},
    set(v){[...this.options].forEach(o=>o.toggleAttribute('selected',(o.getAttribute('value')??o.textContent)===String(v)));},configurable:true
  });
  Object.defineProperty(window.HTMLInputElement.prototype,'checked',{
    get(){return this.hasAttribute('checked');},set(v){this.toggleAttribute('checked',Boolean(v));},configurable:true
  });
  Object.defineProperty(window.HTMLInputElement.prototype,'validity',{
    get(){return {valid:!this.value||Number(this.value)>=Number(this.min||0),badInput:false};},configurable:true
  });
  window.HTMLElement.prototype.showModal=function(){this.open=true;};
  window.HTMLElement.prototype.close=function(){this.open=false;};
  window.HTMLElement.prototype.getBoundingClientRect=function(){return {left:0,top:0,width:900,height:720};};
  Object.defineProperty(window.HTMLElement.prototype,'offsetWidth',{get(){return 160;},configurable:true});
  Object.defineProperty(window.HTMLElement.prototype,'offsetLeft',{get(){return 600;},configurable:true});
  const context=vm.createContext({document,setTimeout,clearTimeout,console,navigator:{platform}});
  vm.runInContext(script,context,{timeout:10000});
  const q=s=>{const element=document.querySelector(s);assert.ok(element,`Missing element: ${s}`);return element;};
  const qa=s=>[...document.querySelectorAll(s)];
  const dispatch=(s,type,props={})=>{const event=new window.Event(type,{bubbles:true,cancelable:true});Object.assign(event,props);q(s).dispatchEvent(event);};
  const press=s=>{assert.notEqual(q(s).disabled,true,`Cannot click disabled control: ${s}`);dispatch(s,'click');};
  const change=(s,value)=>{q(s).value=value;dispatch(s,'change');};
  const input=(s,value)=>{q(s).value=value;dispatch(s,'input');};
  const check=(s,value=true,shiftKey=false)=>{q(s).checked=value;dispatch(s,'click',{shiftKey});};
  const scenario=value=>change('#vbc-state',value);
  const row=id=>q(`[data-select-sample="${id}"]`).closest('tr');
  const cell=(id,key)=>`[data-cell-input="${key}"][data-cell-owner="${id}"]`;
  const field=key=>q(`[data-bulk-value="${key}"]`).closest('.vbn-bulk-field');
  const choose=(selector,checked=true)=>{q(selector).checked=checked;dispatch(selector,'change');};
  const stage=(key,value)=>{choose(`[data-bulk-pick="${key}"]`);input(`[data-bulk-value="${key}"]`,value);};
  const close=()=>press('.vbn-dialog [data-dialog-close]');
  return {document,q,qa,dispatch,press,change,input,check,choose,scenario,row,cell,field,stage,close};
}

async function test(name,run) {
  try {await run();outcomes.push({name,passed:true});console.log('PASS:',name);}
  catch(error) {outcomes.push({name,passed:false,error:error.message});console.error('FAIL:',name,'\n',error.stack);}
}

await test('table cells edit directly, inherit when cleared, and support choice and Boolean values',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');
  assert.equal(f.qa('#vbn-parameter-table .vbn-node-row [scope="colgroup"]').length,3);
  assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,7);
  assert.equal(f.q(f.cell('field_02','threshold')).value,'13000');
  assert.equal(f.q(f.cell('field_01','threshold')).value,'');
  assert.equal(f.q(f.cell('field_01','threshold')).getAttribute('placeholder'),'12,000');
  f.input(f.cell('field_01','threshold'),'15000');
  assert.equal(f.q('#vipp-batch-comparison').dataset.state,'stale');
  assert.equal(f.q(f.cell('field_01','threshold')).closest('td').classList.contains('has-override'),true);
  f.input(f.cell('field_02','threshold'),'');
  assert.equal(f.q(f.cell('field_02','threshold')).closest('td').classList.contains('is-inherited'),true);
  assert.match(f.q('#vbn-cell-feedback').textContent,/uses workflow \(12,000\)/);
  f.change(f.cell('field_01','boundary'),'Constant');
  assert.equal(f.q(f.cell('field_01','boundary')).value,'Constant');
  f.change(f.cell('field_01','above'),'false');
  assert.equal(f.q(f.cell('field_01','above')).value,'false');
  f.press('[data-select-sample="field_03"]');
  assert.equal(f.q(f.cell('field_01','threshold')).value,'15000','Values persist after table rerender');
  assert.equal(f.q(f.cell('field_01','above')).value,'false','False is not treated as blank');
  f.dispatch(f.cell('field_01','boundary'),'contextmenu',{clientX:200,clientY:150});
  assert.equal(f.q('#vbn-cell-menu').hidden,false);
  assert.match(f.q('#vbn-inherit-cell').textContent,/Use workflow value \(Reflect\)/);
  f.press('#vbn-inherit-cell');
  assert.equal(f.q(f.cell('field_01','boundary')).value,'');
  f.change(f.cell('field_01','above'),'');
  assert.equal(f.q(f.cell('field_01','above')).closest('td').classList.contains('is-inherited'),true);
});

await test('bulk editor is optional and edits only explicitly chosen parameters',()=>{
  const f=fixture();
  assert.match(f.q('#vbn-footer-text').textContent,/Checked/);
  f.press('#vbn-selected-overrides-button');
  assert.equal(f.q('#vbn-page-overrides').hidden,false);
  assert.notEqual(f.q('#vbn-bulk-dialog').open,true);
  f.check('[data-sample-check="field_01"]');
  assert.equal(f.q('#vbn-bulk-title').textContent,'2 selected');
  f.press('#vbn-edit-selected');
  assert.equal(f.q('#vbn-bulk-dialog').open,true);
  f.choose('[data-bulk-pick="threshold"]');
  assert.match(f.field('threshold').textContent,/Mixed values/);
  assert.equal(f.q('#vbn-apply-overrides').disabled,true,'Choosing a mixed field must not silently clear it');
  assert.match(f.q('#vbn-bulk-operation-threshold').textContent,/Enter a value or choose/);
  f.choose('[data-bulk-pick="threshold"]',false);
  f.stage('sigma','2.5');
  assert.equal(f.q('#vipp-batch-comparison').dataset.state,'ready','Staging must not dirty applied configuration');
  assert.equal(f.q(f.cell('field_01','sigma')).value,'');
  assert.match(f.q('#vbn-bulk-feedback').textContent,/ready to apply · all other values will be kept/);
  assert.equal(f.q('#vbn-next-button').disabled,true);
  f.press('[data-select-sample="field_03"]');
  assert.equal(f.q('#vbn-bulk-title').textContent,'2 selected','Staged target set cannot silently change');
  assert.match(f.q('#vbn-footer-text').textContent,/Apply or discard/);
  f.press('#vbn-apply-overrides');
  assert.equal(f.q('#vbn-bulk-dialog').open,false);
  assert.equal(f.q('#vipp-batch-comparison').dataset.state,'stale');
  assert.equal(f.q(f.cell('field_01','sigma')).value,'2.5');
  assert.equal(f.q(f.cell('field_02','sigma')).value,'2.5');
  assert.equal(f.q(f.cell('field_02','threshold')).value,'13000','Untouched threshold must be preserved');
  assert.equal(f.q(f.cell('field_03','sigma')).value,'','Unselected sample must be preserved');
  f.press('#vbn-edit-selected');f.choose('[data-bulk-pick="sigma"]');f.press('[data-bulk-inherit="sigma"]');f.press('#vbn-apply-overrides');
  assert.equal(f.q(f.cell('field_01','sigma')).value,'');
  assert.equal(f.q(f.cell('field_02','sigma')).value,'');
  assert.equal(f.q(f.cell('field_02','threshold')).value,'13000');
  f.press('#vbn-edit-selected');f.stage('threshold','9000');f.press('#vbn-discard-draft');
  assert.equal(f.q('#vbn-bulk-dialog').open,false);
  assert.equal(f.q(f.cell('field_02','threshold')).value,'13000');
  f.press('#vbn-edit-selected');assert.equal(f.q('#vbn-apply-overrides').disabled,true);f.press('#vbn-discard-draft');
});

await test('filters preserve selection; missing override means workflow inheritance',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');f.check('[data-sample-check="field_01"]');
  f.change('#vbn-override-filter','changed');
  assert.equal(f.qa('#vbn-sample-body tr').length,1);
  assert.equal(f.q('#vbn-bulk-title').textContent,'2 selected');
  assert.match(f.q('#vbn-hidden-selection').textContent,/1 hidden by filters · still included/);
  f.change('#vbn-override-filter','workflow');assert.equal(f.qa('#vbn-sample-body tr').length,2);
  f.change('#vbn-override-filter','missing');f.change('#vbn-missing-parameter','threshold');
  assert.equal(f.qa('#vbn-sample-body tr').length,2);
  assert.equal(f.document.querySelector('[data-select-sample="field_02"]'),null);
  f.change('#vbn-missing-parameter','sigma');assert.equal(f.qa('#vbn-sample-body tr').length,3);
  f.input('#vbn-override-search','no-match');
  assert.equal(f.q('#vbn-override-empty').hidden,false);
  assert.equal(f.q('#vbn-bulk-title').textContent,'2 selected');
  f.input('#vbn-override-search','green\\field_03.npy');
  assert.equal(f.qa('#vbn-sample-body tr').length,1);assert.match(f.q('#vbn-sample-body').textContent,/field_03/);
});

await test('1,200 samples use bounded pages, select-all-matching, and shift selection',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');f.change('#vbc-sample-count','1200');
  assert.equal(f.qa('#vbn-sample-body tr').length,50);assert.equal(f.qa('#vbn-plan-body tr').length,50);
  assert.equal(f.q('#vbn-sample-range').textContent,'1–50 of 1,200');
  f.press('#vbn-sample-next');assert.equal(f.q('#vbn-sample-range').textContent,'51–100 of 1,200');
  assert.equal(f.q('#vbn-bulk-title').textContent,'1 selected');
  f.press('#vbn-select-matching');assert.equal(f.q('#vbn-bulk-title').textContent,'1,200 selected');
  f.press('#vbn-edit-selected');f.stage('size','150');f.press('#vbn-apply-overrides');
  assert.match(f.q('#vbn-cell-feedback').textContent,/1,200 samples updated/);
  f.input('#vbn-override-search','field_1200');assert.equal(f.qa('#vbn-sample-body tr').length,1);
  assert.equal(f.q(f.cell('field_1200','size')).value,'150');
  assert.match(f.q('#vbn-hidden-selection').textContent,/1,199 hidden/);
  f.input('#vbn-override-search','');f.press('#vbn-clear-selection');f.press('[data-select-sample="field_01"]');
  f.check('[data-sample-check="field_05"]',true,true);
  assert.equal(f.q('#vbn-bulk-title').textContent,'5 selected');
  f.press('[data-new-section="items"]');assert.match(f.q('#vbn-selection-caption').textContent,/5 selected/);
  f.press('#vbn-selected-overrides-button');assert.equal(f.q('#vbn-bulk-title').textContent,'5 selected');
});

await test('sample pagination is hidden for one page and returns after clearing filters',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');
  const paginationHidden=expected=>{
    assert.equal(f.q('#vbn-sample-prev').hidden,expected,'Previous visibility follows the matched sample count');
    assert.equal(f.q('#vbn-sample-next').hidden,expected,'Next visibility follows the matched sample count');
  };
  paginationHidden(true);assert.equal(f.q('#vbn-sample-range').textContent,'1–3 of 3');
  f.input('#vbn-override-search','no-match');paginationHidden(true);
  assert.equal(f.q('#vbn-override-empty').hidden,false);
  f.input('#vbn-override-search','');paginationHidden(true);
  f.change('#vbc-sample-count','1200');paginationHidden(false);
  assert.equal(f.q('#vbn-sample-prev').disabled,true);
  assert.equal(f.q('#vbn-sample-next').disabled,false);
  f.press('#vbn-sample-next');assert.equal(f.q('#vbn-sample-range').textContent,'51–100 of 1,200');
  f.input('#vbn-override-search','field_1200');paginationHidden(true);
  assert.equal(f.q('#vbn-sample-range').textContent,'1–1 of 1');
  f.input('#vbn-override-search','');paginationHidden(false);
  assert.equal(f.q('#vbn-sample-range').textContent,'1–50 of 1,200');
  assert.equal(f.q('#vbn-sample-prev').disabled,true,'Clearing a narrowed filter returns to the first page');
  f.change('#vbc-sample-count','3');paginationHidden(true);
});

await test('column chooser supports nodes, individual columns, and search-to-reveal for 30 nodes',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');f.change('#vbc-node-count','30');
  assert.equal(f.qa('#vbn-parameter-table .vbn-node-row [scope="colgroup"]').length,30);
  assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,58);
  f.press('#vbn-columns-button');
  assert.equal(f.qa('[data-column-node]').length,30);assert.equal(f.qa('[data-column-key]').length,58);
  f.press('#vbn-hide-all-columns');assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,0);
  f.choose('[data-column-node="02"]');assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,2);
  f.choose('[data-column-key="above"]',false);assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,1);
  assert.equal(f.q('[data-column-node="02"]').indeterminate,true);
  f.input('#vbn-column-query','30');assert.equal(f.qa('[data-column-node]').length,1);
  f.close();f.input('#vbn-parameter-search','30');
  assert.equal(f.q('#vbn-parameter-results').hidden,false);
  f.press('[data-jump-param="n30_high"]');
  assert.ok(f.document.querySelector('#vbn-column-n30_high'),'Jump reveals a hidden column');
  assert.equal(f.q('#vbn-column-n30_high').classList.contains('is-found'),true);
  assert.equal(f.q('#vbn-parameter-results').hidden,true);
  assert.match(f.q('#vbn-cell-feedback').textContent,/30.*Upper percentile/);
  f.press('#vbn-columns-button');f.press('#vbn-show-all-columns');
  assert.equal(f.qa('#vbn-parameter-table .vbn-parameter-row th').length,58);
  f.close();f.change('#vbc-sample-count','1200');
  assert.equal(f.qa('#vbn-sample-body tr').length,50);
  assert.equal(f.qa('#vbn-sample-body [data-cell-input]').length,50*58,'Only current page has editors');
});

await test('invalid direct cell values persist across rendering and block checking and running',()=>{
  const f=fixture();f.press('[data-new-section="overrides"]');
  f.input(f.cell('field_02','threshold'),'-1');
  assert.equal(f.q(f.cell('field_02','threshold')).getAttribute('aria-invalid'),'true');
  assert.equal(f.q('#vbn-next-button').disabled,true);
  f.press('[data-select-sample="field_01"]');
  assert.equal(f.q(f.cell('field_02','threshold')).value,'-1');
  assert.equal(f.q('#vbn-edit-selected').disabled,true);
  f.press('[data-new-section="run"]');assert.equal(f.q('#vbn-next-button').disabled,true);
  f.press('[data-new-section="overrides"]');f.input(f.cell('field_02','threshold'),'12000.5');
  assert.equal(f.q(f.cell('field_02','threshold')).getAttribute('aria-invalid'),'true','Integer parameters reject decimals');
  f.input(f.cell('field_02','threshold'),'');
  assert.equal(f.q(f.cell('field_02','threshold')).getAttribute('aria-invalid'),'false');
  assert.equal(f.q('#vbn-next-button').disabled,false);
});

await test('source context menu chooses the source and reveals the exact file',()=>{
  const f=fixture();
  f.dispatch('[data-select-item="field_01"]','contextmenu',{clientX:200,clientY:150});
  assert.equal(f.q('#vbn-item-context-menu').hidden,false);
  assert.match(f.q('#vbn-item-context-menu [data-item-command="reveal"]').textContent,/Find in File Explorer/);
  f.press('#vbn-item-context-menu [data-item-command="reveal"]');
  assert.equal(f.qa('.vbn-dialog [data-reveal-source]').length,2);
  assert.match(f.q('.vbn-dialog').textContent,/Red source/);assert.match(f.q('.vbn-dialog').textContent,/Green source/);
  f.press('.vbn-dialog [data-reveal-source="green"]');
  assert.equal(f.q('#vbn-dialog-title').textContent,'File Explorer · file selected');
  assert.match(f.q('.vbn-dialog .vbn-file-preview').textContent,/green\\field_01\.npy$/);
  assert.match(f.q('.vbn-explorer-list .is-selected').textContent,/field_01.npy/);
  f.close();f.scenario('blocked');f.press('#vbn-item-context-menu [data-item-command="reveal"]');
  assert.equal(f.qa('.vbn-dialog [data-reveal-source]').length,1);
  assert.match(f.q('.vbn-dialog').textContent,/Missing file · cannot reveal/);
  const mac=fixture('MacIntel');
  assert.match(mac.q('#vbn-item-context-menu [data-item-command="reveal"]').textContent,/Find in Finder/);
});

await test('saved output filenames reveal exact files, unavailable files do not, and item links retain navigation',()=>{
  const f=fixture();f.scenario('complete');
  assert.equal(f.qa('#vbn-run-complete [data-reveal-output]').length,8);
  assert.equal(f.document.querySelector('#vbn-run-complete [data-reveal-output="field_03__measurements.tsv"]'),null);
  f.press('#vbn-run-complete [data-reveal-output="field_03__overlap.tif"]');
  assert.match(f.q('.vbn-dialog .vbn-file-preview').textContent,/results\\field_03__overlap\.tif$/);
  assert.match(f.q('.vbn-explorer-list .is-selected').textContent,/field_03__overlap.tif/);
  assert.equal(f.qa('.vbn-explorer-list li').length,2);
  f.close();f.press('[data-review-item="field_03"]');
  assert.equal(f.q('#vbn-page-items').hidden,false);assert.equal(f.q('#vbn-detail-item-name').textContent,'field_03');
  assert.match(f.q('#vbn-detail-outputs').textContent,/Not saved/);
  assert.equal(f.qa('#vbn-detail-outputs [data-reveal-output]').length,2);
  f.scenario('ready');assert.equal(f.qa('#vbn-detail-outputs [data-reveal-output]').length,0);
  assert.match(f.q('#vbn-detail-outputs').textContent,/Not created yet/);
});

await test('read-only source and saved-output reveals remain usable during a run',()=>{
  const f=fixture();f.scenario('running');
  assert.equal(f.q('#vbn-red-folder').disabled,true);assert.equal(f.q('#vbn-footer-progress').getAttribute('aria-valuenow'),'1');
  f.press('#vbn-run-running [data-reveal-output="field_01__combined.npy"]');
  assert.match(f.q('.vbn-dialog .vbn-file-preview').textContent,/field_01__combined.npy$/);f.close();
  f.press('[data-new-section="items"]');f.press('[data-select-item="field_02"]');
  f.press('#vbn-item-context-menu [data-item-command="reveal"]');f.press('.vbn-dialog [data-reveal-source="red"]');
  assert.match(f.q('.vbn-dialog .vbn-file-preview').textContent,/red\\field_02.npy$/);f.close();
  f.press('[data-new-section="overrides"]');assert.equal(f.q(f.cell('field_02','threshold')).disabled,true);assert.equal(f.q('#vbn-edit-selected').disabled,true);
});

await test('recheck, run, safe stop, partial retry, and missing-source recovery still work',async()=>{
  const f=fixture();f.press('[data-select-item="field_01"]');f.press('#vbn-selected-overrides-button');
  f.input(f.cell('field_01','threshold'),'15000');assert.equal(f.q('#vipp-batch-comparison').dataset.state,'stale');
  f.press('#vbn-next-button');await delay(650);assert.equal(f.q('#vipp-batch-comparison').dataset.state,'ready');
  assert.match(f.q('#vbn-detail-override').textContent,/15,000/);
  f.press('[data-new-section="run"]');f.press('#vbn-next-button');assert.equal(f.q('#vipp-batch-comparison').dataset.state,'running');
  f.press('#vbn-stop-button');await delay(850);assert.equal(f.q('#vipp-batch-comparison').dataset.state,'stopped');
  assert.match(f.q('#vbn-run-complete').textContent,/3 outputs saved/);
  f.scenario('complete');f.press('[data-review-item="field_03"]');f.press('#vbn-detail-recovery [data-action="retry"]');
  assert.match(f.q('.vbn-dialog').textContent,/2 output files already exist/);
  f.press('[data-dialog-action="retry"]');assert.equal(f.q('#vipp-batch-comparison').dataset.state,'success');
  f.scenario('blocked');f.press('[data-action="locate-source"]');f.press('[data-dialog-action="locate"]');await delay(650);
  assert.equal(f.q('#vipp-batch-comparison').dataset.state,'ready');
});

console.log(`\n${outcomes.filter(t=>t.passed).length}/${outcomes.length} interaction suites passed.`);
if(outcomes.some(t=>!t.passed))process.exitCode=1;
