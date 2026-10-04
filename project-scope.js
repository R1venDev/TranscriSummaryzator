'use strict';
(() => {
  const selected = document.querySelector('meta[name="transcri-project"]')?.content || 'default';
  const urlFor = raw => {
    const u = new URL(raw, location.href);
    if (u.origin === location.origin) u.searchParams.set('project', selected);
    return u;
  };
  const originalFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const u = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (u.origin !== location.origin) return originalFetch(input, init);
    const headers = new Headers(init.headers || (input instanceof Request ? input.headers : undefined));
    headers.set('X-Transcri-Project', selected);
    if ((init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase() === 'POST') headers.set('X-Requested-With','TranscriSummaryzator-Admin');
    return originalFetch(input, {...init, headers});
  };
  const originalOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    const u = new URL(url, location.href);
    return originalOpen.call(this, method, u.origin === location.origin ? String(urlFor(url)) : url, ...rest);
  };
  function ready() {
  function links() {
    document.querySelectorAll('audio[src]').forEach(a => {
      const u = new URL(a.src, location.href);
      if (u.origin === location.origin && u.pathname === '/api/profiles/audio') a.src = String(urlFor(a.src));
    });
    document.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href');
      if (!href || href.startsWith('#') || a.dataset.projectLink) return;
      const u = new URL(href, location.href);
      if (u.origin === location.origin) a.href = String(urlFor(href));
    });
  }
  links();
  new MutationObserver(links).observe(document.body, {childList:true, subtree:true});
  const main = document.querySelector('main');
  if (!main) return;
  const bar = document.createElement('nav');
  bar.style.cssText = 'display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:26px;padding:16px 20px;border:1px solid #394354;border-radius:14px;background:#171a22;color:#dce8f8';
  const label = document.createElement('label'); label.textContent = 'Профиль проекта ';
  const select = document.createElement('select'); select.setAttribute('aria-label','Профиль проекта');
  select.style.cssText = 'margin-left:10px;padding:8px 12px;background:#202530;color:white;border:1px solid #46526a;border-radius:8px;font:inherit';
  label.append(select);
  const manage = document.createElement('a'); manage.textContent = 'Управление профилями'; manage.href = '/project-folders?project='+encodeURIComponent(selected); manage.style.color = '#b8d9ff';
  const keys = document.createElement('a'); keys.textContent = 'Ключи моделей'; keys.href = '/summary-settings?project='+encodeURIComponent(selected); keys.style.color = '#b8d9ff';
  const plane = document.createElement('a'); plane.textContent = 'Plane'; plane.href = '/plane-settings?project='+encodeURIComponent(selected); plane.style.color = '#b8d9ff';
  bar.append(label,manage,keys,plane); main.prepend(bar);
  originalFetch('/api/projects?project='+encodeURIComponent(selected), {cache:'no-store'}).then(r=>r.json()).then(data=> {
    if (!Array.isArray(data.projects)) throw Error('Профили недоступны');
    const current=data.projects.find(p=>p.id===selected);
    const rename=document.querySelector('#project-rename');
    if(rename && current) {
      rename.elements.name.value=current.name;
      rename.addEventListener('submit',async event=>{
        event.preventDefault();const button=rename.querySelector('button');button.disabled=true;
        try {const r=await window.fetch('/api/projects/rename',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'TranscriSummaryzator-Admin'},body:JSON.stringify({name:rename.elements.name.value,expected_revision:current.revision})});const d=await r.json();if(!r.ok)throw Error(d.error);location.reload();}
        catch(e){document.querySelector('#rename-message').textContent=e.message;button.disabled=false;}
      });
    }
    data.projects.forEach(p=>{ const o=document.createElement('option'); o.value=p.id; o.textContent=p.name; o.selected=p.id===selected; select.append(o); });
  }).catch(()=> {select.disabled=true; label.textContent='Не удалось загрузить профили';});
  select.addEventListener('change',()=>{
    // Move to a profile's library instead of reusing another profile's job ID.
    location.href = '/?project='+encodeURIComponent(select.value);
  });
  const create = document.querySelector('#project-create');
  if (create) create.addEventListener('submit', async event=> {
    event.preventDefault(); const button=create.querySelector('button'); button.disabled=true;
    const message=document.querySelector('#project-message');
    try {
      const r=await window.fetch('/api/projects/create',{method:'POST',headers:{'Content-Type':'application/json','X-Requested-With':'TranscriSummaryzator-Admin'},body:JSON.stringify({name:create.elements.name.value})});
      const d=await r.json(); if(!r.ok) throw Error(d.error||'Не удалось создать профиль');
      location.href='/?project='+encodeURIComponent(d.project.id);
    } catch(e) {message.textContent=e.message; button.disabled=false;}
  });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready); else ready();
})();
