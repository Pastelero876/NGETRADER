async function getSummaryKit(){
	try{
		const r = await fetch("/api/kit/ui/summary");
		if(!r.ok) return;
		const s = await r.json();
		const g = document.getElementById("global-semaforo");
		if(g){ g.textContent = String(s.viability_global||"--").toUpperCase(); g.className = `light ${s.viability_global||'gray'}`; }
		const list = document.getElementById("symbols-semaforo");
		if(list){
			list.innerHTML = "";
			const v = s.viability || {};
			Object.entries(v).forEach(([sym,val])=>{
				const li = document.createElement("li");
				const cost = ((val.fees_bps||0)+(val.half_spread_bps||0)+(val.buffer_bps||0)).toFixed(1);
				li.textContent = `${sym}: ${val.trade_viable?'OK':'KO'} (edge=${val.edge_bps} | cost=${cost})`;
				li.className = val.trade_viable ? "ok" : "ko";
				list.appendChild(li);
			});
		}
	}catch{}
}

async function getRiskKit(){
	try{
		const r = await fetch("/api/kit/risk"); const x = await r.json();
		const el = document.getElementById("risk-box");
		if(el){ el.textContent = `R/trade=${((x.risk_per_trade||0)*100).toFixed(2)}% | used_R=${((x.used_R||0)*100).toFixed(1)}% | budget_left=${((x.daily_budget_left||0)*100).toFixed(1)}%`; }
	}catch{}
}

async function getSLOKit(){
	try{
		const r = await fetch("/api/kit/slo"); const s = await r.json();
		const el = document.getElementById("slo-table");
		if(!el) return;
		el.innerHTML = "";
		const hdr = document.createElement("tr");
		["Símbolo","slippage_p95","p95_ms","gate","Fuente"].forEach(h=>{const th=document.createElement("th");th.textContent=h;hdr.appendChild(th);});
		el.appendChild(hdr);
		if(s.per_symbol){
			Object.entries(s.per_symbol).forEach(([sym,v])=>{
				const tr = document.createElement("tr");
				[sym, Number(v.slippage_bps||0).toFixed(1), Number(v.p95_ms||0).toFixed(0), v.gate ? "ON" : "OFF", String(v.source||"unknown").toUpperCase()].forEach(t=>{const td=document.createElement("td"); td.textContent=t; tr.appendChild(td);});
				el.appendChild(tr);
			});
		}
	}catch{}
}

function startKitAutoRefresh(){
	getSummaryKit(); getRiskKit(); getSLOKit();
	setInterval(()=>{ getSummaryKit(); getRiskKit(); getSLOKit(); }, 10000);
}

document.addEventListener("DOMContentLoaded", startKitAutoRefresh);


