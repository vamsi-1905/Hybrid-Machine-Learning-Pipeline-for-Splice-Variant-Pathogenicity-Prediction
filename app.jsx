import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

/* ═══════════════════════════════════════════════════════════
   SPLICE VARIANT LAB  —  Scientific Instrument Aesthetic
   Deep void · Phosphor amber · Sequence green
═══════════════════════════════════════════════════════════ */

const C = {
  void:      "#04050a",
  abyss:     "#07090f",
  deep:      "#0b0e18",
  surface:   "#0e1220",
  raised:    "#131828",
  border:    "#1a2135",
  dim:       "#24304a",
  muted:     "#3a4f6a",
  ghost:     "#58718a",
  slate:     "#7a90a8",
  silver:    "#a8bcd0",
  cloud:     "#d4e0ec",
  white:     "#edf4fc",
  amber:     "#e8a030",
  amberDim:  "#e8a03022",
  seq:       "#22d97a",
  seqDim:    "#22d97a18",
  red:       "#e05050",
  redDim:    "#e0505020",
  teal:      "#20b8d0",
  tealDim:   "#20b8d018",
  purple:    "#9070e8",
  purpleDim: "#9070e820",
  orange:    "#e87040",
};

const F = {
  display: "'EB Garamond','Playfair Display',Georgia,serif",
  mono:    "'Fira Code','Cascadia Code','SF Mono',monospace",
  sans:    "'DM Sans','Outfit',system-ui,sans-serif",
};

const MODES = [
  { id:"xgb",          label:"XGBoost",      sub:"k-mer",        icon:"▦", color:C.amber  },
  { id:"dnabert",      label:"DNABERT-2",    sub:"sequence",     icon:"◈", color:C.seq    },
  { id:"ensemble",     label:"Ensemble",     sub:"fusion",       icon:"⊕", color:C.teal   },
  { id:"splice_sites", label:"Splice Sites", sub:"PWM+ESE+BP",   icon:"⌁", color:C.purple },
];

const EX = {
  chrom:"1", position:"925952", ref:"G", alt:"A",
  ref_seq:"AGCTGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

/* ── tiny helpers ─────────────────────────────────────────── */
const clamp  = (v,mn,mx) => Math.min(Math.max(v,mn),mx);
const baseC  = b => ({A:C.red,T:C.amber,G:C.seq,C:C.teal}[b?.toUpperCase()] ?? C.ghost);
const sigC   = s => s?.startsWith("Strong") ? C.red : s?.startsWith("Moderate") ? C.amber : s?.startsWith("Low") ? C.teal : C.seq;
const sevC   = s => ({critical:C.red,high:C.amber,moderate:C.orange,low:C.teal,minimal:C.ghost}[s] ?? C.ghost);
const mutC   = t => ({SNV:C.amber,Deletion:C.red,Insertion:C.seq,MNV:C.purple}[t] ?? C.ghost);
const pathC  = s => ({high:C.red,moderate:C.amber,low:C.teal,minimal:C.ghost}[s] ?? C.ghost);

/* ── animated counter ─────────────────────────────────────── */
function Ticker({ to, dur=900, dec=0 }) {
  const [v, sv] = useState(0);
  useEffect(() => {
    const s = Date.now();
    const f = () => { const p=Math.min((Date.now()-s)/dur,1); sv(to*(1-Math.pow(1-p,4))); p<1&&requestAnimationFrame(f); };
    requestAnimationFrame(f);
  }, [to, dur]);
  return <>{v.toFixed(dec)}</>;
}

/* ── bar ──────────────────────────────────────────────────── */
function Bar({ val, max=1, color=C.amber, h=3, delay=0, label }) {
  const [w, sw] = useState(0);
  useEffect(() => { const t=setTimeout(()=>sw(clamp(val/max,0,1)*100),80+delay); return ()=>clearTimeout(t); },[val,max,delay]);
  return (
    <div>
      {label && (
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:3}}>
          <span style={{fontFamily:F.mono,fontSize:8,color:C.ghost,letterSpacing:1}}>{label}</span>
          <span style={{fontFamily:F.mono,fontSize:9,color,fontWeight:600}}>{(val*100).toFixed(1)}%</span>
        </div>
      )}
      <div style={{height:h,background:C.dim,borderRadius:1,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${w}%`,borderRadius:1,
          background:`linear-gradient(90deg,${color}99,${color})`,
          boxShadow:`0 0 8px ${color}55`,
          transition:`width 1s cubic-bezier(.16,1,.3,1) ${delay}ms`}}/>
      </div>
    </div>
  );
}

/* ── tag ──────────────────────────────────────────────────── */
function Tag({ label, color=C.ghost, small }) {
  return (
    <span style={{fontFamily:F.mono,fontSize:small?7:8,letterSpacing:1.5,fontWeight:700,
      textTransform:"uppercase",padding:small?"2px 6px":"3px 9px",borderRadius:2,
      background:`${color}18`,border:`1px solid ${color}44`,color,lineHeight:1,whiteSpace:"nowrap"}}>
      {label}
    </span>
  );
}

/* ── section rule ─────────────────────────────────────────── */
function Sep({ label, color=C.muted }) {
  return (
    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:12}}>
      <div style={{height:1,width:12,background:C.border}}/>
      <span style={{fontFamily:F.mono,fontSize:7,letterSpacing:3,color,textTransform:"uppercase"}}>{label}</span>
      <div style={{height:1,flex:1,background:C.border}}/>
    </div>
  );
}

/* ── circular gauge ───────────────────────────────────────── */
function Gauge({ prob, isPath, size=108 }) {
  const R=42, circ=2*Math.PI*R, color=isPath?C.red:C.seq;
  const [dash,sd]=useState(circ);
  useEffect(()=>{ const t=setTimeout(()=>sd(circ*(1-prob)),120); return()=>clearTimeout(t); },[prob]);
  return (
    <div style={{position:"relative",width:size,height:size,flexShrink:0}}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
           style={{position:"absolute",inset:0,transform:"rotate(-90deg)"}}>
        <circle cx={size/2} cy={size/2} r={R} fill="none" stroke={C.dim} strokeWidth={6}/>
        <circle cx={size/2} cy={size/2} r={R} fill="none" stroke={color} strokeWidth={6}
          strokeDasharray={circ} strokeDashoffset={dash} strokeLinecap="round"
          style={{transition:"stroke-dashoffset 1.1s cubic-bezier(.16,1,.3,1)",filter:`drop-shadow(0 0 4px ${color})`}}/>
      </svg>
      <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center"}}>
        <span style={{fontFamily:F.mono,fontWeight:700,fontSize:18,color,lineHeight:1,textShadow:`0 0 20px ${color}88`}}>
          <Ticker to={Math.round(prob*100)}/><span style={{fontSize:11}}>%</span>
        </span>
        <span style={{fontFamily:F.mono,fontSize:6,color:C.ghost,letterSpacing:2,marginTop:2}}>PROB</span>
      </div>
    </div>
  );
}

/* ── delta badge ──────────────────────────────────────────── */
function Delta({ v }) {
  const col = v > 0.05?C.red : v < -0.05?C.seq : C.ghost;
  return (
    <span style={{fontFamily:F.mono,fontSize:10,fontWeight:700,color:col,
      background:`${col}18`,border:`1px solid ${col}44`,borderRadius:2,padding:"2px 7px"}}>
      {v>=0?"+":""}{v.toFixed(3)}
    </span>
  );
}

/* ── k-mer diff ───────────────────────────────────────────── */
function KmerDiff({ ref_kmer, alt_kmer }) {
  if (!ref_kmer && !alt_kmer) return null;
  const render = (seq, other, isAlt) => {
    if (!seq) return <span style={{color:C.muted,fontFamily:F.mono,fontSize:11}}>—</span>;
    return (
      <span style={{fontFamily:F.mono,fontSize:11,letterSpacing:2.5}}>
        {seq.split("").map((c,i)=>{
          const diff=other&&other[i]!==c;
          return <span key={i} style={{color:diff&&isAlt?"#000":baseC(c),background:diff&&isAlt?baseC(c):"transparent",borderRadius:1,padding:"0 1px"}}>{c}</span>;
        })}
      </span>
    );
  };
  return (
    <div style={{background:C.abyss,borderRadius:4,padding:"10px 14px",border:`1px solid ${C.border}`,marginTop:8}}>
      {[["REF",ref_kmer,false],["ALT",alt_kmer,true]].map(([lbl,seq,isAlt])=>(
        <div key={lbl} style={{display:"flex",gap:12,alignItems:"center",marginBottom:lbl==="REF"?5:0}}>
          <span style={{fontFamily:F.mono,fontSize:7,color:C.ghost,width:24,letterSpacing:1}}>{lbl}</span>
          {render(seq,lbl==="REF"?alt_kmer:ref_kmer,isAlt)}
        </div>
      ))}
      {ref_kmer&&alt_kmer&&ref_kmer!==alt_kmer&&(
        <div style={{marginTop:6,fontFamily:F.mono,fontSize:8,color:C.amber}}>
          ↑ Highlighted bases changed — altered spliceosome recognition context
        </div>
      )}
    </div>
  );
}

/* ── DNA strip ────────────────────────────────────────────── */
function DNAStrip({ seq, highlight, maxLen=80 }) {
  if (!seq) return null;
  const display = seq.slice(0,maxLen);
  return (
    <div style={{fontFamily:F.mono,fontSize:10,letterSpacing:1,lineHeight:1.6,wordBreak:"break-all"}}>
      {display.split("").map((b,i)=>{
        const isHL=highlight&&i>=highlight.start&&i<highlight.end;
        return <span key={i} style={{color:isHL?"#000":baseC(b),background:isHL?baseC(b):"transparent",borderRadius:1}}>{b}</span>;
      })}
      {seq.length>maxLen&&<span style={{color:C.muted}}>…+{seq.length-maxLen}bp</span>}
    </div>
  );
}

/* ── site landscape ───────────────────────────────────────── */
function SiteLandscape({ sites }) {
  if (!sites?.length) return null;
  const allPos=sites.map(s=>s.position), minP=Math.min(...allPos), maxP=Math.max(...allPos), span=Math.max(maxP-minP,1);
  const Track=({items,label})=>{
    if(!items.length) return null;
    return (
      <div style={{marginBottom:10}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
          <span style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:1.5}}>{label}</span>
          <div style={{display:"flex",gap:8}}>
            {[["DISRUPT",C.red],["CREATED",C.seq],["CRYPTIC",C.purple]].map(([l,c])=>(
              <span key={l} style={{display:"flex",alignItems:"center",gap:3,fontFamily:F.mono,fontSize:6,color:C.ghost}}>
                <span style={{width:5,height:5,background:c,display:"inline-block",borderRadius:1}}/>{l}
              </span>
            ))}
          </div>
        </div>
        <div style={{position:"relative",height:28,background:C.abyss,border:`1px solid ${C.border}`,borderRadius:3,overflow:"hidden"}}>
          <div style={{position:"absolute",left:"50%",top:0,width:1,height:"100%",background:C.dim}}/>
          {items.map((s,i)=>{
            const x=span>0?((s.position-minP)/span)*86+7:50;
            const col=s.disrupted?C.red:s.created?C.seq:s.cryptic?C.purple:C.ghost;
            const ht=Math.max(s.alt_score*26,3);
            return <div key={i} title={`${s.type} pos ${s.position>=0?"+":""}${s.position}  Δ${s.delta.toFixed(3)}`}
              style={{position:"absolute",bottom:0,left:`${x}%`,width:3,height:`${ht}px`,background:col,
                borderRadius:"1px 1px 0 0",transform:"translateX(-50%)",boxShadow:`0 0 6px ${col}88`}}/>;
          })}
        </div>
        <div style={{display:"flex",justifyContent:"space-between",marginTop:2}}>
          <span style={{fontFamily:F.mono,fontSize:6,color:C.muted}}>{minP>=0?"+":""}{minP}bp</span>
          <span style={{fontFamily:F.mono,fontSize:6,color:C.amber}}>▼ variant</span>
          <span style={{fontFamily:F.mono,fontSize:6,color:C.muted}}>{maxP>=0?"+":""}{maxP}bp</span>
        </div>
      </div>
    );
  };
  return (
    <div style={{marginBottom:16,padding:"12px 14px",background:C.deep,borderRadius:5,border:`1px solid ${C.border}`}}>
      <Sep label="Splice Site Landscape · mini genome view"/>
      <Track items={sites.filter(s=>s.type==="donor")}    label="5′ Donor sites"/>
      <Track items={sites.filter(s=>s.type==="acceptor")} label="3′ Acceptor sites"/>
    </div>
  );
}

/* ── ESE panel ────────────────────────────────────────────── */
function ESEPanel({ findings }) {
  if (!findings?.length) return null;
  return (
    <div style={{marginBottom:14}}>
      <Sep label="Exonic Splicing Enhancers (ESE) — SR Protein Binding"/>
      <div style={{fontFamily:F.mono,fontSize:8,color:C.ghost,lineHeight:1.8,marginBottom:10}}>
        SR proteins (SRSF1–5) bind ESE hexamers within exons to recruit U2AF and stabilise splice site selection.
        Disruption causes exon skipping; de novo ESE may activate cryptic splicing.
      </div>
      {findings.map((f,i)=>(
        <div key={i} style={{marginBottom:5,padding:"9px 12px",
          background:f.type==="ESE_disrupted"?C.redDim:C.seqDim,
          border:`1px solid ${f.type==="ESE_disrupted"?C.red+"44":C.seq+"44"}`,borderRadius:3}}>
          <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:4}}>
            <Tag label={f.type==="ESE_disrupted"?"ESE LOST":"ESE GAINED"} color={f.type==="ESE_disrupted"?C.red:C.seq} small/>
            <span style={{fontFamily:F.mono,fontSize:9,color:C.amber,letterSpacing:1}}>{f.protein}</span>
            <span style={{fontFamily:F.mono,fontSize:10,color:C.cloud,letterSpacing:2}}>{f.motif}</span>
          </div>
          <div style={{fontFamily:F.mono,fontSize:9,color:C.slate,lineHeight:1.7}}>{f.effect}</div>
        </div>
      ))}
    </div>
  );
}

/* ── Branch point panel ───────────────────────────────────── */
function BranchPointPanel({ ref_bp, alt_bp }) {
  if (!ref_bp?.length && !alt_bp?.length) return null;
  const BPEntry=({bp,label,color})=>(
    <div style={{flex:1}}>
      <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:1,marginBottom:6}}>{label}</div>
      {!bp?.length
        ? <div style={{fontFamily:F.mono,fontSize:9,color:C.muted}}>None found</div>
        : bp.map((b,i)=>(
          <div key={i} style={{marginBottom:4,padding:"7px 10px",background:`${color}10`,border:`1px solid ${color}33`,borderRadius:3}}>
            <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:3}}>
              <span style={{fontFamily:F.mono,fontSize:10,color,letterSpacing:2}}>{b.seq}</span>
              <Tag label={`pos ${b.pos>=0?"+":""}${b.pos}`} color={color} small/>
              <span style={{fontFamily:F.mono,fontSize:8,color:C.ghost}}>score {b.score.toFixed(2)}</span>
            </div>
            <div style={{fontFamily:F.mono,fontSize:8,color:C.ghost}}>
              Branch A at {b.branch_A>=0?"+":""}{b.branch_A} · YNYURAY consensus
            </div>
          </div>
        ))
      }
    </div>
  );
  return (
    <div style={{marginBottom:14}}>
      <Sep label="Branch Point Adenosine Candidates (18–40nt upstream of acceptor)"/>
      <div style={{fontFamily:F.mono,fontSize:8,color:C.ghost,lineHeight:1.8,marginBottom:8}}>
        The branch point A nucleophilically attacks the 5′ splice site phosphate in step 1 of splicing.
        Consensus: YNYURAY. Mutations here impair lariat formation.
      </div>
      <div style={{display:"flex",gap:10}}>
        <BPEntry bp={ref_bp||[]} label="REF sequence" color={C.seq}/>
        <div style={{width:1,background:C.border}}/>
        <BPEntry bp={alt_bp||[]} label="ALT sequence" color={C.amber}/>
      </div>
    </div>
  );
}

/* ── individual site row ──────────────────────────────────── */
function SiteRow({ s, idx }) {
  const [open,setOpen]=useState(false);
  const tag  = s.disrupted?"DISRUPTED":s.created?"DE NOVO":s.cryptic?"CRYPTIC":"SHIFTED";
  const col  = s.disrupted?C.red:s.created?C.seq:s.cryptic?C.purple:C.muted;
  const pos  = `${s.position>=0?"+":""}${s.position}`;
  return (
    <div style={{background:open?C.raised:C.surface,border:`1px solid ${open?col+"55":C.border}`,borderRadius:5,overflow:"hidden",
      animation:`siteIn 0.25s ${idx*30}ms both`,transition:"border-color 0.2s,background 0.2s"}}>
      <div onClick={()=>setOpen(v=>!v)} style={{padding:"11px 16px",cursor:"pointer",
        display:"grid",gridTemplateColumns:"110px 1fr 130px 28px",gap:12,alignItems:"center"}}>
        <div>
          <div style={{display:"flex",gap:5,alignItems:"center",marginBottom:4}}>
            <Tag label={tag} color={col} small/>
            {s.canonical&&<Tag label="CANONICAL" color={C.amber} small/>}
          </div>
          <div style={{fontFamily:F.mono,fontSize:7,color:C.muted,lineHeight:1.6}}>
            {s.type} · {pos} · {s.dinucleotide||"—"}
          </div>
        </div>
        <div>
          {[["REF",s.ref_score,C.ghost],["ALT",s.alt_score,col]].map(([lbl,v,c])=>(
            <div key={lbl} style={{display:"flex",gap:8,alignItems:"center",marginBottom:lbl==="REF"?3:0}}>
              <span style={{fontFamily:F.mono,fontSize:7,color:c,width:20}}>{lbl}</span>
              <div style={{flex:1,height:3,background:C.dim,borderRadius:1,overflow:"hidden"}}>
                <div style={{height:"100%",width:`${v*100}%`,background:c,borderRadius:1,boxShadow:lbl==="ALT"?`0 0 4px ${c}88`:undefined}}/>
              </div>
              <span style={{fontFamily:F.mono,fontSize:8,color:c,width:32,textAlign:"right"}}>{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
        <div style={{display:"flex",justifyContent:"flex-end"}}><Delta v={s.delta}/></div>
        <span style={{fontFamily:F.mono,fontSize:9,color:C.muted,textAlign:"center"}}>{open?"▲":"▼"}</span>
      </div>
      {open&&(
        <div style={{borderTop:`1px solid ${C.border}`,padding:"14px 16px",animation:"fadeIn 0.18s ease"}}>
          <KmerDiff ref_kmer={s.ref_kmer} alt_kmer={s.alt_kmer}/>
          <div style={{marginTop:12,padding:"12px 14px",background:C.abyss,borderRadius:4,borderLeft:`2px solid ${col}`}}>
            <Sep label="Molecular reasoning"/>
            <p style={{fontFamily:F.mono,fontSize:10,color:C.silver,lineHeight:1.95,margin:0}}>{s.reasoning}</p>
          </div>
          <div style={{marginTop:10,display:"flex",gap:8,flexWrap:"wrap"}}>
            <Tag label={s.bio_consequence} color={col} small/>
            <Tag label={`pathogenicity: ${s.pathogenicity_contribution}`} color={pathC(s.pathogenicity_contribution)} small/>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═════════════════════════════════════════════════════════════
   PREDICTION CARD  (XGB / DNABERT / Ensemble)
═════════════════════════════════════════════════════════════ */
function PredCard({ r, onClose }) {
  const isPath = r.prediction==="Pathogenic";
  const color  = isPath?C.red:C.seq;
  const [tab,setTab]=useState("overview");

  return (
    <div style={{background:C.surface,border:`1px solid ${color}40`,borderRadius:8,overflow:"hidden",
      animation:"cardIn 0.5s cubic-bezier(.16,1,.3,1)",boxShadow:`0 0 80px ${color}0c,0 4px 40px #00000088`}}>
      <div style={{height:3,background:`linear-gradient(90deg,transparent,${color},transparent)`}}/>
      {/* header */}
      <div style={{padding:"20px 24px",background:`linear-gradient(135deg,${color}14 0%,transparent 60%)`,
        borderBottom:`1px solid ${color}22`,display:"flex",alignItems:"center",gap:16}}>
        <div style={{flex:1}}>
          <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:3,marginBottom:6}}>
            {r.model?.toUpperCase()} · {r.latency_ms}ms
          </div>
          <div style={{display:"flex",alignItems:"baseline",gap:12,marginBottom:8}}>
            <span style={{fontFamily:F.display,fontSize:38,color,lineHeight:1,fontStyle:"italic"}}>{r.prediction}</span>
            <Tag label={r.confidence} color={color}/>
            <Tag label={r.mutation_severity?.toUpperCase()||""} color={sevC(r.mutation_severity)}/>
          </div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            <Tag label={r.mutation_type} color={mutC(r.mutation_type)}/>
            <Tag label={`chr${r.chrom}:${r.position}`} color={C.ghost}/>
            <Tag label={`${r.ref}→${r.alt}`} color={C.amber}/>
          </div>
        </div>
        <Gauge prob={r.probability} isPath={isPath}/>
        <button onClick={onClose} style={{background:"none",border:`1px solid ${C.border}`,borderRadius:4,
          color:C.ghost,cursor:"pointer",width:28,height:28,fontFamily:F.mono,fontSize:11,alignSelf:"flex-start",flexShrink:0}}>✕</button>
      </div>

      <div style={{padding:"20px 24px"}}>
        {/* prob spectrum */}
        <div style={{marginBottom:18}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
            <span style={{fontFamily:F.mono,fontSize:7,color:C.seq,letterSpacing:1}}>← BENIGN</span>
            <span style={{fontFamily:F.mono,fontSize:7,color:C.ghost}}>threshold {(r.threshold_used*100).toFixed(0)}%</span>
            <span style={{fontFamily:F.mono,fontSize:7,color:C.red,letterSpacing:1}}>PATHOGENIC →</span>
          </div>
          <div style={{position:"relative",height:8,background:C.dim,borderRadius:4}}>
            <div style={{position:"absolute",inset:0,borderRadius:4,
              background:`linear-gradient(90deg,${C.seq}66 0%,${C.teal}44 40%,${C.amber}44 60%,${C.red}88 100%)`,opacity:0.4}}/>
            <div style={{position:"absolute",top:"50%",transform:"translate(-50%,-50%)",left:`${r.probability*100}%`,
              width:12,height:12,borderRadius:"50%",background:color,border:"2px solid #000",
              boxShadow:`0 0 12px ${color}`,transition:"left 0.9s cubic-bezier(.16,1,.3,1)"}}/>
            <div style={{position:"absolute",top:-3,left:`${r.threshold_used*100}%`,transform:"translateX(-50%)",
              width:1,height:14,background:C.amber,boxShadow:`0 0 6px ${C.amber}`}}/>
          </div>
          <div style={{display:"flex",justifyContent:"space-between",marginTop:3}}>
            <span style={{fontFamily:F.mono,fontSize:7,color:C.muted}}>0%</span>
            <span style={{fontFamily:F.mono,fontSize:7,color:C.muted}}>100%</span>
          </div>
        </div>

        {/* tabs */}
        <div style={{display:"flex",borderBottom:`1px solid ${C.border}`,marginBottom:18}}>
          {["overview","mechanism","model"].map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{background:"none",border:"none",cursor:"pointer",
              fontFamily:F.mono,fontSize:8,letterSpacing:1.5,textTransform:"uppercase",
              color:tab===t?color:C.ghost,borderBottom:tab===t?`2px solid ${color}`:"2px solid transparent",
              padding:"8px 16px",marginBottom:-1,transition:"color 0.2s"}}>{t}</button>
          ))}
        </div>

        {tab==="overview"&&(
          <div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:6,marginBottom:16}}>
              {[["CHROMOSOME",`chr${r.chrom}`,C.teal],["POSITION",r.position?.toLocaleString(),C.silver],
                ["REF → ALT",`${r.ref} → ${r.alt}`,C.amber],["MODEL",r.model?.split(" ")[0],C.ghost],
                ["PROBABILITY",`${(r.probability*100).toFixed(2)}%`,color],["LATENCY",`${r.latency_ms}ms`,C.ghost],
              ].map(([k,v,c])=>(
                <div key={k} style={{background:C.deep,borderRadius:4,padding:"10px 13px",border:`1px solid ${C.border}`}}>
                  <div style={{fontFamily:F.mono,fontSize:6,color:C.muted,letterSpacing:1.5,marginBottom:4}}>{k}</div>
                  <div style={{fontFamily:F.mono,fontSize:12,fontWeight:700,color:c}}>{v}</div>
                </div>
              ))}
            </div>
            <div style={{padding:"12px 14px",background:C.deep,borderRadius:5,border:`1px solid ${C.border}`}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:10}}>
                <div>
                  <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:2,marginBottom:3}}>CONFIDENCE</div>
                  <div style={{fontFamily:F.display,fontSize:24,color,fontStyle:"italic"}}>{r.confidence}</div>
                </div>
                <div style={{textAlign:"right"}}>
                  <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:2,marginBottom:3}}>MARGIN</div>
                  <div style={{fontFamily:F.mono,fontSize:20,fontWeight:700,color:C.amber}}>
                    {(Math.abs(r.probability-r.threshold_used)*100).toFixed(1)}pp
                  </div>
                </div>
              </div>
              <Bar val={Math.abs(r.probability-r.threshold_used)} max={0.5} color={C.amber} h={4} label="DISTANCE FROM DECISION BOUNDARY"/>
              <div style={{marginTop:8,fontFamily:F.mono,fontSize:8,color:C.ghost,lineHeight:1.8}}>
                {Math.abs(r.probability-r.threshold_used)<0.1
                  ?"⚠ Near-boundary — treat with caution. Run splice site analysis for confirmation."
                  :Math.abs(r.probability-r.threshold_used)<0.3
                  ?"Moderate separation. Ensemble model will provide additional confidence."
                  :"Strong separation from decision boundary — high model certainty."}
              </div>
            </div>
          </div>
        )}

        {tab==="mechanism"&&(
          <div style={{padding:"14px 16px",background:C.deep,borderRadius:5,borderLeft:`3px solid ${sevC(r.mutation_severity)}`}}>
            <div style={{display:"flex",gap:8,marginBottom:10}}>
              <Tag label={r.mutation_type} color={mutC(r.mutation_type)}/>
              <Tag label={`severity: ${r.mutation_severity}`} color={sevC(r.mutation_severity)}/>
            </div>
            <Sep label="Molecular mechanism"/>
            <p style={{fontFamily:F.mono,fontSize:10,color:C.silver,lineHeight:2,margin:0}}>{r.mechanism}</p>
          </div>
        )}

        {tab==="model"&&(
          <div>
            <Sep label="How the model processed this variant"/>
            <div style={{fontFamily:F.mono,fontSize:9,color:C.ghost,lineHeight:2,marginBottom:12}}>
              {(r.model?.includes("XGBoost")||r.model?.includes("Ensemble"))&&(
                <><span style={{color:C.amber}}>XGBoost k-mer features: </span>
                The 200bp reference and alternate sequences are decomposed into overlapping 6-mers.
                Frequency vectors of 4,096 dimensions are computed, plus direction, magnitude,
                variant type and biological flags — 12,301 features total.
                XGBoost learned which k-mer frequency shifts predict pathogenicity from ~60k balanced variants.<br/><br/></>
              )}
              {(r.model?.includes("DNABERT")||r.model?.includes("Ensemble"))&&(
                <><span style={{color:C.seq}}>DNABERT-2 sequence encoding: </span>
                The reference and alternate 200bp windows are tokenised with Byte Pair Encoding
                and passed through a 117M-parameter bidirectional transformer pretrained on the human genome.
                The [CLS] token embedding captures global sequence context attending to conserved splice signals,
                ESE motifs, and local sequence grammar.</>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═════════════════════════════════════════════════════════════
   SPLICE SITES CARD
═════════════════════════════════════════════════════════════ */
function SpliceCard({ r, onClose }) {
  const sc=sigC(r.pathogenicity_signal);
  const [showAll,setShowAll]=useState(false);
  const [tab,setTab]=useState("sites");
  const visible=showAll?r.sites:(r.sites||[]).slice(0,6);
  const donorNet=(r.alt_active_donors||0)-(r.ref_active_donors||0);
  const accNet  =(r.alt_active_acceptors||0)-(r.ref_active_acceptors||0);

  return (
    <div style={{background:C.surface,border:`1px solid ${sc}40`,borderRadius:8,overflow:"hidden",
      animation:"cardIn 0.5s cubic-bezier(.16,1,.3,1)",boxShadow:`0 0 80px ${sc}0c,0 4px 40px #00000088`}}>
      <div style={{height:3,background:`linear-gradient(90deg,transparent,${sc},transparent)`}}/>

      {/* header */}
      <div style={{padding:"20px 24px",background:`linear-gradient(135deg,${sc}14 0%,transparent 60%)`,
        borderBottom:`1px solid ${sc}22`,display:"flex",alignItems:"flex-start",gap:12}}>
        <div style={{flex:1}}>
          <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:3,marginBottom:6}}>
            SPLICE SITE ANALYSIS · {r.latency_ms}ms · {r.sites_found} sites evaluated
          </div>
          <div style={{fontFamily:F.display,fontSize:22,color:sc,lineHeight:1.2,marginBottom:8,fontStyle:"italic"}}>
            {r.pathogenicity_signal}
          </div>
          <div style={{fontFamily:F.mono,fontSize:9,color:C.slate,marginBottom:10}}>{r.summary}</div>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            <Tag label={r.mutation_type} color={mutC(r.mutation_type)}/>
            <Tag label={r.mutation_detail} color={C.ghost}/>
            <Tag label={`severity: ${r.mutation_severity}`} color={sevC(r.mutation_severity)}/>
          </div>
        </div>
        <button onClick={onClose} style={{background:"none",border:`1px solid ${C.border}`,borderRadius:4,
          color:C.ghost,cursor:"pointer",width:28,height:28,fontFamily:F.mono,fontSize:11,flexShrink:0}}>✕</button>
      </div>

      <div style={{padding:"20px 24px"}}>
        {/* active site counters */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:8,marginBottom:16}}>
          {[["Active Donors REF",r.ref_active_donors||0,C.ghost],
            ["Active Donors ALT",r.alt_active_donors||0,donorNet<0?C.red:donorNet>0?C.seq:C.ghost],
            ["Active Acceptors REF",r.ref_active_acceptors||0,C.ghost],
            ["Active Acceptors ALT",r.alt_active_acceptors||0,accNet<0?C.red:accNet>0?C.seq:C.ghost],
          ].map(([lbl,val,col])=>(
            <div key={lbl} style={{padding:"9px 12px",background:C.deep,borderRadius:4,border:`1px solid ${C.border}`}}>
              <div style={{fontFamily:F.mono,fontSize:6,color:C.muted,letterSpacing:1,marginBottom:3}}>{lbl}</div>
              <div style={{fontFamily:F.mono,fontSize:18,fontWeight:700,color:col}}>{val}</div>
            </div>
          ))}
        </div>

        {/* stat pills */}
        <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:14}}>
          {[["Disrupted Donors",r.disrupted_donors,C.red],["Disrupted Acceptors",r.disrupted_acceptors,C.red],
            ["De Novo Donors",r.created_donors,C.seq],["De Novo Acceptors",r.created_acceptors,C.seq],
            ["Cryptic Donors",r.cryptic_donors,C.purple],["Cryptic Acceptors",r.cryptic_acceptors,C.purple],
          ].map(([lbl,val,col])=>{
            const active=val>0;
            return (
              <div key={lbl} style={{flex:1,minWidth:80,textAlign:"center",padding:"9px 8px",borderRadius:4,
                background:active?`${col}14`:C.deep,border:`1px solid ${active?col+"44":C.border}`}}>
                <div style={{fontFamily:F.mono,fontSize:18,fontWeight:700,color:active?col:C.muted}}>{val}</div>
                <div style={{fontFamily:F.mono,fontSize:6,color:C.ghost,marginTop:3,lineHeight:1.5}}>{lbl}</div>
              </div>
            );
          })}
        </div>

        {/* max disruption bar */}
        {r.max_disruption>0&&(
          <div style={{marginBottom:14}}>
            <Bar val={r.max_disruption} color={sc} h={5} label="MAX PWM SCORE CHANGE"/>
          </div>
        )}

        {/* context chips */}
        <div style={{display:"flex",gap:8,marginBottom:14,flexWrap:"wrap"}}>
          <div style={{padding:"7px 12px",background:C.deep,borderRadius:4,border:`1px solid ${C.border}`}}>
            <div style={{fontFamily:F.mono,fontSize:6,color:C.muted,marginBottom:2}}>WINDOW GC%</div>
            <div style={{fontFamily:F.mono,fontSize:13,color:C.amber,fontWeight:700}}>
              {((r.window_gc||0)*100).toFixed(1)}%
            </div>
          </div>
          <div style={{padding:"7px 12px",background:C.deep,borderRadius:4,border:`1px solid ${C.border}`}}>
            <div style={{fontFamily:F.mono,fontSize:6,color:C.muted,marginBottom:2}}>POLYPYRIMIDINE TRACT</div>
            <div style={{fontFamily:F.mono,fontSize:13,color:C.teal,fontWeight:700}}>
              {((r.ppt_score||0)*100).toFixed(1)}% C/T
            </div>
          </div>
        </div>

        {/* landscape */}
        {r.sites?.length>0&&<SiteLandscape sites={r.sites}/>}

        {/* mutation mechanism */}
        <div style={{padding:"12px 14px",background:C.deep,borderRadius:5,
          borderLeft:`3px solid ${mutC(r.mutation_type)}`,marginBottom:14}}>
          <Sep label="Mutation mechanism"/>
          <p style={{fontFamily:F.mono,fontSize:10,color:C.silver,lineHeight:2,margin:0}}>{r.mutation_mechanism}</p>
        </div>

        {/* tabs */}
        <div style={{display:"flex",borderBottom:`1px solid ${C.border}`,marginBottom:14}}>
          {["sites","ese & bp","sequence"].map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{background:"none",border:"none",cursor:"pointer",
              fontFamily:F.mono,fontSize:8,letterSpacing:1.5,textTransform:"uppercase",
              color:tab===t?sc:C.ghost,borderBottom:tab===t?`2px solid ${sc}`:"2px solid transparent",
              padding:"8px 14px",marginBottom:-1,transition:"color 0.2s"}}>{t}</button>
          ))}
        </div>

        {tab==="sites"&&(
          <div>
            {visible.length>0
              ? <div style={{display:"flex",flexDirection:"column",gap:6}}>
                  {visible.map((s,i)=><SiteRow key={i} s={s} idx={i}/>)}
                </div>
              : <div style={{fontFamily:F.mono,fontSize:10,color:C.muted,textAlign:"center",
                  padding:32,border:`1px dashed ${C.border}`,borderRadius:6}}>
                  No significant splice site changes detected in this window
                </div>
            }
            {(r.sites||[]).length>6&&(
              <button onClick={()=>setShowAll(v=>!v)} style={{marginTop:10,width:"100%",background:"none",
                border:`1px solid ${C.border}`,borderRadius:4,color:C.ghost,fontFamily:F.mono,fontSize:8,padding:10,cursor:"pointer"}}>
                {showAll?`Show fewer ▲`:`Show all ${r.sites.length} sites ▼`}
              </button>
            )}
          </div>
        )}

        {tab==="ese & bp"&&(
          <div>
            <ESEPanel findings={r.ese_findings}/>
            <BranchPointPanel ref_bp={r.bp_candidates_ref} alt_bp={r.bp_candidates_alt}/>
            {(!r.ese_findings?.length&&!r.bp_candidates_ref?.length&&!r.bp_candidates_alt?.length)&&(
              <div style={{fontFamily:F.mono,fontSize:9,color:C.muted,textAlign:"center",padding:24}}>
                No ESE disruptions or branch point candidates detected in this window
              </div>
            )}
          </div>
        )}

        {tab==="sequence"&&(
          <div>
            <Sep label="200bp reference sequence window"/>
            <div style={{background:C.abyss,borderRadius:4,padding:"12px 14px",border:`1px solid ${C.border}`,marginBottom:10}}>
              <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,marginBottom:6}}>REF — variant position at center</div>
              <DNAStrip seq={r.ref_seq_display||""} highlight={{start:95,end:105}}/>
            </div>
            <div style={{fontFamily:F.mono,fontSize:8,color:C.ghost,lineHeight:1.8}}>
              <span style={{color:C.red}}>■</span> A &nbsp;
              <span style={{color:C.amber}}>■</span> T &nbsp;
              <span style={{color:C.seq}}>■</span> G &nbsp;
              <span style={{color:C.teal}}>■</span> C
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── form input ───────────────────────────────────────────── */
function Field({ label, hint, required, value, onChange, placeholder, type, maxLength, tall }) {
  const [focused,sf]=useState(false);
  return (
    <div>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
        <label style={{fontFamily:F.mono,fontSize:7,letterSpacing:2,color:C.ghost,textTransform:"uppercase"}}>{label}</label>
        {hint&&<span style={{fontFamily:F.mono,fontSize:7,color:required?C.amber:C.ghost}}>({hint})</span>}
      </div>
      {tall
        ? <textarea value={value} onChange={onChange} placeholder={placeholder}
            onFocus={()=>sf(true)} onBlur={()=>sf(false)}
            style={{width:"100%",background:C.raised,border:`1px solid ${focused?C.amber+"88":C.border}`,
              borderRadius:4,padding:"10px 13px",color:C.white,fontFamily:F.mono,fontSize:10,
              letterSpacing:0.5,lineHeight:1.6,outline:"none",boxSizing:"border-box",resize:"none",height:70,
              boxShadow:focused?`0 0 0 3px ${C.amber}18`:"none",transition:"border-color 0.15s,box-shadow 0.15s"}}/>
        : <input type={type||"text"} value={value} onChange={onChange} placeholder={placeholder} maxLength={maxLength}
            onFocus={()=>sf(true)} onBlur={()=>sf(false)}
            style={{width:"100%",background:C.raised,border:`1px solid ${focused?C.amber+"88":C.border}`,
              borderRadius:4,padding:"10px 13px",color:C.white,fontFamily:F.mono,fontSize:12,outline:"none",
              boxSizing:"border-box",boxShadow:focused?`0 0 0 3px ${C.amber}18`:"none",
              transition:"border-color 0.15s,box-shadow 0.15s"}}/>
      }
    </div>
  );
}

/* ── health dot ───────────────────────────────────────────── */
function HealthDot({ ok, label }) {
  return (
    <span style={{display:"flex",alignItems:"center",gap:5}}>
      <span style={{width:5,height:5,borderRadius:"50%",display:"inline-block",background:ok?C.seq:C.red,
        boxShadow:ok?`0 0 8px ${C.seq}`:undefined,animation:ok?"healthPulse 3s infinite":undefined}}/>
      <span style={{fontFamily:F.mono,fontSize:7,color:ok?C.ghost:C.muted}}>{label}</span>
    </span>
  );
}

/* ═════════════════════════════════════════════════════════════
   MAIN APP
═════════════════════════════════════════════════════════════ */
export default function App() {
  const [form,setForm]=useState({chrom:"",position:"",ref:"",alt:"",ref_seq:"",alt_seq:""});
  const [mode,setMode]=useState("xgb");
  const [loading,setLoading]=useState(false);
  const [result,setResult]=useState(null);
  const [error,setError]=useState(null);
  const [health,setHealth]=useState(null);
  const resultRef=useRef(null);
  const set=k=>e=>setForm(f=>({...f,[k]:e.target.value}));

  useEffect(()=>{ fetch(`${API}/health`).then(r=>r.json()).then(setHealth).catch(()=>{}); },[]);
  useEffect(()=>{ if(result&&resultRef.current) resultRef.current.scrollIntoView({behavior:"smooth",block:"start"}); },[result]);

  const run=async()=>{
    const{chrom,position,ref,alt}=form;
    if(!chrom||!position||!ref||!alt){ setError("Chromosome, position, REF and ALT are required."); return; }
    if(mode==="splice_sites"&&!form.ref_seq){ setError("REF sequence (±200bp) is required for splice site analysis."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const body={chrom,position:parseInt(position),ref:ref.toUpperCase(),alt:alt.toUpperCase(),
        ref_seq:form.ref_seq||null,alt_seq:form.alt_seq||null};
      const res=await fetch(`${API}/predict/${mode}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      if(!res.ok){ const e=await res.json(); throw new Error(e.detail||`HTTP ${res.status}`); }
      setResult({...await res.json(),_mode:mode});
    } catch(e){ setError(e.message); }
    finally{ setLoading(false); }
  };

  const loadExample=()=>{ setForm({...EX,alt_seq:""}); setResult(null); setError(null); };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400;1,600&family=DM+Sans:wght@300;400;500;600&family=Fira+Code:wght@400;500;600&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        html,body{background:${C.void};min-height:100vh;color:${C.cloud};-webkit-font-smoothing:antialiased}
        ::selection{background:${C.amber}44;color:${C.white}}
        input,textarea{color-scheme:dark}
        input::placeholder,textarea::placeholder{color:${C.muted};font-size:11px}
        ::-webkit-scrollbar{width:4px;height:4px}
        ::-webkit-scrollbar-track{background:${C.abyss}}
        ::-webkit-scrollbar-thumb{background:${C.dim};border-radius:2px}
        button{transition:opacity 0.15s,transform 0.1s}
        button:hover:not(:disabled){opacity:0.85}
        button:active:not(:disabled){transform:scale(0.98)}
        @keyframes cardIn{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
        @keyframes siteIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes healthPulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes glowPulse{0%,100%{box-shadow:0 0 8px ${C.amber}44}50%{box-shadow:0 0 20px ${C.amber}88,0 0 40px ${C.amber}22}}
        @keyframes titleReveal{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
      `}</style>

      {/* dot grid bg */}
      <div style={{position:"fixed",inset:0,zIndex:0,pointerEvents:"none",
        backgroundImage:`radial-gradient(circle,${C.dim}55 1px,transparent 1px)`,backgroundSize:"28px 28px",
        maskImage:"radial-gradient(ellipse 100% 80% at 50% 0%,black 20%,transparent 100%)",
        WebkitMaskImage:"radial-gradient(ellipse 100% 80% at 50% 0%,black 20%,transparent 100%)"}}/>
      {/* top accent */}
      <div style={{position:"fixed",top:0,left:0,right:0,height:1,zIndex:10,
        background:`linear-gradient(90deg,transparent 5%,${C.amber}88 35%,${C.seq}88 65%,transparent 95%)`}}/>

      <div style={{position:"relative",zIndex:2,minHeight:"100vh",display:"flex",justifyContent:"center",padding:"52px 20px 120px"}}>
        <div style={{width:"100%",maxWidth:660}}>

          {/* status bar */}
          <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:44,padding:"7px 14px",
            background:C.surface,border:`1px solid ${C.border}`,borderRadius:4}}>
            <div style={{width:5,height:5,borderRadius:"50%",background:C.seq,
              animation:"healthPulse 3s infinite",boxShadow:`0 0 8px ${C.seq}`}}/>
            <span style={{fontFamily:F.mono,fontSize:7,letterSpacing:3,color:C.ghost}}>SPLICE VARIANT LAB · v6.1</span>
            {health&&(
              <div style={{marginLeft:"auto",display:"flex",gap:14,alignItems:"center"}}>
                <HealthDot ok={health.xgb}     label="XGB"/>
                <HealthDot ok={health.dnabert} label="BERT"/>
                <div style={{width:1,height:10,background:C.border}}/>
                <span style={{fontFamily:F.mono,fontSize:6,color:C.muted}}>
                  τ<sub>xgb</sub>={health.xgb_threshold?.toFixed(3)}
                </span>
                <span style={{fontFamily:F.mono,fontSize:6,color:C.muted,textTransform:"uppercase"}}>{health.device}</span>
              </div>
            )}
          </div>

          {/* hero */}
          <div style={{marginBottom:52,animation:"titleReveal 0.7s ease"}}>
            <div style={{fontFamily:F.mono,fontSize:7,letterSpacing:4,color:C.amber,marginBottom:12,textTransform:"uppercase"}}>
              ◈ Self-supervised splice site disruption predictor
            </div>
            <h1 style={{fontFamily:F.display,fontSize:"clamp(44px,8vw,68px)",fontWeight:400,
              lineHeight:0.9,letterSpacing:-1,marginBottom:18,color:C.white}}>
              <span style={{fontStyle:"italic"}}>Splice</span><br/>
              <span style={{fontStyle:"italic",background:`linear-gradient(135deg,${C.amber},${C.seq} 60%)`,
                WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text"}}>Variant Lab</span>
            </h1>
            <div style={{fontFamily:F.mono,fontSize:8,color:C.ghost,lineHeight:2.2,letterSpacing:0.3}}>
              XGBoost 6-mer k-mer features (12,301 dims) · DNABERT-2-117M bidirectional transformer<br/>
              PWM donor/acceptor scoring · ESE SR-protein binding · Branch point detection · Cryptic site activation
            </div>
          </div>

          {/* mode selector */}
          <div style={{marginBottom:20}}>
            <div style={{fontFamily:F.mono,fontSize:7,color:C.ghost,letterSpacing:3,marginBottom:10,textTransform:"uppercase"}}>Analysis mode</div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:6}}>
              {MODES.map(m=>(
                <button key={m.id} onClick={()=>setMode(m.id)} style={{
                  background:mode===m.id?C.raised:C.surface,
                  border:`1px solid ${mode===m.id?m.color+"66":C.border}`,
                  borderRadius:5,padding:"13px 10px",cursor:"pointer",textAlign:"left",
                  boxShadow:mode===m.id?`0 0 24px ${m.color}14,inset 0 0 0 1px ${m.color}22`:"none",
                  transition:"all 0.2s"}}>
                  <div style={{fontFamily:F.mono,fontSize:15,color:mode===m.id?m.color:C.muted,marginBottom:5,lineHeight:1}}>{m.icon}</div>
                  <div style={{fontFamily:F.sans,fontWeight:600,fontSize:12,color:mode===m.id?C.white:C.silver,marginBottom:3}}>{m.label}</div>
                  <div style={{fontFamily:F.mono,fontSize:7,color:C.muted,letterSpacing:0.3}}>{m.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* form */}
          <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,padding:"20px 22px",marginBottom:12}}>
            <div style={{display:"flex",flexDirection:"column",gap:12}}>
              <div style={{display:"grid",gridTemplateColumns:"120px 1fr",gap:10}}>
                <Field label="Chromosome" value={form.chrom} onChange={set("chrom")} placeholder="1"/>
                <Field label="Position (GRCh38)" type="number" value={form.position} onChange={set("position")} placeholder="925952"/>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
                <Field label="REF allele" value={form.ref} onChange={set("ref")} placeholder="G" maxLength={50}/>
                <Field label="ALT allele" value={form.alt} onChange={set("alt")} placeholder="A" maxLength={50}/>
              </div>
              <Field label="REF sequence ±200bp"
                hint={mode==="splice_sites"?"required":"optional — improves accuracy"}
                required={mode==="splice_sites"}
                value={form.ref_seq} onChange={set("ref_seq")}
                placeholder="Genomic context window centered on variant position…" tall/>
              <Field label="ALT sequence"
                hint="auto-constructed from REF if blank"
                value={form.alt_seq} onChange={set("alt_seq")}
                placeholder="Leave blank — mutation will be applied to REF automatically" tall/>
            </div>
            <div style={{display:"flex",gap:8,marginTop:18}}>
              <button onClick={run} disabled={loading} style={{
                flex:1,padding:"13px 0",
                background:loading?C.raised:`linear-gradient(135deg,${C.amber}ee,${C.seq}cc)`,
                border:`1px solid ${loading?C.border:C.amber+"44"}`,borderRadius:5,
                color:loading?C.muted:C.void,fontFamily:F.sans,fontWeight:700,fontSize:14,
                cursor:loading?"not-allowed":"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:10,
                boxShadow:loading?"none":`0 0 40px ${C.amber}18`,transition:"all 0.2s",
                animation:loading?"none":"glowPulse 3s infinite"}}>
                {loading
                  ? <><div style={{width:14,height:14,border:`2px solid ${C.muted}`,borderTopColor:C.amber,
                      borderRadius:"50%",animation:"spin 0.6s linear infinite"}}/>Analysing sequence…</>
                  : "Run Analysis →"
                }
              </button>
              <button onClick={loadExample} style={{padding:"13px 18px",background:C.raised,
                border:`1px solid ${C.border}`,borderRadius:5,color:C.ghost,fontFamily:F.mono,fontSize:8,cursor:"pointer"}}>
                Example
              </button>
              {result&&(
                <button onClick={()=>setResult(null)} style={{padding:"13px 18px",background:C.raised,
                  border:`1px solid ${C.border}`,borderRadius:5,color:C.ghost,fontFamily:F.mono,fontSize:8,cursor:"pointer"}}>
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* error */}
          {error&&(
            <div style={{marginBottom:12,padding:"12px 16px",background:C.redDim,border:`1px solid ${C.red}55`,
              borderRadius:5,color:C.red,fontFamily:F.mono,fontSize:10,animation:"fadeIn 0.2s ease"}}>
              ⚠ {error}
            </div>
          )}

          {/* result */}
          <div ref={resultRef} style={{scrollMarginTop:24}}>
            {result&&(
              result._mode==="splice_sites"
                ? <SpliceCard r={result} onClose={()=>setResult(null)}/>
                : <PredCard   r={result} onClose={()=>setResult(null)}/>
            )}
          </div>

          {/* footer */}
          <div style={{marginTop:72,paddingTop:24,borderTop:`1px solid ${C.border}`,textAlign:"center",
            fontFamily:F.mono,fontSize:6,color:C.dim,letterSpacing:2,lineHeight:3}}>
            {API} · XGBoost 12,301-dim k-mer · DNABERT-2-117M · PWM donor/acceptor · ESE SR-protein · Branch point adenosine<br/>
            AUC 0.999 (XGB) · AUC 0.996 (DNABERT-2) · Trained on 99,310 balanced splice variants · GRCh38<br/>
            <span style={{color:C.amber+"80"}}>Not for clinical use · Validate with ClinVar / ACMG / MaxEntScan / SpliceAI</span>
          </div>

        </div>
      </div>
    </>
  );
}