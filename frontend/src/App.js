import { useState, useRef, useEffect } from "react";

const API = "http://localhost:8000";

const EX = {
  chrom:"1", position:"925952", ref:"G", alt:"A",
  // Real-ish genomic sequence with GT/AG motifs so splice sites actually fire
  ref_seq:"CTTGAATGACGTACAGATGCTTACGTAGTGACGTATCAGTTGCATGCATGCAGTGACTCAGTACGATGCATCGATCGATCGATCGTAGTGCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
};

// ── PALETTE ──────────────────────────────────────────────────────────
const C = {
  bg:      "#07090e",
  s1:      "#0d1018",
  s2:      "#121620",
  s3:      "#181d2a",
  border:  "#1e2535",
  border2: "#252d40",
  muted:   "#3d4a63",
  ghost:   "#5a6a8a",
  dim:     "#7a8aaa",
  soft:    "#9aabb0",
  text:    "#dde8f0",
  bright:  "#eef4f8",
  // accents
  lime:    "#7fff6a",
  limeD:   "#7fff6a18",
  limeW:   "#7fff6a44",
  red:     "#ff5c5c",
  redD:    "#ff5c5c18",
  amber:   "#ffb84d",
  amberD:  "#ffb84d18",
  blue:    "#4db8ff",
  blueD:   "#4db8ff18",
  violet:  "#b48aff",
  violetD: "#b48aff18",
  teal:    "#4dffd4",
};

const MON = "'JetBrains Mono','Fira Code',monospace";
const SER = "'Instrument Serif','Georgia',serif";
const SAN = "'DM Sans','system-ui',sans-serif";

const MODES = [
  { id:"xgb",          label:"XGBoost",      sub:"k-mer · CPU",    icon:"▦" },
  { id:"dnabert",      label:"DNABERT-2",    sub:"sequence · GPU", icon:"◈" },
  { id:"ensemble",     label:"Ensemble",     sub:"XGB+DNA fusion",  icon:"⊕" },
  { id:"splice_sites", label:"Splice Sites", sub:"PWM · cryptic",   icon:"⌁" },
];

// ── HELPERS ──────────────────────────────────────────────────────────
const sigCol = s =>
  s?.startsWith("Strong") ? C.red :
  s?.startsWith("Moderate") ? C.amber :
  s?.startsWith("Weak") || s?.startsWith("Low") ? C.blue : C.lime;

const mutCol = t =>
  t==="SNV"?C.blue : t==="Deletion"?C.red : t==="Insertion"?C.amber : C.violet;

const predCol = isPath => isPath ? C.red : C.lime;

// ── ANIMATED NUMBER ───────────────────────────────────────────────────
function Ticker({ to, dur=750, dec=0 }) {
  const [v,sv] = useState(0);
  useEffect(() => {
    const s=Date.now();
    const go=()=>{ const p=Math.min((Date.now()-s)/dur,1), e=1-Math.pow(1-p,3);
      sv(to*e); if(p<1) requestAnimationFrame(go); };
    requestAnimationFrame(go);
  }, [to]);
  return <>{v.toFixed(dec)}</>;
}

// ── SCANLINE BAR ──────────────────────────────────────────────────────
function Bar({ pct, color, h=3, glow=true }) {
  const [w,sw] = useState(0);
  useEffect(()=>{ const t=setTimeout(()=>sw(pct),80); return()=>clearTimeout(t); },[pct]);
  return (
    <div style={{ height:h, background:C.s1, borderRadius:1, overflow:"hidden", position:"relative" }}>
      <div style={{
        height:"100%", width:`${w}%`, background:color, borderRadius:1,
        transition:"width 0.85s cubic-bezier(.16,1,.3,1)",
        boxShadow: glow ? `0 0 8px ${color}99` : "none",
      }}/>
    </div>
  );
}

// ── CHIP ──────────────────────────────────────────────────────────────
function Chip({ label, color=C.ghost, sm=false }) {
  return (
    <span style={{
      display:"inline-block", fontFamily:MON,
      fontSize: sm?8:9, letterSpacing:1.2, fontWeight:600, textTransform:"uppercase",
      padding: sm?"2px 6px":"3px 9px", borderRadius:2,
      background:`${color}1a`, border:`1px solid ${color}55`, color,
    }}>{label}</span>
  );
}

// ── SECTION RULE ──────────────────────────────────────────────────────
function Rule({ label }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, margin:"14px 0 10px" }}>
      <div style={{ height:1, width:12, background:C.border2 }}/>
      <span style={{ fontFamily:MON, fontSize:8, letterSpacing:2, color:C.muted, textTransform:"uppercase" }}>{label}</span>
      <div style={{ height:1, flex:1, background:C.border2 }}/>
    </div>
  );
}

// ── GAUGE RING ────────────────────────────────────────────────────────
function Ring({ prob, color, size=100 }) {
  const r=42, c=2*Math.PI*r;
  return (
    <div style={{ position:"relative", width:size, height:size, flexShrink:0 }}>
      <svg width={size} height={size} viewBox="0 0 96 96" style={{ transform:"rotate(-90deg)", position:"absolute", inset:0 }}>
        <circle cx={48} cy={48} r={r} fill="none" stroke={C.border2} strokeWidth={6}/>
        <circle cx={48} cy={48} r={r} fill="none" stroke={color} strokeWidth={6}
          strokeDasharray={c} strokeDashoffset={c*(1-prob)} strokeLinecap="butt"
          style={{ transition:"stroke-dashoffset 1s cubic-bezier(.16,1,.3,1)" }}/>
      </svg>
      <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
        <span style={{ fontFamily:MON, fontSize:18, fontWeight:700, color, lineHeight:1 }}>
          <Ticker to={Math.round(prob*100)}/><span style={{ fontSize:11 }}>%</span>
        </span>
        <span style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:2, marginTop:2 }}>PROB</span>
      </div>
    </div>
  );
}

// ── EXACT SEQUENCE VIEWER ─────────────────────────────────────────────
function SeqView({ label, context, color }) {
  if (!context) return null;
  // context looks like: "ACGTA[GTAGTGAC]TGCA"
  const m = context.match(/^(.*)\[(.+)\](.*)$/);
  if (!m) return <span style={{ fontFamily:MON, fontSize:10, color:C.dim }}>{context}</span>;
  const [,pre,kmer,post] = m;
  // For donor: motif is at positions 3-4 inside kmer (GT)
  // For acceptor: motif at 12-13 (AG)
  return (
    <div style={{ marginTop:6, background:C.bg, borderRadius:4, padding:"8px 10px", border:`1px solid ${C.border}` }}>
      <div style={{ fontFamily:MON, fontSize:8, color:C.muted, letterSpacing:1.5, marginBottom:5 }}>{label}</div>
      <div style={{ fontFamily:MON, fontSize:11, letterSpacing:1.5, lineHeight:1.8, wordBreak:"break-all" }}>
        <span style={{ color:C.dim }}>{pre}</span>
        {kmer.split("").map((ch,i) => {
          // positions 3-4 are GT motif for donor, 12-13 for acceptor
          const isMotif = (i===3||i===4) || (i===12||i===13);
          return (
            <span key={i} style={{
              color: isMotif ? color : C.text,
              fontWeight: isMotif ? 700 : 400,
              background: isMotif ? `${color}22` : "transparent",
              borderRadius: 2,
              textDecoration: isMotif ? "underline" : "none",
              textDecorationColor: color,
            }}>{ch}</span>
          );
        })}
        <span style={{ color:C.dim }}>{post}</span>
      </div>
    </div>
  );
}

// ── SITE ROW ──────────────────────────────────────────────────────────
function SiteRow({ s, idx }) {
  const [open, setOpen] = useState(false);
  const tag  = s.disrupted?"DISRUPTED":s.created?"CREATED":s.cryptic?"CRYPTIC":"SHIFTED";
  const col  = s.disrupted?C.red:s.created?C.lime:s.cryptic?C.amber:C.muted;
  const pos  = `${s.position>=0?"+":""}${s.position}`;

  return (
    <div style={{
      background:C.s1, border:`1px solid ${open?col+"55":C.border}`,
      borderRadius:5, overflow:"hidden", transition:"border-color 0.2s",
      animation:`slideIn 0.2s ${idx*30}ms both`,
    }}>
      <div onClick={()=>setOpen(v=>!v)} style={{
        padding:"10px 14px", cursor:"pointer",
        display:"grid", gridTemplateColumns:"90px 1fr 90px 24px",
        gap:10, alignItems:"center",
      }}>
        <div>
          <Chip label={tag} color={col} sm/>
          <div style={{ fontFamily:MON, fontSize:8, color:C.muted, marginTop:4 }}>
            {s.type} · {pos}
          </div>
        </div>

        <div>
          {[["REF",s.ref_score,C.ghost],["ALT",s.alt_score,col]].map(([l,v,c])=>(
            <div key={l} style={{ display:"flex", gap:6, alignItems:"center", marginBottom:l==="REF"?3:0 }}>
              <span style={{ fontFamily:MON, fontSize:7, color:C.muted, width:18 }}>{l}</span>
              <div style={{ flex:1 }}><Bar pct={v*100} color={c} h={2} glow={l==="ALT"}/></div>
              <span style={{ fontFamily:MON, fontSize:9, color:c, width:28, textAlign:"right" }}>{v.toFixed(2)}</span>
            </div>
          ))}
        </div>

        <div style={{ textAlign:"right" }}>
          <span style={{
            fontFamily:MON, fontSize:10, fontWeight:700,
            color: s.delta>=0?C.lime:C.red,
            background: s.delta>=0?C.limeD:C.redD,
            border:`1px solid ${s.delta>=0?C.limeW:C.red+"44"}`,
            borderRadius:2, padding:"2px 6px",
          }}>{s.delta>=0?"+":""}{s.delta.toFixed(3)}</span>
        </div>

        <span style={{ fontFamily:MON, fontSize:9, color:C.muted, textAlign:"center" }}>
          {open?"▲":"▼"}
        </span>
      </div>

      {open && (
        <div style={{ borderTop:`1px solid ${C.border}`, padding:"10px 14px", animation:"fadeIn 0.15s ease" }}>
          {/* exact sequences */}
          <SeqView label="REF sequence context" context={s.ref_context} color={C.ghost}/>
          <SeqView label="ALT sequence context" context={s.alt_context} color={col}/>

          {/* kmer diff */}
          {(s.ref_kmer||s.alt_kmer) && (
            <div style={{ marginTop:8, background:C.bg, borderRadius:4, padding:"8px 10px", border:`1px solid ${C.border}` }}>
              <div style={{ fontFamily:MON, fontSize:8, color:C.muted, letterSpacing:1.5, marginBottom:6 }}>MATCHED K-MER</div>
              {[["REF",s.ref_kmer,false],["ALT",s.alt_kmer,true]].map(([l,seq,isAlt])=>(
                <div key={l} style={{ display:"flex", gap:8, alignItems:"center", marginBottom:l==="REF"?4:0 }}>
                  <span style={{ fontFamily:MON, fontSize:7, color:C.muted, width:22 }}>{l}</span>
                  <span style={{ fontFamily:MON, fontSize:11, letterSpacing:1.5 }}>
                    {(seq||"—").split("").map((ch,i)=>{
                      const changed = s.ref_kmer && s.alt_kmer && s.ref_kmer[i]!==s.alt_kmer[i];
                      return <span key={i} style={{
                        color: changed&&isAlt?C.amber:C.soft,
                        background: changed&&isAlt?C.amberD:"transparent",
                        borderRadius:1,
                      }}>{ch}</span>;
                    })}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* reasoning */}
          <div style={{ marginTop:8, padding:"10px 12px", background:C.bg, borderRadius:4, borderLeft:`2px solid ${col}` }}>
            <div style={{ fontFamily:MON, fontSize:8, color:C.muted, letterSpacing:1.5, marginBottom:5 }}>REASONING</div>
            <p style={{ fontFamily:MON, fontSize:11, color:C.soft, lineHeight:1.85, margin:0 }}>{s.reasoning}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── LANDSCAPE HEATMAP ─────────────────────────────────────────────────
function Landscape({ sites }) {
  if (!sites?.length) return null;

  const track = (items, label) => {
    if (!items.length) return null;
    const positions = items.map(s=>s.position);
    const minP = Math.min(...positions), maxP = Math.max(...positions);
    const span = Math.max(maxP-minP, 1);
    return (
      <div style={{ marginBottom:12 }}>
        <div style={{ fontFamily:MON, fontSize:8, color:C.muted, letterSpacing:1.5, marginBottom:4 }}>{label.toUpperCase()}</div>
        <div style={{ position:"relative", height:28, background:C.bg, borderRadius:3, border:`1px solid ${C.border}` }}>
          <div style={{ position:"absolute", left:"50%", top:0, width:1, height:"100%", background:C.border2 }}/>
          {items.map((s,i)=>{
            const xp = span>0 ? ((s.position-minP)/span)*88+6 : 50;
            const col = s.disrupted?C.red:s.created?C.lime:s.cryptic?C.amber:C.muted;
            const ht = Math.max(s.alt_score*26,4);
            return <div key={i} title={`${s.type} ${s.position>=0?"+":""}${s.position} Δ${s.delta.toFixed(3)}`} style={{
              position:"absolute", bottom:0, left:`${xp}%`, width:3, height:ht,
              background:col, borderRadius:"2px 2px 0 0", transform:"translateX(-50%)",
              boxShadow:`0 0 4px ${col}99`,
            }}/>;
          })}
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", marginTop:2 }}>
          <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>{minP>=0?"+":""}{minP}bp</span>
          <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>variant ↑</span>
          <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>{maxP>=0?"+":""}{maxP}bp</span>
        </div>
      </div>
    );
  };

  return (
    <div style={{ background:C.bg, border:`1px solid ${C.border}`, borderRadius:5, padding:"12px 14px", marginBottom:14 }}>
      <Rule label="Splice Site Landscape"/>
      {track(sites.filter(s=>s.type==="donor"), "Donor sites")}
      {track(sites.filter(s=>s.type==="acceptor"), "Acceptor sites")}
      <div style={{ display:"flex", gap:12, marginTop:4 }}>
        {[["DISRUPTED",C.red],["CREATED",C.lime],["CRYPTIC",C.amber],["SHIFTED",C.muted]].map(([l,c])=>(
          <span key={l} style={{ display:"flex", alignItems:"center", gap:4 }}>
            <span style={{ width:6, height:6, background:c, borderRadius:1, display:"inline-block" }}/>
            <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>{l}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

// ── BIOLOGY RULES ─────────────────────────────────────────────────────
function BioRules({ ref, alt, mutation_type }) {
  const r=ref?.toUpperCase()||"", a=alt?.toUpperCase()||"", mt=mutation_type||"";
  const rules = [
    { label:"+1G invariant donor",  col:r==="G"&&a!=="G"?C.red:C.muted, fired:r==="G"&&a!=="G",
      desc: r==="G"&&a!=="G"?"REF=G → ALT disrupts invariant +1G of GT donor. Position >99% conserved — virtually always pathogenic.":"Not affected." },
    { label:"+2T invariant donor",  col:r==="T"&&a!=="T"?C.red:C.muted, fired:r==="T"&&a!=="T",
      desc: r==="T"&&a!=="T"?"REF=T → ALT disrupts +2T of GT donor. Most substitutions abolish splicing.":"Not affected." },
    { label:"−2A invariant acceptor",col:r==="A"&&a!=="A"?C.red:C.muted, fired:r==="A"&&a!=="A",
      desc: r==="A"&&a!=="A"?"REF=A → ALT disrupts −2A of AG acceptor. Loss causes exon skipping or intron retention.":"Not affected." },
    { label:"GT donor loss",        col:r.includes("GT")&&!a.includes("GT")?C.amber:C.muted,
      fired:r.includes("GT")&&!a.includes("GT"),
      desc: r.includes("GT")&&!a.includes("GT")?"GT dinucleotide present in REF but lost in ALT — donor site disruption.":"GT not affected." },
    { label:"AG acceptor loss",     col:r.includes("AG")&&!a.includes("AG")?C.amber:C.muted,
      fired:r.includes("AG")&&!a.includes("AG"),
      desc: r.includes("AG")&&!a.includes("AG")?"AG dinucleotide present in REF but lost in ALT — acceptor disruption.":"AG not affected." },
    { label:"Frameshift risk",      col:(mt==="Insertion"||mt==="Deletion")&&(Math.abs(r.length-a.length)%3!==0)?C.red:C.muted,
      fired:(mt==="Insertion"||mt==="Deletion")&&(Math.abs(r.length-a.length)%3!==0),
      desc: (mt==="Insertion"||mt==="Deletion")&&(Math.abs(r.length-a.length)%3!==0)?`${Math.abs(r.length-a.length)}bp frameshift — disrupts reading frame and downstream splice geometry.`:"No frameshift." },
  ];
  return (
    <div>
      <Rule label="Biological Rule Evaluation"/>
      <p style={{ fontFamily:MON, fontSize:9, color:C.muted, marginBottom:10, lineHeight:1.7 }}>
        Heuristic rules based on splice site conservation. Independent of ML model scores.
      </p>
      {rules.map((f,i)=>(
        <div key={i} style={{
          marginBottom:5, padding:"9px 11px",
          background: f.fired?`${f.col}12`:C.bg,
          border:`1px solid ${f.fired?f.col+"44":C.border}`,
          borderRadius:4,
        }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:4 }}>
            <span style={{ fontFamily:MON, fontSize:10, color:f.fired?f.col:C.muted }}>
              {f.fired?"● ":"○ "}{f.label}
            </span>
            {f.fired && <Chip label="TRIGGERED" color={f.col} sm/>}
          </div>
          <p style={{ fontFamily:MON, fontSize:10, color:C.muted, margin:0, lineHeight:1.7 }}>{f.desc}</p>
        </div>
      ))}
    </div>
  );
}

// ── CONFIDENCE PANEL ──────────────────────────────────────────────────
function ConfPanel({ prob, thresh }) {
  const margin = Math.abs(prob-thresh);
  const isPath = prob>=thresh;
  const col = isPath?C.red:C.lime;
  const certCol = margin>0.35?C.lime:margin>0.15?C.amber:C.red;
  const certLabel = margin>0.35?"High":margin>0.15?"Medium":"Low";
  return (
    <div>
      <Rule label="Prediction Confidence"/>
      <div style={{ display:"flex", gap:10, marginBottom:12 }}>
        <div style={{ flex:1, background:C.bg, border:`1px solid ${certCol}44`, borderRadius:5, padding:"14px 16px" }}>
          <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:2, marginBottom:5 }}>CONFIDENCE</div>
          <div style={{ fontFamily:SER, fontSize:26, color:certCol, lineHeight:1 }}>{certLabel}</div>
        </div>
        <div style={{ flex:1, background:C.bg, border:`1px solid ${col}44`, borderRadius:5, padding:"14px 16px", textAlign:"right" }}>
          <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:2, marginBottom:5 }}>PROBABILITY</div>
          <div style={{ fontFamily:MON, fontSize:22, fontWeight:700, color:col }}>{(prob*100).toFixed(1)}%</div>
        </div>
      </div>
      <div style={{ marginBottom:10 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
          <span style={{ fontFamily:MON, fontSize:8, color:C.muted }}>Distance from decision boundary</span>
          <span style={{ fontFamily:MON, fontSize:9, color:C.blue }}>{(margin*100).toFixed(1)}pp</span>
        </div>
        <Bar pct={margin*200} color={C.blue} h={4}/>
        <p style={{ fontFamily:MON, fontSize:9, color:C.muted, marginTop:6, lineHeight:1.7 }}>
          {margin<0.1?"Near-boundary — model uncertain. Run splice site analysis for confirmation."
          :margin<0.25?"Moderate margin. Consider ensemble and splice site analysis."
          :"Strong separation from threshold — high model confidence."}
        </p>
      </div>
      <div style={{ padding:"9px 11px", background:C.bg, border:`1px solid ${C.border}`, borderRadius:4, fontFamily:MON, fontSize:9, color:C.muted, lineHeight:1.8 }}>
        ℹ Confidence is margin-based, not calibrated. Validate with ClinVar/ACMG and MaxEntScan for clinical decisions.
      </div>
    </div>
  );
}

// ── VAR GRID ──────────────────────────────────────────────────────────
function VarGrid({ r }) {
  const rows=[["CHR",r.chrom,C.blue],["POS",r.position?.toLocaleString(),C.soft],
              ["REF",r.ref,C.lime],["ALT",r.alt,C.amber],
              ["TYPE",r.mutation_type,mutCol(r.mutation_type)],["TIME",`${r.latency_ms}ms`,C.muted]];
  return (
    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:5 }}>
      {rows.map(([k,v,c])=>(
        <div key={k} style={{ background:C.bg, border:`1px solid ${C.border}`, borderRadius:4, padding:"8px 10px" }}>
          <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:1.5, marginBottom:2 }}>{k}</div>
          <div style={{ fontFamily:MON, fontSize:12, fontWeight:700, color:c }}>{v}</div>
        </div>
      ))}
    </div>
  );
}

// ── PREDICTION CARD ───────────────────────────────────────────────────
function PredCard({ r, onClose }) {
  const isPath = r.prediction==="Pathogenic";
  const col = predCol(isPath);
  const [tab, setTab] = useState("overview");

  return (
    <div style={{
      background:C.s1, border:`1px solid ${col}30`, borderRadius:8,
      overflow:"hidden", animation:"slideUp 0.4s cubic-bezier(.16,1,.3,1)",
    }}>
      {/* header */}
      <div style={{
        background:`linear-gradient(90deg, ${col}15, transparent 60%)`,
        borderBottom:`1px solid ${col}20`, padding:"16px 20px",
        display:"flex", alignItems:"center", gap:14,
      }}>
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:2.5, marginBottom:5 }}>
            {r.model?.toUpperCase()} · {r.latency_ms}ms
          </div>
          <div style={{ display:"flex", alignItems:"baseline", gap:10, flexWrap:"wrap" }}>
            <span style={{ fontFamily:SER, fontSize:30, color:col, lineHeight:1 }}>{r.prediction}</span>
            <Chip label={r.confidence} color={col}/>
            <Chip label={r.mutation_type} color={mutCol(r.mutation_type)}/>
          </div>
        </div>
        <Ring prob={r.probability} color={col}/>
        <button onClick={onClose} style={{
          background:"none", border:`1px solid ${C.border2}`, borderRadius:4,
          color:C.muted, cursor:"pointer", width:26, height:26,
          fontFamily:MON, fontSize:11, alignSelf:"flex-start",
        }}>✕</button>
      </div>

      <div style={{ padding:"16px 20px" }}>
        {/* prob track */}
        <div style={{ marginBottom:16 }}>
          <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
            <span style={{ fontFamily:MON, fontSize:7, color:C.lime, letterSpacing:1 }}>BENIGN</span>
            <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>τ = {(r.threshold_used*100).toFixed(0)}%</span>
            <span style={{ fontFamily:MON, fontSize:7, color:C.red, letterSpacing:1 }}>PATHOGENIC</span>
          </div>
          <div style={{ position:"relative", height:5, background:C.s2, borderRadius:1 }}>
            <div style={{
              position:"absolute", left:0, top:0, height:"100%",
              width:`${r.probability*100}%`, background:col, borderRadius:1,
              transition:"width 0.9s cubic-bezier(.16,1,.3,1)",
              boxShadow:`0 0 8px ${col}88`,
            }}/>
            <div style={{
              position:"absolute", left:`${r.threshold_used*100}%`, top:-3,
              transform:"translateX(-50%)", width:1, height:11, background:C.amber,
            }}/>
          </div>
        </div>

        {/* mechanism */}
        <div style={{ padding:"10px 12px", background:C.bg, borderRadius:4, borderLeft:`2px solid ${col}`, marginBottom:14 }}>
          <Rule label="Molecular mechanism"/>
          <p style={{ fontFamily:MON, fontSize:11, color:C.soft, lineHeight:1.9, margin:0 }}>{r.mechanism}</p>
        </div>

        {/* tabs */}
        <div style={{ display:"flex", borderBottom:`1px solid ${C.border}`, marginBottom:14 }}>
          {["overview","rules","confidence"].map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:MON, fontSize:8, letterSpacing:1.5, textTransform:"uppercase",
              color: tab===t?col:C.muted,
              borderBottom: `1px solid ${tab===t?col:"transparent"}`,
              padding:"7px 14px", marginBottom:-1, transition:"color 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="overview"    && <VarGrid r={r}/>}
        {tab==="rules"       && <BioRules ref={r.ref} alt={r.alt} mutation_type={r.mutation_type}/>}
        {tab==="confidence"  && <ConfPanel prob={r.probability} thresh={r.threshold_used}/>}
      </div>
    </div>
  );
}

// ── SPLICE CARD ───────────────────────────────────────────────────────
function SpliceCard({ r, onClose }) {
  const sc = sigCol(r.pathogenicity_signal);
  const [showAll, setShowAll] = useState(false);
  const [tab, setTab] = useState("sites");
  const vis = showAll ? r.sites : r.sites?.slice(0,7)||[];

  return (
    <div style={{
      background:C.s1, border:`1px solid ${sc}30`, borderRadius:8,
      overflow:"hidden", animation:"slideUp 0.4s cubic-bezier(.16,1,.3,1)",
    }}>
      {/* header */}
      <div style={{
        background:`linear-gradient(90deg, ${sc}12, transparent 60%)`,
        borderBottom:`1px solid ${sc}20`, padding:"16px 20px",
        display:"flex", alignItems:"flex-start", gap:10,
      }}>
        <div style={{ flex:1 }}>
          <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:2.5, marginBottom:5 }}>
            SPLICE ANALYSIS · {r.latency_ms}ms · {r.sites_found} sites scored
          </div>
          <div style={{ fontFamily:SER, fontSize:20, color:sc, lineHeight:1.2, marginBottom:5 }}>
            {r.pathogenicity_signal}
          </div>
          <div style={{ fontFamily:MON, fontSize:10, color:C.muted }}>{r.summary}</div>
        </div>
        <button onClick={onClose} style={{
          background:"none", border:`1px solid ${C.border2}`, borderRadius:4,
          color:C.muted, cursor:"pointer", width:26, height:26, fontFamily:MON, fontSize:11,
        }}>✕</button>
      </div>

      <div style={{ padding:"16px 20px" }}>
        {/* mutation */}
        <div style={{ padding:"10px 12px", background:C.bg, borderRadius:4, borderLeft:`2px solid ${mutCol(r.mutation_type)}`, marginBottom:14 }}>
          <div style={{ display:"flex", gap:8, alignItems:"center", marginBottom:6 }}>
            <Chip label={r.mutation_type} color={mutCol(r.mutation_type)} sm/>
            <span style={{ fontFamily:MON, fontSize:10, color:C.muted }}>{r.mutation_detail}</span>
          </div>
          <p style={{ fontFamily:MON, fontSize:11, color:C.soft, lineHeight:1.85, margin:0 }}>{r.mutation_mechanism}</p>
        </div>

        {/* pills */}
        <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:14 }}>
          {[
            ["Disrupted Donors",   r.disrupted_donors,   C.red],
            ["Disrupted Acceptors",r.disrupted_acceptors, C.red],
            ["New Donors",         r.created_donors,      C.lime],
            ["New Acceptors",      r.created_acceptors,   C.lime],
            ["Cryptic Donors",     r.cryptic_donors,      C.amber],
            ["Cryptic Acceptors",  r.cryptic_acceptors,   C.amber],
          ].map(([l,v,c])=>(
            <div key={l} style={{
              flex:1, minWidth:80, textAlign:"center",
              padding:"9px 8px", borderRadius:4,
              background: v>0?`${c}12`:C.bg,
              border:`1px solid ${v>0?c+"44":C.border}`,
            }}>
              <div style={{ fontFamily:MON, fontSize:18, fontWeight:700, color:v>0?c:C.muted }}>{v}</div>
              <div style={{ fontFamily:MON, fontSize:7, color:C.muted, marginTop:2, lineHeight:1.5 }}>{l}</div>
            </div>
          ))}
        </div>

        {/* max disruption */}
        {r.max_disruption>0 && (
          <div style={{ marginBottom:14 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
              <span style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:1 }}>MAX SCORE CHANGE</span>
              <span style={{ fontFamily:MON, fontSize:9, fontWeight:700, color:sc }}>{(r.max_disruption*100).toFixed(1)}%</span>
            </div>
            <Bar pct={r.max_disruption*100} color={sc} h={5}/>
          </div>
        )}

        {/* landscape */}
        {r.sites?.length>0 && <Landscape sites={r.sites}/>}

        {/* tabs */}
        <div style={{ display:"flex", borderBottom:`1px solid ${C.border}`, marginBottom:12 }}>
          {["sites","variant"].map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none", border:"none", cursor:"pointer",
              fontFamily:MON, fontSize:8, letterSpacing:1.5, textTransform:"uppercase",
              color: tab===t?sc:C.muted,
              borderBottom:`1px solid ${tab===t?sc:"transparent"}`,
              padding:"7px 14px", marginBottom:-1, transition:"color 0.2s",
            }}>{t}</button>
          ))}
        </div>

        {tab==="sites" && (
          <div>
            {vis.length>0 ? (
              <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
                {vis.map((s,i)=><SiteRow key={i} s={s} idx={i}/>)}
              </div>
            ) : (
              <div style={{ fontFamily:MON, fontSize:11, color:C.muted, textAlign:"center", padding:28, border:`1px dashed ${C.border}`, borderRadius:5 }}>
                No significant splice site changes detected
              </div>
            )}
            {r.sites?.length>7 && (
              <button onClick={()=>setShowAll(v=>!v)} style={{
                marginTop:8, width:"100%", background:"none",
                border:`1px solid ${C.border}`, borderRadius:4,
                color:C.muted, fontFamily:MON, fontSize:9, padding:9, cursor:"pointer",
              }}>{showAll?`Show less ▲`:`Show all ${r.sites.length} sites ▼`}</button>
            )}
          </div>
        )}

        {tab==="variant" && <div style={{ marginTop:4 }}><VarGrid r={{...r, latency_ms:r.latency_ms}}/></div>}
      </div>
    </div>
  );
}

// ── INPUT ─────────────────────────────────────────────────────────────
function Field({ label, hint, req, ...props }) {
  const [f,sf] = useState(false);
  return (
    <div>
      <label style={{ fontFamily:MON, fontSize:7, letterSpacing:2, textTransform:"uppercase", color:C.muted, marginBottom:5, display:"flex", gap:6 }}>
        {label}
        {hint && <span style={{ color:req?C.amber:C.muted, textTransform:"none", letterSpacing:0 }}>({hint})</span>}
      </label>
      <input {...props} onFocus={()=>sf(true)} onBlur={()=>sf(false)} style={{
        width:"100%", background:C.s2,
        border:`1px solid ${f?C.limeW:C.border}`,
        borderRadius:4, padding:"10px 12px", color:C.text,
        fontFamily:MON, fontSize:11, outline:"none", boxSizing:"border-box",
        transition:"border-color 0.15s",
        boxShadow: f?`0 0 0 3px ${C.limeD}`:"none",
        ...props.style,
      }}/>
    </div>
  );
}

// ── HEALTH DOT ────────────────────────────────────────────────────────
function HDot({ ok, label }) {
  return (
    <span style={{ display:"flex", alignItems:"center", gap:4 }}>
      <span style={{ width:5, height:5, borderRadius:"50%", background:ok?C.lime:C.red, boxShadow:ok?`0 0 6px ${C.lime}`:"none", display:"inline-block" }}/>
      <span style={{ fontFamily:MON, fontSize:7, color:ok?C.dim:C.muted }}>{label}</span>
    </span>
  );
}

// ── MAIN ──────────────────────────────────────────────────────────────
export default function App() {
  const [form,sf]    = useState({ chrom:"",position:"",ref:"",alt:"",ref_seq:"",alt_seq:"" });
  const [mode,sm]    = useState("xgb");
  const [loading,sl] = useState(false);
  const [result,sr]  = useState(null);
  const [error,se]   = useState(null);
  const [health,sh]  = useState(null);
  const ref = useRef(null);

  const set = k => e => sf(f=>({...f,[k]:e.target.value}));

  useEffect(()=>{ fetch(`${API}/health`).then(r=>r.json()).then(sh).catch(()=>{}); },[]);
  useEffect(()=>{ if(result&&ref.current) ref.current.scrollIntoView({behavior:"smooth",block:"nearest"}); },[result]);

  const run = async () => {
    const {chrom,position,ref:r,alt} = form;
    if (!chrom||!position||!r||!alt) { se("chrom, position, ref, alt are required."); return; }
    if (mode==="splice_sites"&&!form.ref_seq) { se("ref_seq required for splice site analysis."); return; }
    sl(true); se(null); sr(null);
    try {
      const body={chrom,position:parseInt(position),ref:r.toUpperCase(),alt:alt.toUpperCase(),
                  ref_seq:form.ref_seq||null,alt_seq:form.alt_seq||null};
      const res=await fetch(`${API}/predict/${mode}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      if(!res.ok){const e=await res.json();throw new Error(e.detail||`HTTP ${res.status}`);}
      sr({...await res.json(),_mode:mode});
    } catch(e){se(e.message);}
    finally{sl(false);}
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        html,body{background:${C.bg};min-height:100vh;color:${C.text}}
        input::placeholder{color:${C.muted};font-size:10px;font-family:${MON}}
        button{transition:opacity 0.15s} button:hover{opacity:0.8}
        @keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes slideIn{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:translateX(0)}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        @keyframes spin{to{transform:rotate(360deg)}}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border2};border-radius:2px}
      `}</style>

      {/* grid bg */}
      <div style={{
        position:"fixed", inset:0, zIndex:0, pointerEvents:"none",
        backgroundImage:`linear-gradient(${C.border}55 1px,transparent 1px),linear-gradient(90deg,${C.border}55 1px,transparent 1px)`,
        backgroundSize:"48px 48px",
        maskImage:"radial-gradient(ellipse 70% 50% at 50% 0%,black 20%,transparent 100%)",
        WebkitMaskImage:"radial-gradient(ellipse 70% 50% at 50% 0%,black 20%,transparent 100%)",
      }}/>

      {/* top accent line */}
      <div style={{
        position:"fixed", top:0, left:"25%", right:"25%", height:1, zIndex:1,
        background:`linear-gradient(90deg,transparent,${C.lime}88,transparent)`,
      }}/>

      <div style={{ position:"relative", zIndex:2, minHeight:"100vh", display:"flex", justifyContent:"center", padding:"52px 20px 100px" }}>
        <div style={{ width:"100%", maxWidth:600 }}>

          {/* STATUS BAR */}
          <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:36, padding:"6px 12px", background:C.s1, border:`1px solid ${C.border}`, borderRadius:4 }}>
            <div style={{ width:5, height:5, borderRadius:"50%", background:C.lime, boxShadow:`0 0 8px ${C.lime}`, animation:"pulse 2.5s infinite" }}/>
            <span style={{ fontFamily:MON, fontSize:7, letterSpacing:3, color:C.muted }}>SPLICE VARIANT LAB · v5.0</span>
            {health && (
              <div style={{ marginLeft:"auto", display:"flex", gap:12, alignItems:"center" }}>
                <HDot ok={health.xgb} label="XGB"/>
                <HDot ok={health.dnabert} label="DNA"/>
                <span style={{ fontFamily:MON, fontSize:7, color:C.muted }}>τ={health.xgb_threshold?.toFixed(3)}</span>
                <span style={{ fontFamily:MON, fontSize:7, color:C.muted, textTransform:"uppercase" }}>{health.device}</span>
              </div>
            )}
          </div>

          {/* HEADING */}
          <div style={{ marginBottom:44 }}>
            <h1 style={{ fontFamily:SER, fontSize:58, fontWeight:400, color:C.bright, letterSpacing:-1.5, lineHeight:0.95, marginBottom:12 }}>
              <span style={{ fontStyle:"italic" }}>Splice</span><br/>
              <span style={{
                background:`linear-gradient(120deg,${C.lime},${C.teal})`,
                WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", backgroundClip:"text",
              }}>Variant Lab</span>
            </h1>
            <p style={{ fontFamily:MON, fontSize:8, color:C.muted, lineHeight:2.2, letterSpacing:0.5 }}>
              XGBoost k-mer · DNABERT-2-117M · PWM donor/acceptor · Cryptic site detection
            </p>
          </div>

          {/* MODE */}
          <div style={{ marginBottom:22 }}>
            <div style={{ fontFamily:MON, fontSize:7, color:C.muted, letterSpacing:3, marginBottom:9 }}>ANALYSIS MODE</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:6 }}>
              {MODES.map(m=>(
                <button key={m.id} onClick={()=>sm(m.id)} style={{
                  background: mode===m.id?C.s2:C.s1,
                  border:`1px solid ${mode===m.id?C.limeW:C.border}`,
                  borderRadius:5, padding:"11px 7px", cursor:"pointer", textAlign:"left",
                  boxShadow: mode===m.id?`0 0 18px ${C.lime}0c`:"none", transition:"all 0.2s",
                }}>
                  <div style={{ fontFamily:MON, fontSize:13, color:mode===m.id?C.lime:C.muted, marginBottom:4 }}>{m.icon}</div>
                  <div style={{ fontFamily:SAN, fontWeight:600, fontSize:11, color:mode===m.id?C.bright:C.dim, marginBottom:2 }}>{m.label}</div>
                  <div style={{ fontFamily:MON, fontSize:7, color:C.muted }}>{m.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* FORM */}
          <div style={{ background:C.s1, border:`1px solid ${C.border}`, borderRadius:6, padding:"18px 20px", marginBottom:12 }}>
            <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 2fr", gap:10 }}>
                <Field label="Chromosome" value={form.chrom} onChange={set("chrom")} placeholder="1"/>
                <Field label="Position" type="number" value={form.position} onChange={set("position")} placeholder="925952"/>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                <Field label="REF Allele" value={form.ref} onChange={set("ref")} placeholder="G" maxLength={50}/>
                <Field label="ALT Allele" value={form.alt} onChange={set("alt")} placeholder="A" maxLength={50}/>
              </div>
              <Field label="REF Sequence ±200bp"
                hint={mode==="splice_sites"?"required":"optional"}
                req={mode==="splice_sites"}
                value={form.ref_seq} onChange={set("ref_seq")}
                placeholder="Genomic window with GT/AG motifs…"
                style={{ fontSize:10, letterSpacing:0.5 }}/>
              <Field label="ALT Sequence" hint="auto-built if blank"
                value={form.alt_seq} onChange={set("alt_seq")}
                placeholder="Leave blank to auto-compute…"
                style={{ fontSize:10 }}/>
            </div>

            <div style={{ display:"flex", gap:8, marginTop:16 }}>
              <button onClick={run} disabled={loading} style={{
                flex:1, padding:"12px 0",
                background: loading?C.s2:`linear-gradient(135deg,${C.lime}dd,${C.teal}bb)`,
                border:`1px solid ${loading?C.border:C.limeW}`,
                borderRadius:5, color:loading?C.muted:C.bg,
                fontFamily:SAN, fontWeight:700, fontSize:13,
                cursor:loading?"not-allowed":"pointer",
                display:"flex", alignItems:"center", justifyContent:"center", gap:8,
                boxShadow:loading?"none":`0 0 24px ${C.lime}18`, transition:"all 0.2s",
              }}>
                {loading ? (
                  <><div style={{ width:12,height:12,border:`2px solid ${C.muted}`,borderTopColor:C.lime,borderRadius:"50%",animation:"spin 0.5s linear infinite" }}/> Analysing…</>
                ) : "Run Analysis →"}
              </button>
              <button onClick={()=>{sf({...EX,alt_seq:""});sr(null);se(null);}} style={{ padding:"12px 16px",background:C.s2,border:`1px solid ${C.border}`,borderRadius:5,color:C.ghost,fontFamily:MON,fontSize:8,cursor:"pointer" }}>
                Example
              </button>
              {result && <button onClick={()=>sr(null)} style={{ padding:"12px 16px",background:C.s2,border:`1px solid ${C.border}`,borderRadius:5,color:C.ghost,fontFamily:MON,fontSize:8,cursor:"pointer" }}>Clear</button>}
            </div>
          </div>

          {error && (
            <div style={{ marginBottom:12,padding:"11px 14px",background:C.redD,border:`1px solid ${C.red}44`,borderRadius:5,color:C.red,fontFamily:MON,fontSize:11,animation:"fadeIn 0.2s ease" }}>
              ⚠ {error}
            </div>
          )}

          {result && (
            <div ref={ref}>
              {result._mode==="splice_sites"
                ? <SpliceCard r={result} onClose={()=>sr(null)}/>
                : <PredCard   r={result} onClose={()=>sr(null)}/>}
            </div>
          )}

          <div style={{ marginTop:56,textAlign:"center",fontFamily:MON,fontSize:7,color:C.muted,letterSpacing:2,lineHeight:2.5 }}>
            {API} · XGBoost k-mer · DNABERT-2-117M · PWM splice<br/>
            Not for clinical use · Validate with ClinVar / ACMG / MaxEntScan
          </div>
        </div>
      </div>
    </>
  );
}