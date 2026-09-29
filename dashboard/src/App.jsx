import { useMemo, useState } from "react";
import { models, classes, bestConfusionMatrix } from "./data/models";

function Metric({label, value, suffix=""}) {
  return <div className="metric">
    <div className="metric-label">{label}</div>
    <div className="metric-value">{value}{suffix}</div>
  </div>
}

function AccuracyBars() {
  const max = Math.max(...models.map(m => m.accuracy));
  return <div className="bars">
    {models.map(m => (
      <div className="bar-row" key={m.id}>
        <div className="bar-name">{m.short}</div>
        <div className="bar-track">
          <div className="bar-fill" style={{width:`${(m.accuracy / 65) * 100}%`}} />
        </div>
        <div className="bar-value">{m.accuracy.toFixed(2)}%</div>
      </div>
    ))}
  </div>
}

function ConfusionMatrix({matrix}) {
  const max = Math.max(...matrix.flat());
  return <div className="cm-wrap">
    <table className="cm">
      <thead>
        <tr><th>Actual \ Pred.</th>{classes.map(c => <th key={c}>{c.slice(0,3)}</th>)}</tr>
      </thead>
      <tbody>
        {matrix.map((row,i) => (
          <tr key={classes[i]}>
            <th>{classes[i]}</th>
            {row.map((v,j) => {
              const alpha = 0.08 + (v/max)*0.82;
              return <td key={j} style={{background:`rgba(56,189,248,${alpha})`}}>{v}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
}

function App() {
  const [selected, setSelected] = useState(models[4].id);
  const [file, setFile] = useState(null);
  const [apiMessage, setApiMessage] = useState("");
  const model = models.find(m => m.id === selected);

  const maxAcc = Math.max(...models.map(m => m.accuracy));
  const minAcc = Math.min(...models.map(m => m.accuracy));

  const liveReady = selected === "cnn-bilstm-mha-specaug";

  async function runPrediction() {
    if (!file) {
      setApiMessage("Please choose a WAV file first.");
      return;
    }
    setApiMessage("Connecting to inference API...");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/predict", {method:"POST", body:form});
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      setApiMessage(`Prediction: ${data.emotion} • confidence ${(data.confidence*100).toFixed(1)}%`);
    } catch (err) {
      setApiMessage(`Live inference not available: ${err.message}`);
    }
  }

  return <div className="app">
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">SER</div>
        <div>
          <div className="brand-title">EmotionLab</div>
          <div className="brand-sub">SE4050 Deep Learning</div>
        </div>
      </div>
      <nav>
        <a href="#overview">Overview</a>
        <a href="#models">Six Models</a>
        <a href="#comparison">Comparison</a>
        <a href="#analysis">Model Analysis</a>
        <a href="#prediction">Prediction</a>
        <a href="#methodology">Methodology</a>
      </nav>
      <div className="side-note">
        <b>Dataset</b><br/>CREMA-D<br/>7,442 recordings<br/>91 actors • 6 emotions
      </div>
    </aside>

    <main className="main">
      <header className="topbar">
        <div>
          <div className="eyebrow">SPEECH EMOTION RECOGNITION</div>
          <h1>Deep Learning Model Dashboard</h1>
          <p>Speaker-independent six-class emotion recognition with comparative model analysis.</p>
        </div>
        <div className="status-pill"><span/> Test set locked</div>
      </header>

      <section id="overview" className="hero">
        <div>
          <span className="tag">FINAL EXPERIMENT</span>
          <h2>CNN-BiLSTM + Multi-Head Attention + SpecAugment</h2>
          <p>Best observed test performance under the evaluated experimental configuration.</p>
        </div>
        <div className="hero-score">
          <div className="score">{models[4].accuracy.toFixed(2)}%</div>
          <div>Test Accuracy</div>
        </div>
      </section>

      <section className="kpi-grid">
        <Metric label="Best Test Accuracy" value={models[4].accuracy.toFixed(2)} suffix="%" />
        <Metric label="Best Macro F1" value={models[4].f1.toFixed(2)} suffix="%" />
        <Metric label="Test Samples" value="1,229" />
        <Metric label="Emotion Classes" value="6" />
      </section>

      <section id="models">
        <div className="section-head">
          <div><div className="eyebrow">MODEL LIBRARY</div><h2>Six Evaluated Architectures</h2></div>
          <span className="muted">Click a card to inspect details</span>
        </div>
        <div className="model-grid">
          {models.map((m, idx) => (
            <button className={`model-card ${selected===m.id ? "active":""}`} key={m.id} onClick={()=>setSelected(m.id)}>
              <div className="card-top"><span className="model-number">0{idx+1}</span><span className="mini-status">{m.accuracy===maxAcc?"BEST":""}</span></div>
              <h3>{m.short}</h3>
              <div className="card-accuracy">{m.accuracy.toFixed(2)}%</div>
              <div className="card-label">test accuracy</div>
              <div className="tiny-progress"><span style={{width:`${(m.accuracy/65)*100}%`}}/></div>
              <p>{m.strength}</p>
            </button>
          ))}
        </div>
      </section>

      <section id="comparison" className="two-col">
        <div className="panel">
          <div className="panel-title"><div><div className="eyebrow">RESULTS</div><h2>Test Accuracy Comparison</h2></div><span className="muted">6 models</span></div>
          <AccuracyBars/>
          <div className="range-note">Observed range: <b>{minAcc.toFixed(2)}%</b> – <b>{maxAcc.toFixed(2)}%</b></div>
        </div>
        <div className="panel">
          <div className="panel-title"><div><div className="eyebrow">CLASSIFICATION</div><h2>Best Model Confusion Matrix</h2></div></div>
          <ConfusionMatrix matrix={bestConfusionMatrix}/>
          <div className="cm-note">Rows = actual emotion • Columns = predicted emotion</div>
        </div>
      </section>

      <section id="analysis" className="panel analysis">
        <div className="section-head">
          <div><div className="eyebrow">MODEL INSPECTOR</div><h2>{model.name}</h2></div>
          <select value={selected} onChange={e=>setSelected(e.target.value)}>
            {models.map(m=><option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </div>
        <div className="detail-grid">
          <div className="detail-card">
            <span>Architecture</span><strong>{model.description}</strong>
            <small>Input: {model.input}</small>
          </div>
          <div className="detail-card">
            <span>Test Performance</span>
            <div className="mini-metrics">
              <Metric label="Accuracy" value={model.accuracy.toFixed(2)} suffix="%" />
              <Metric label="Macro F1" value={model.f1.toFixed(2)} suffix="%" />
            </div>
          </div>
          <div className="detail-card">
            <span>Generalization</span>
            <div className="gap-value">{model.gap.toFixed(2)} pp</div>
            <small>Training–validation gap</small>
            <div className="compare-line"><span style={{width:`${model.train}%`}}/><i style={{width:`${model.val}%`}}/></div>
            <div className="legend"><b>Train {model.train.toFixed(2)}%</b><b>Val {model.val.toFixed(2)}%</b></div>
          </div>
          <div className="detail-card">
            <span>Model Complexity</span>
            <div className="gap-value">{model.params.toLocaleString()}</div>
            <small>Approx. trainable parameters</small>
            <div className="badge">{model.status}</div>
          </div>
        </div>
      </section>

      <section id="prediction" className="panel prediction">
        <div className="section-head">
          <div><div className="eyebrow">INTERACTIVE DEMO</div><h2>Speech Emotion Prediction</h2></div>
          <span className={`api-badge ${liveReady ? "ready":""}`}>{liveReady ? "Model 5 selected" : "Select Model 5 for live API"}</span>
        </div>
        <div className="prediction-box">
          <div className="upload">
            <div className="upload-icon">♫</div>
            <h3>Upload a WAV speech clip</h3>
            <p>Use a short audio sample for the live inference API.</p>
            <input type="file" accept=".wav,audio/wav" onChange={e=>setFile(e.target.files?.[0] || null)} />
            {file && <div className="file-name">{file.name}</div>}
            <button className="primary" disabled={!liveReady} onClick={runPrediction}>Run Emotion Prediction</button>
            <div className="api-message">{apiMessage || "Live inference requires the saved Model 5 artifact and matching preprocessing data."}</div>
          </div>
          <div className="emotion-list">
            {classes.map((c,i)=><div className="emotion-row" key={c}><span>{c}</span><div><i style={{width:`${[82,65,62,50,55,45][i]}%`}}/></div></div>)}
            <small>Probability preview is intentionally UI-only until the inference API is connected.</small>
          </div>
        </div>
      </section>

      <section id="methodology" className="method-grid">
        <div className="method-card"><span>01</span><h3>Actor-independent split</h3><p>63 train actors • 13 validation actors • 15 test actors</p></div>
        <div className="method-card"><span>02</span><h3>Acoustic features</h3><p>MFCC-based features and Log-Mel spectrograms</p></div>
        <div className="method-card"><span>03</span><h3>Evaluation</h3><p>Accuracy, precision, recall, macro F1 and confusion matrices</p></div>
        <div className="method-card"><span>04</span><h3>Leakage control</h3><p>Test set remains unseen during training and model selection</p></div>
      </section>

      <footer>
        <span>SE4050 • Deep Learning • Speech Emotion Recognition</span>
        <span>CREMA-D • 7,442 recordings • 91 actors</span>
      </footer>
    </main>
  </div>
}

export default App;
