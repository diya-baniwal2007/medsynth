import { useCallback, useEffect, useState } from 'react'
import {
  Activity, ArrowDownToLine, ArrowUpRight, Beaker, Boxes, BrainCircuit, Check,
  ChevronDown, CircleAlert, CircleHelp, Clock3, Database, Download, FlaskConical,
  HeartPulse, Image as ImageIcon, Layers3, LayoutDashboard, LoaderCircle, Menu,
  Microscope, MoreHorizontal, Play, RefreshCw, Settings2, ShieldCheck,
  Sparkles, Square, X,
} from 'lucide-react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { API_BASE, api, errorMessage } from './services/api'

const nav = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, section: 'Workspace' },
  { id: 'dataset', label: 'Dataset explorer', icon: Database },
  { id: 'studio', label: 'Image studio', icon: Sparkles },
  { id: 'training', label: 'GAN monitor', icon: Activity, section: 'Research' },
  { id: 'experiments', label: 'Experiment lab', icon: FlaskConical },
  { id: 'models', label: 'Model registry', icon: Boxes, section: 'System' },
  { id: 'about', label: 'About MedSynth', icon: CircleHelp },
]
const titles = {
  overview: ['Research overview', 'Your study, at a glance'],
  dataset: ['Dataset explorer', 'Explore the benchmark without crossing data splits'],
  studio: ['Synthetic image studio', 'Generate class-conditioned images from a trained checkpoint'],
  training: ['GAN training monitor', 'Track the generator and discriminator as they learn'],
  experiments: ['Experiment lab', 'Evaluate augmentation against the same held-out test split'],
  models: ['Model registry', 'Checkpoints and experiment models created in this workspace'],
  about: ['About MedSynth', 'Research context, method, and limitations'],
}
const classColors = ['#2eaaa2', '#596ed1']
const shortDate = (value) => value ? new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'
const timeAgo = (value) => {
  if (!value) return '—'
  const mins = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 60000))
  if (mins < 1) return 'Just now'
  if (mins < 60) return `${mins} min ago`
  const hours = Math.floor(mins / 60)
  return hours < 24 ? `${hours} hr ago` : `${Math.floor(hours / 24)} days ago`
}
const variantName = (value) => ({ baseline: 'Baseline CNN', conventional: 'Conventional augmentation', gan: 'GAN-augmented CNN' })[value] || value
const className = (label) => Number(label) === 0 ? 'Normal' : 'Pneumonia'

function App() {
  const [page, setPage] = useState('overview')
  const [mobileNav, setMobileNav] = useState(false)
  const [connection, setConnection] = useState('checking')
  const [summary, setSummary] = useState(null)
  const [activity, setActivity] = useState([])
  const [training, setTraining] = useState(null)
  const [notice, setNotice] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const notify = useCallback((message, type = 'success') => {
    setNotice({ message, type })
    window.setTimeout(() => setNotice(null), 4200)
  }, [])

  const refresh = useCallback(async () => {
    try {
      const health = await api.get('/health', { timeout: 6000 })
      setConnection(health.data.status === 'ok' ? health.data : 'offline')
    } catch { setConnection('offline') }
    try {
      const [dashboard, recent, status] = await Promise.all([
        api.get('/api/dashboard/summary'), api.get('/api/dashboard/activity'), api.get('/api/training/status'),
      ])
      setSummary(dashboard.data)
      setActivity(recent.data)
      setTraining(status.data)
    } catch (error) {
      notify(errorMessage(error), 'error')
    }
  }, [notify])

  useEffect(() => { refresh() }, [refresh, refreshKey])
  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const { data } = await api.get('/api/training/status', { timeout: 5000 })
        setTraining(data)
        if (['running', 'queued'].includes(data.status)) setRefreshKey((value) => value + 1)
      } catch { /* Connection badge reflects service availability. */ }
    }, 5000)
    return () => window.clearInterval(timer)
  }, [])

  const navigate = (id) => { setPage(id); setMobileNav(false) }
  const current = titles[page]
  const pageProps = { summary, activity, training, onRefresh: refresh, notify }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><HeartPulse size={20} strokeWidth={2.2} /></div><div><b>MedSynth</b><span>AI research workspace</span></div><button className="mobile-close icon-button" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18} /></button></div>
      <div className="workspace-switch"><div className="workspace-icon"><Microscope size={16} /></div><div><span>Current workspace</span><b>Chest X-ray study</b></div><ChevronDown size={15} /></div>
      <nav className="side-nav" aria-label="Main navigation">
        {nav.map((item) => <div key={item.id}>
          {item.section && <div className="nav-section">{item.section}</div>}
          <button className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => navigate(item.id)}>
            <item.icon size={17} strokeWidth={1.8} /><span>{item.label}</span>{item.id === 'training' && ['running', 'queued'].includes(training?.status) && <i className="nav-live" />}
          </button>
        </div>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="safety-note"><ShieldCheck size={15} /><span>Research use only<br /><b>Not for clinical decisions</b></span></div>
        <div className="profile-row"><div className="avatar">DS</div><div><b>Research workspace</b><span>Local instance</span></div><MoreHorizontal size={17} className="profile-more" /></div>
      </div>
    </aside>

    <div className="main-shell">
      <header className="topbar">
        <div className="topbar-left"><button className="mobile-menu icon-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20} /></button><div className="breadcrumbs"><span>MedSynth</span><span className="crumb-slash">/</span><b>{current[0]}</b></div></div>
        <div className="topbar-right">
          <div className={`connection-pill ${connection === 'offline' || connection === 'checking' ? 'is-muted' : ''}`}><span className="connection-dot" />{connection === 'offline' ? 'API offline' : connection === 'checking' ? 'Connecting' : 'API connected'}</div>
          <div className={`dataset-pill ${summary?.dataset?.available ? '' : 'is-muted'}`}><span className="dataset-pulse" />{summary?.dataset?.available ? 'PneumoniaMNIST ready' : 'Dataset not loaded'}</div>
          <button className="icon-button top-action" title="Refresh data" onClick={() => { setRefreshKey((value) => value + 1); refresh() }}><RefreshCw size={17} /></button>
          <div className="top-avatar">DS</div>
        </div>
      </header>

      <main className="content-area">
        <div className="page-heading"><div><div className="eyebrow">MEDSYNTH AI <span>·</span> RESEARCH PROTOTYPE</div><h1>{current[0]}</h1><p>{current[1]}</p></div>{page === 'overview' && <button className="button button-secondary" onClick={() => navigate('dataset')}><Database size={16} /> Explore dataset <ArrowUpRight size={15} /></button>}</div>
        {connection === 'offline' && <div className="banner banner-warning"><CircleAlert size={17} /><div><b>Backend not reachable</b><span>Start FastAPI from the backend folder, then refresh this page.</span></div><button className="text-button" onClick={refresh}>Retry</button></div>}
        {page === 'overview' && <Overview {...pageProps} onNavigate={navigate} />}
        {page === 'dataset' && <DatasetExplorer notify={notify} onNavigate={navigate} />}
        {page === 'studio' && <ImageStudio notify={notify} />}
        {page === 'training' && <TrainingMonitor {...pageProps} />}
        {page === 'experiments' && <ExperimentLab notify={notify} />}
        {page === 'models' && <ModelRegistry />}
        {page === 'about' && <About />}
      </main>
      <footer className="app-footer"><span>MedSynth AI <b>v1.0</b></span><span>Academic research prototype <i /> No clinical use</span></footer>
    </div>
    {mobileNav && <button className="nav-backdrop" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
    {notice && <div className={`toast ${notice.type}`}><span className="toast-icon">{notice.type === 'error' ? <CircleAlert size={17} /> : <Check size={17} />}</span>{notice.message}<button onClick={() => setNotice(null)} aria-label="Dismiss"><X size={15} /></button></div>}
  </div>
}

function Card({ children, className = '' }) { return <section className={`card ${className}`}>{children}</section> }
function SectionTitle({ title, note, action }) { return <div className="section-title"><div><h2>{title}</h2>{note && <p>{note}</p>}</div>{action}</div> }
function EmptyState({ icon: Icon = ImageIcon, title, body, action }) { return <div className="empty-state"><div className="empty-icon"><Icon size={21} /></div><b>{title}</b><p>{body}</p>{action}</div> }
function ImagePreview({ item, source = 'synthetic', onClose }) {
  return <div className="preview-backdrop" role="presentation" onClick={onClose}><div className="preview-modal" role="dialog" aria-modal="true" aria-label="Image preview" onClick={(event) => event.stopPropagation()}><button className="preview-close icon-button" onClick={onClose} aria-label="Close preview"><X size={18} /></button><div className="preview-image"><img src={`${API_BASE}${item.image_url}`} alt={`${item.class_name} ${source} chest X-ray preview`} /></div><div className="preview-details"><div><span className="eyebrow">{source === 'real' ? 'ORIGINAL DATASET IMAGE' : 'SYNTHETIC IMAGE'}</span><h2>{item.class_name} sample</h2><p>{source === 'real' ? `PneumoniaMNIST · ${item.split} split · sample ${item.index + 1}` : `Generated sample #${item.id} · ${item.model_version || 'cGAN checkpoint'}`}</p></div>{source === 'synthetic' && <a className="button button-secondary small" href={`${API_BASE}${item.download_url || item.image_url}`} download={`medsynth-${item.id}.png`}><Download size={14} /> Download</a>}</div></div></div>
}
function StatusBadge({ status }) {
  const type = ['completed', 'ready'].includes(status) ? 'good' : ['running', 'queued', 'training'].includes(status) ? 'active' : ['failed', 'blocked'].includes(status) ? 'bad' : 'neutral'
  return <span className={`status-badge ${type}`}><i />{status || 'unknown'}</span>
}
function MetricCard({ icon: Icon, label, value, hint, color, tag }) {
  return <Card className="metric-card"><div className={`metric-icon ${color}`}><Icon size={18} /></div><div className="metric-label">{label}{tag && <span className="metric-tag">{tag}</span>}</div><div className="metric-value">{value}</div><div className="metric-hint">{hint}</div></Card>
}

function Overview({ summary, activity, training, onNavigate }) {
  const dataset = summary?.dataset
  const total = dataset?.available ? dataset.total_images.toLocaleString() : '—'
  const distribution = dataset?.available ? dataset.classes.map((item) => ({ name: item.name, value: item.count })) : []
  const losses = training?.losses || []
  const latestMetrics = summary?.latest_experiment?.metrics?.at(-1)
  return <>
    <div className="welcome-strip"><div className="welcome-orb"><BrainCircuit size={22} /></div><div><b>Welcome to your research workspace</b><span>Investigate conditional synthesis across <strong>Normal</strong> and <strong>Pneumonia</strong> classes.</span></div><div className="welcome-caption"><span>ACTIVE DATASET</span><b>PneumoniaMNIST <small>28 × 28</small></b></div></div>
    <div className="metric-grid">
      <MetricCard icon={Database} label="Real images" value={total} hint={dataset?.available ? `${dataset.splits?.length || 0} official data splits` : 'Load dataset to view counts'} color="teal" />
      <MetricCard icon={Sparkles} label="Synthetic images" value={summary?.total_generated?.toLocaleString() ?? '—'} hint="Generated from trained checkpoints" color="purple" />
      <MetricCard icon={Boxes} label="Ready models" value={summary?.trained_models ?? '—'} hint="GAN and classifier checkpoints" color="blue" />
      <MetricCard icon={FlaskConical} label="Completed experiments" value={summary?.completed_experiments ?? '—'} hint="Measured on the same test split" color="orange" />
    </div>
    <div className="overview-grid">
      <Card className="chart-card distribution-card"><SectionTitle title="Class distribution" note="PneumoniaMNIST · official splits" action={<button className="subtle-link" onClick={() => onNavigate('dataset')}>View dataset <ArrowUpRight size={14} /></button>} />
        {distribution.length ? <div className="distribution-layout"><div className="pie-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={63} outerRadius={87} paddingAngle={4} stroke="none">{distribution.map((item, i) => <Cell key={item.name} fill={classColors[i]} />)}</Pie><Tooltip formatter={(v) => [Number(v).toLocaleString(), 'Images']} /></PieChart></ResponsiveContainer><div className="pie-center"><b>{dataset.total_images.toLocaleString()}</b><span>images</span></div></div><div className="distribution-legend">{distribution.map((item, i) => <div className="legend-row" key={item.name}><i style={{ background: classColors[i] }} /><div><b>{item.name}</b><span>{item.value.toLocaleString()} images</span></div><strong>{dataset.total_images ? `${Math.round(item.value / dataset.total_images * 100)}%` : '—'}</strong></div>)}</div></div> : <EmptyState icon={Database} title="Dataset not available yet" body={dataset?.error || 'The dashboard will show class counts after the dataset loads.'} action={<button className="button button-secondary small" onClick={() => onNavigate('dataset')}>Open explorer</button>} />}
      </Card>
      <Card className="chart-card loss-card"><SectionTitle title="Latest GAN losses" note={training?.id ? `Run #${training.id} · ${training.completed_epochs || 0}/${training.epochs || 0} epochs` : 'No training run yet'} action={<StatusBadge status={training?.status || 'idle'} />} />
        {losses.length ? <><div className="loss-legend"><span><i className="legend-teal" />Generator</span><span><i className="legend-blue" />Discriminator</span></div><div className="line-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={losses} margin={{ top: 8, right: 8, bottom: 0, left: -22 }}><defs><linearGradient id="genFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#36b8ae" stopOpacity={0.18} /><stop offset="95%" stopColor="#36b8ae" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9eff1" /><XAxis dataKey="epoch" tickLine={false} axisLine={false} tick={{ fill: '#93a3ad', fontSize: 11 }} tickFormatter={(v) => `E${v}`} /><YAxis tickLine={false} axisLine={false} tick={{ fill: '#93a3ad', fontSize: 11 }} /><Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #e6edef', boxShadow: '0 8px 24px #16374818' }} /><Area type="monotone" dataKey="generator_loss" name="Generator" stroke="#27a79d" strokeWidth={2.5} fill="url(#genFill)" /><Area type="monotone" dataKey="discriminator_loss" name="Discriminator" stroke="#6579db" strokeWidth={2.2} fill="none" /></AreaChart></ResponsiveContainer></div></> : <EmptyState icon={Activity} title="No loss history yet" body="Start a training run to see epoch-by-epoch generator and discriminator losses." action={<button className="button button-primary small" onClick={() => onNavigate('training')}><Play size={14} /> Configure training</button>} />}
      </Card>
      <Card className="activity-card"><SectionTitle title="Recent activity" action={<span className="live-indicator"><i /> LIVE</span>} />
        {activity?.length ? <div className="activity-list">{activity.slice(0, 6).map((item, i) => <div className="activity-row" key={`${item.type}-${item.id}`}><div className={`activity-icon ${item.type}`}>{item.type === 'training' ? <Activity size={15} /> : item.type === 'generation' ? <Sparkles size={15} /> : <FlaskConical size={15} />}</div><div className="activity-copy"><b>{item.title}</b><span><StatusBadge status={item.status} /> <em>{timeAgo(item.date)}</em></span></div><ChevronDown className="activity-chevron" size={14} /></div>)}</div> : <EmptyState icon={Clock3} title="Nothing here yet" body="Training, generation, and experiment events will appear here." />}
      </Card>
      <Card className="latest-card"><SectionTitle title="Latest experiment result" note="Most recent completed run" action={<button className="subtle-link" onClick={() => onNavigate('experiments')}>Experiment lab <ArrowUpRight size={14} /></button>} />
        {latestMetrics ? <div className="latest-result"><div className="result-name"><div className="result-type-icon"><FlaskConical size={18} /></div><div><b>{summary.latest_experiment.name}</b><span>{shortDate(summary.latest_experiment.completed_at)}</span></div><StatusBadge status={summary.latest_experiment.status} /></div><div className="mini-metrics">{[['Accuracy', latestMetrics.accuracy], ['Precision', latestMetrics.precision], ['Recall', latestMetrics.recall], ['Macro-F1', latestMetrics.macro_f1]].map(([label, value]) => <div key={label}><span>{label}</span><b>{(value * 100).toFixed(1)}<small>%</small></b></div>)}</div></div> : <EmptyState icon={Beaker} title="No measured results yet" body="Results will appear after a classifier experiment completes." action={<button className="button button-secondary small" onClick={() => onNavigate('experiments')}>View experiments</button>} />}
      </Card>
    </div>
    <div className="prototype-disclaimer"><ShieldCheck size={16} /><span><b>Academic research prototype.</b> MedSynth does not diagnose, treat, or replace professional medical judgment.</span><span className="disclaimer-version">Local research workspace</span></div>
  </>
}

function DatasetExplorer({ notify, onNavigate }) {
  const [source, setSource] = useState('real')
  const [split, setSplit] = useState('train')
  const [label, setLabel] = useState('all')
  const [data, setData] = useState(null)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const [preview, setPreview] = useState(null)
  const pageSize = 24
  const load = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const [list, meta] = await Promise.all([
        api.get('/api/dataset/images', { params: { source, split, class_label: label === 'all' ? undefined : Number(label), limit: pageSize, offset: page * pageSize } }),
        api.get('/api/dataset/summary'),
      ])
      setData(list.data); setSummary(meta.data)
    } catch (e) { setData(null); setError(errorMessage(e)) }
    finally { setLoading(false) }
  }, [source, split, label, page])
  useEffect(() => { load() }, [load])
  const setSourceValue = (value) => { setSource(value); setPage(0) }
  const currentClass = label === 'all' ? null : summary?.classes?.find((item) => item.label === Number(label))
  return <div className="page-stack">
    <div className="dataset-summary-grid">{(summary?.classes || []).map((item, i) => <Card className="class-summary" key={item.name}><div className={`class-swatch swatch-${i}`} /><div><span>{item.name}</span><b>{item.count.toLocaleString()}</b><small>{summary?.total_images ? `${(item.count / summary.total_images * 100).toFixed(1)}% of dataset` : '—'}</small></div></Card>)}<Card className="class-summary dataset-meta"><div className="meta-icon"><Layers3 size={17} /></div><div><span>Dataset splits</span><b>{summary?.splits?.length ?? '—'}</b><small>Train · validation · test kept apart</small></div></Card></div>
    <Card className="gallery-card"><div className="gallery-toolbar"><div><SectionTitle title="Image library" note={source === 'real' ? 'Original images from the official dataset partitions' : 'Images generated by a saved MedSynth cGAN checkpoint'} /></div><div className="filter-controls"><div className="segmented"><button className={source === 'real' ? 'selected' : ''} onClick={() => setSourceValue('real')}>Real images</button><button className={source === 'synthetic' ? 'selected' : ''} onClick={() => setSourceValue('synthetic')}>Synthetic</button></div>{source === 'real' && <select value={split} onChange={(e) => { setSplit(e.target.value); setPage(0) }} aria-label="Dataset split"><option value="train">Train split</option><option value="val">Validation split</option><option value="test">Test split</option></select>}<select value={label} onChange={(e) => { setLabel(e.target.value); setPage(0) }} aria-label="Class filter"><option value="all">All classes</option><option value="0">Normal</option><option value="1">Pneumonia</option></select></div></div>
      <div className="gallery-meta"><span>{source === 'real' ? <><i className="source-dot real-dot" /> Real · {split} split</> : <><i className="source-dot synthetic-dot" /> Generated · checkpoint images</>}</span><span>{data?.total != null ? `${data.total.toLocaleString()} ${currentClass ? currentClass.name.toLowerCase() + ' images' : 'images'}` : '—'}</span></div>
      {loading ? <div className="loading-state"><LoaderCircle className="spin" size={22} />Loading dataset images…</div> : error ? <EmptyState icon={CircleAlert} title="Could not load dataset" body={error} action={<button className="button button-secondary small" onClick={load}>Try again</button>} /> : data?.items?.length ? <><div className="image-grid">{data.items.map((item) => <figure className="image-tile" key={item.id ?? `${data.split}-${item.index}`}><div className="image-frame"><img className="previewable" onClick={() => setPreview({ ...item, split: data.split })} src={`${API_BASE}${item.image_url}`} alt={`${item.class_name} ${source} chest X-ray`} loading="lazy" /><span className={`class-chip ${item.class_label === 0 ? 'normal' : 'pneumonia'}`}>{item.class_name}</span></div><figcaption><span>{source === 'real' ? `Sample ${String(item.index + 1).padStart(4, '0')}` : `Generated #${item.id}`}</span><span>{source === 'real' ? `${split} split` : 'cGAN output'}</span></figcaption></figure>)}</div><div className="pagination"><span>Showing {page * pageSize + 1}–{Math.min(page * pageSize + data.items.length, data.total)} of {data.total.toLocaleString()}</span><div><button disabled={!page} onClick={() => setPage(page - 1)}>Previous</button><button disabled={(page + 1) * pageSize >= data.total} onClick={() => setPage(page + 1)}>Next <ArrowUpRight size={13} /></button></div></div></> : <EmptyState icon={ImageIcon} title={source === 'synthetic' ? 'No generated images yet' : 'No images in this selection'} body={source === 'synthetic' ? 'Train the conditional GAN and create samples in Image Studio. Real images are not shown as generated.' : 'Try another split or class filter.'} action={source === 'synthetic' ? <button className="button button-primary small" onClick={() => onNavigate('studio')}>Open Image Studio <ArrowUpRight size={13} /></button> : null} />}
    </Card>
    {preview && <ImagePreview item={preview} source={source} onClose={() => setPreview(null)} />}
  </div>
}

function ImageStudio({ notify }) {
  const [target, setTarget] = useState(0)
  const [count, setCount] = useState(8)
  const [images, setImages] = useState([])
  const [history, setHistory] = useState([])
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [checkpointReady, setCheckpointReady] = useState(null)
  const [preview, setPreview] = useState(null)
  const loadHistory = useCallback(async () => {
    try {
      const [{ data: rows }, { data: status }] = await Promise.all([api.get('/api/generation/history'), api.get('/api/models')])
      setHistory(rows); setCheckpointReady(status.some((model) => model.model_type === 'cGAN' && model.status === 'ready' && model.checkpoint_path))
    } catch { setHistory([]); setCheckpointReady(false) }
  }, [])
  useEffect(() => { loadHistory() }, [loadHistory])
  const generateImages = async () => {
    setGenerating(true); setError('')
    try {
      const { data } = await api.post('/api/generation/generate', { class_label: target, count })
      setImages(data.items); await loadHistory(); notify(`${data.count} ${className(target).toLowerCase()} images generated from the trained checkpoint.`)
    } catch (e) { setError(errorMessage(e)); notify(errorMessage(e), 'error') }
    finally { setGenerating(false) }
  }
  return <div className="page-stack">
    <div className="studio-banner"><div className="studio-icon"><Sparkles size={20} /></div><div><span className="eyebrow">CONDITIONAL SYNTHESIS</span><h2>Generate a new sample batch</h2><p>Each image is sampled from the trained generator using the selected class label and fresh latent noise.</p></div><div className="studio-model"><span>GENERATOR</span><b>{checkpointReady ? 'Checkpoint ready' : 'Training required'}</b><span className={`model-state ${checkpointReady ? 'ready' : ''}`}><i />{checkpointReady ? 'Available for synthesis' : 'No trained model found'}</span></div></div>
    <div className="studio-layout"><Card className="generation-form"><SectionTitle title="Generation settings" note="Configure your conditional batch" /><label className="field-label">TARGET CLASS</label><div className="class-options"><button onClick={() => setTarget(0)} className={target === 0 ? 'chosen' : ''}><div className="class-radio" /><div><b>Normal</b><span>Class 0 · Normal sample</span></div><div className="class-mini normal-mini"><HeartPulse size={15} /></div></button><button onClick={() => setTarget(1)} className={target === 1 ? 'chosen' : ''}><div className="class-radio" /><div><b>Pneumonia</b><span>Class 1 · Pneumonia sample</span></div><div className="class-mini pneumonia-mini"><Activity size={15} /></div></button></div><label className="field-label count-label">NUMBER OF IMAGES <span>{count} selected</span></label><div className="count-control"><button onClick={() => setCount(Math.max(1, count - 1))} disabled={count <= 1}>−</button><input type="range" min="1" max="64" value={count} onChange={(e) => setCount(Number(e.target.value))} aria-label="Number of generated images" /><button onClick={() => setCount(Math.min(64, count + 1))} disabled={count >= 64}>+</button></div><div className="range-labels"><span>1 image</span><span>64 images</span></div><div className="generation-note"><ShieldCheck size={15} /><span>Generation is enabled only after a genuine cGAN checkpoint has been saved.</span></div><button className="button button-primary generate-button" disabled={generating || !checkpointReady} onClick={generateImages}>{generating ? <><LoaderCircle size={17} className="spin" />Generating samples…</> : <><Sparkles size={17} /> Generate {count} images</>}</button>{error && <p className="inline-error">{error}</p>}</Card>
      <Card className="studio-results"><div className="results-heading"><div><h2>Generated batch</h2><p>{images.length ? `${images.length} samples · ${className(target)} · just now` : 'Your synthetic images will appear here'}</p></div>{images.length > 0 && <button className="button button-secondary small" onClick={() => images.forEach((image) => { const link = document.createElement('a'); link.href = `${API_BASE}${image.download_url}`; link.download = `medsynth-${image.id}.png`; link.click() })}><Download size={14} /> Download all</button>}</div>{images.length ? <div className="generated-grid">{images.map((image) => <div className="generated-tile" key={image.id}><img className="previewable" onClick={() => setPreview(image)} src={`${API_BASE}${image.image_url}`} alt={`Generated ${image.class_name} chest X-ray ${image.id}`} /><span>{image.class_name}</span><a href={`${API_BASE}${image.download_url}`} download={`medsynth-${image.id}.png`} title="Download image"><ArrowDownToLine size={15} /></a></div>)}</div> : <div className="studio-empty"><div className="scan-frame"><div className="scan-cross">+</div><ImageIcon size={34} strokeWidth={1.3} /><span>28 × 28 px</span></div><b>{checkpointReady ? 'Ready when you are' : 'Train your generator first'}</b><p>{checkpointReady ? 'Choose a class and generate a batch to inspect the samples.' : 'Start a GAN training run from the monitor to create the required checkpoint.'}</p></div>}</Card></div>
    <Card className="history-card"><SectionTitle title="Generation history" note="Previously saved synthetic images" action={<span className="history-count">{history.length} images</span>} />{history.length ? <div className="history-list">{history.slice(0, 6).map((image) => <div className="history-row" key={image.id}><img src={`${API_BASE}${image.image_url}`} alt={`${image.class_name} generated sample`} /><div className="history-info"><b>{image.class_name} sample <span>#{image.id}</span></b><span>{shortDate(image.created_at)}</span></div><span className="version-tag">{image.model_version}</span><a href={`${API_BASE}${image.download_url}`} download={`medsynth-${image.id}.png`} className="icon-button" title="Download image"><Download size={15} /></a></div>)}</div> : <div className="history-empty">No generations have been saved. Generated files and metadata will be recorded here.</div>}</Card>
    <div className="comparison-note"><div><div className="compare-sample sample-real"><Database size={15} /></div><b>Original data</b><span>Shown in the Dataset Explorer with official split metadata.</span></div><div className="comparison-arrow">↔</div><div><div className="compare-sample sample-generated"><Sparkles size={15} /></div><b>Synthetic data</b><span>Saved only after inference from a trained generator checkpoint.</span></div></div>
    {preview && <ImagePreview item={preview} onClose={() => setPreview(null)} />}
  </div>
}

function TrainingMonitor({ training, onRefresh, notify }) {
  const [config, setConfig] = useState({ epochs: 5, batch_size: 64, latent_dim: 100, learning_rate: 0.0002 })
  const [history, setHistory] = useState([])
  const [samples, setSamples] = useState([])
  const [busy, setBusy] = useState(false)
  const load = useCallback(async () => {
    try {
      const [runs, snapshots] = await Promise.all([api.get('/api/training/history'), api.get('/api/training/samples')])
      setHistory(runs.data); setSamples(snapshots.data)
    } catch { setHistory([]); setSamples([]) }
  }, [])
  useEffect(() => { load() }, [load, training?.status, training?.completed_epochs])
  const start = async () => {
    setBusy(true)
    try { await api.post('/api/training/start', config); notify('GAN training started in the background.'); await onRefresh(); await load() }
    catch (e) { notify(errorMessage(e), 'error') }
    finally { setBusy(false) }
  }
  const stop = async () => { try { await api.post('/api/training/stop'); notify('Stop requested. The current batch will finish first.'); await onRefresh() } catch (e) { notify(errorMessage(e), 'error') } }
  const losses = training?.losses || []
  const active = ['running', 'queued'].includes(training?.status)
  return <div className="page-stack">
    <div className="training-top"><Card className="training-progress-card"><div className="progress-head"><div><span className="eyebrow">CURRENT RUN {training?.id ? `· #${training.id}` : ''}</span><h2>{training?.status === 'running' ? 'Training in progress' : training?.status === 'completed' ? 'Training complete' : training?.status === 'failed' ? 'Training failed' : 'Ready to train'}</h2><p>{training?.started_at ? `Started ${shortDate(training.started_at)}` : 'Set your configuration and start a conditional GAN run.'}</p></div><StatusBadge status={training?.status || 'idle'} /></div><div className="progress-line"><span style={{ width: `${training?.epochs ? Math.min(100, (training.completed_epochs / training.epochs) * 100) : 0}%` }} /></div><div className="progress-stats"><span><b>{training?.completed_epochs || 0}</b> / {training?.epochs || 0} epochs</span><span>{training?.epochs ? `${Math.round((training.completed_epochs || 0) / training.epochs * 100)}% complete` : 'No run started'}</span></div>{training?.error && <div className="inline-error"><CircleAlert size={14} />{training.error}</div>}</Card>
      <Card className="training-config-card"><SectionTitle title="Configuration" note="Applied to the next run" /><div className="config-grid"><label>Epochs<input type="number" min="1" max="500" value={config.epochs} disabled={active} onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })} /></label><label>Batch size<select value={config.batch_size} disabled={active} onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}><option>16</option><option>32</option><option>64</option><option>128</option></select></label><label>Latent dimension<input type="number" min="32" max="512" step="32" value={config.latent_dim} disabled={active} onChange={(e) => setConfig({ ...config, latent_dim: Number(e.target.value) })} /></label><label>Learning rate<input type="number" min="0.00001" max="0.01" step="0.0001" value={config.learning_rate} disabled={active} onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })} /></label></div><div className="config-actions">{active ? <button className="button button-danger" onClick={stop}><Square size={15} /> Stop training</button> : <button className="button button-primary" disabled={busy} onClick={start}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} />}{busy ? 'Starting…' : training?.status === 'completed' ? 'Start another run' : 'Start training'}</button>}<span><Settings2 size={14} /> Auto device: {connectionDevice(training)}</span></div></Card></div>
    <div className="training-charts"><Card className="chart-card"><SectionTitle title="Generator loss" note="Per epoch · binary cross-entropy" /><LossSingleChart data={losses} dataKey="generator_loss" color="#26a79d" /></Card><Card className="chart-card"><SectionTitle title="Discriminator loss" note="Per epoch · real and generated batches" /><LossSingleChart data={losses} dataKey="discriminator_loss" color="#6579db" /></Card></div>
    <Card className="snapshot-card"><SectionTitle title="Generated samples by epoch" note="The same trained generator, conditioned on Normal and Pneumonia labels" action={<span className="history-count">{samples.length} snapshots</span>} />{samples.length ? <div className="snapshot-grid">{samples.slice(-6).reverse().map((sample) => <figure key={sample.epoch}><img src={`${API_BASE}${sample.image_url}`} alt={`Conditional GAN samples at epoch ${sample.epoch}`} /><figcaption><span>Epoch {sample.epoch}</span><small>{shortDate(sample.created_at)}</small></figcaption></figure>)}</div> : <EmptyState icon={ImageIcon} title="No sample grids saved yet" body="A class-labeled 16-image grid is saved after each completed GAN epoch." />}</Card>
    <Card className="checkpoint-card"><div className="checkpoint-icon"><Boxes size={18} /></div><div><b>Latest checkpoint</b><span>{training?.checkpoint_path ? `${training.model_version || 'cGAN'} · ${training.checkpoint_path.split(/[\\/]/).at(-1)}` : 'A checkpoint is saved after the first completed epoch.'}</span></div><div className="checkpoint-value">{training?.checkpoint_path ? `${training.completed_epochs} saved epoch${training.completed_epochs === 1 ? '' : 's'}` : 'No checkpoint yet'}</div></Card>
    <Card className="history-card"><SectionTitle title="Training history" note="Persisted training runs" />{history.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Run</th><th>Configuration</th><th>Epochs</th><th>Generator</th><th>Discriminator</th><th>Status</th><th>Started</th></tr></thead><tbody>{history.map((run) => <tr key={run.id}><td><b>#{run.id}</b></td><td>{run.config?.latent_dim ?? '—'} latent · bs {run.config?.batch_size ?? '—'}</td><td>{run.completed_epochs}/{run.epochs}</td><td>{run.generator_loss?.toFixed(4) ?? '—'}</td><td>{run.discriminator_loss?.toFixed(4) ?? '—'}</td><td><StatusBadge status={run.status} /></td><td>{shortDate(run.started_at)}</td></tr>)}</tbody></table></div> : <div className="history-empty">No training runs have been recorded yet.</div>}</Card>
  </div>
}
function connectionDevice(training) { return training?.config?.device || 'CUDA when available, otherwise CPU' }
function LossSingleChart({ data, dataKey, color }) { return data?.length ? <div className="single-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: -16 }}><defs><linearGradient id={`fill-${dataKey}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.2} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9eff1" /><XAxis dataKey="epoch" tickLine={false} axisLine={false} tick={{ fill: '#93a3ad', fontSize: 11 }} tickFormatter={(v) => `E${v}`} /><YAxis tickLine={false} axisLine={false} tick={{ fill: '#93a3ad', fontSize: 11 }} /><Tooltip /><Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#fill-${dataKey})`} strokeWidth={2.5} /></AreaChart></ResponsiveContainer></div> : <EmptyState icon={Activity} title="No epoch data" body="Loss curves appear here as training progress is saved." /> }

function ConfusionMatrix({ run }) {
  const matrix = run.metrics?.at(-1)?.confusion_matrix
  return <div className="confusion-panel"><b>{variantName(run.augmentation_type)}</b><div className="cm-axis">True class <span>Predicted →</span></div><div className="cm-table"><span className="cm-head">Class</span><span className="cm-head">Normal</span><span className="cm-head">Pneumonia</span><b>Normal</b><strong>{matrix?.[0]?.[0] ?? '—'}</strong><strong>{matrix?.[0]?.[1] ?? '—'}</strong><b>Pneumonia</b><strong>{matrix?.[1]?.[0] ?? '—'}</strong><strong>{matrix?.[1]?.[1] ?? '—'}</strong></div></div>
}

function ExperimentLossPlot({ run }) {
  const metric = run.metrics?.at(-1)
  const data = (metric?.training_loss || []).map((value, index) => ({ epoch: index + 1, training: value, validation: metric.validation_loss?.[index] }))
  return <div className="experiment-loss-plot"><div className="loss-plot-title"><b>{variantName(run.augmentation_type)}</b><span>Train <i className="loss-key train-key" /> Validation <i className="loss-key validation-key" /></span></div>{data.length ? <ResponsiveContainer width="100%" height={82}><AreaChart data={data} margin={{ top: 4, right: 3, bottom: 0, left: -27 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#edf1f2" /><XAxis dataKey="epoch" tickLine={false} axisLine={false} tick={{ fill: '#9aa8ac', fontSize: 8 }} tickFormatter={(v) => `E${v}`} /><YAxis tickLine={false} axisLine={false} tick={{ fill: '#9aa8ac', fontSize: 8 }} /><Tooltip formatter={(v) => Number(v).toFixed(3)} /><Area type="monotone" dataKey="training" stroke="#2ca89c" fill="#2ca89c" fillOpacity={0.09} strokeWidth={1.8} /><Area type="monotone" dataKey="validation" stroke="#6c7ed0" fill="#6c7ed0" fillOpacity={0.07} strokeWidth={1.8} /></AreaChart></ResponsiveContainer> : <span className="loss-empty">No loss history yet.</span>}</div>
}

function ExperimentLab({ notify }) {
  const [runs, setRuns] = useState([])
  const [epochs, setEpochs] = useState(5)
  const [repeats, setRepeats] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(async () => {
    try { setRuns((await api.get('/api/experiments')).data) }
    catch (e) { setError(errorMessage(e)) }
  }, [])
  useEffect(() => { load() }, [load])
  useEffect(() => { const timer = setInterval(load, 5000); return () => clearInterval(timer) }, [load])
  const active = runs.some((run) => ['queued', 'running'].includes(run.status))
  const execute = async () => {
    setBusy(true)
    try {
      const { data } = await api.post('/api/experiments/run', {
        epochs, batch_size: 64, seed: 42, repeats, variants: ['baseline', 'conventional', 'gan'],
      })
      notify(`${data.items.length} runs queued across ${repeats} seed${repeats === 1 ? '' : 's'}.`)
      await load()
    } catch (e) { notify(errorMessage(e), 'error') }
    finally { setBusy(false) }
  }

  const variants = ['baseline', 'conventional', 'gan']
  const latestBatchId = runs.find((run) => run.config?.batch_id)?.config.batch_id
  const batchRows = latestBatchId ? runs.filter((run) => run.config?.batch_id === latestBatchId) : []
  const summarize = (variant) => {
    const group = latestBatchId
      ? batchRows.filter((run) => run.augmentation_type === variant)
      : runs.filter((run) => run.augmentation_type === variant).slice(0, 1)
    const completedRuns = group.filter((run) => run.status === 'completed' && run.metrics?.length)
    const latest = group[0] || runs.find((run) => run.augmentation_type === variant)
    const metric = (key) => {
      const values = completedRuns.map((run) => Number(run.metrics.at(-1)[key])).filter(Number.isFinite)
      if (!values.length) return null
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length
      const standardDeviation = values.length > 1
        ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1))
        : 0
      return { mean, standardDeviation, count: values.length }
    }
    return {
      group, completedRuns, latest,
      running: group.some((run) => ['queued', 'running'].includes(run.status)),
      allCompleted: group.length > 0 && group.every((run) => run.status === 'completed'),
      accuracy: metric('accuracy'), precision: metric('precision'), recall: metric('recall'), macroF1: metric('macro_f1'),
    }
  }
  const summaries = Object.fromEntries(variants.map((variant) => [variant, summarize(variant)]))
  const diagnostics = variants.map((variant) => summaries[variant].completedRuns[0]).filter(Boolean)
  const chartData = variants.map((variant) => {
    const summary = summaries[variant]
    if (!summary.macroF1) return null
    return {
      name: variant === 'conventional' ? 'Conventional' : variant === 'baseline' ? 'Baseline' : 'GAN augmented',
      Accuracy: summary.accuracy.mean * 100,
      'Macro-F1': summary.macroF1.mean * 100,
    }
  }).filter(Boolean)
  const formatMeanSd = (metric) => metric
    ? `${(metric.mean * 100).toFixed(1)} +/- ${metric.count > 1 ? (metric.standardDeviation * 100).toFixed(1) : '--'}`
    : '--'

  return <div className="page-stack">
    <div className="experiment-intro">
      <div>
        <div className="eyebrow">CONTROLLED COMPARISON</div>
        <h2>Does augmentation change classifier performance?</h2>
        <p>Each seed trains all three CNN configurations on the same partitions. The official test split remains reserved for evaluation; multi-seed results are summarized with mean and sample standard deviation.</p>
      </div>
      <div className="experiment-controls">
        <label>Epochs<input type="number" min="1" max="100" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} disabled={active} /></label>
        <label>Seed repeats<select value={repeats} onChange={(e) => setRepeats(Number(e.target.value))} disabled={active}><option value={1}>1 seed</option><option value={3}>3 seeds</option><option value={5}>5 seeds</option></select></label>
        <button className="button button-primary" onClick={execute} disabled={busy || active}>
          {busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} />}
          {active ? 'Batch running...' : busy ? 'Starting...' : `Run ${3 * repeats} experiments`}
        </button>
      </div>
    </div>

    <div className="experiment-cards">
      {variants.map((variant, i) => {
        const summary = summaries[variant]
        const run = summary.group.find((row) => ['running', 'queued'].includes(row.status)) || summary.latest
        const status = summary.running ? 'running' : summary.allCompleted ? 'completed' : run?.status || 'pending'
        const metric = summary.macroF1
        return <Card key={variant} className={`experiment-card exp-${i}`}>
          <div className="exp-card-head">
            <div className={`exp-icon exp-icon-${i}`}>{i === 0 ? <BrainCircuit size={17} /> : i === 1 ? <Activity size={17} /> : <Sparkles size={17} />}</div>
            <StatusBadge status={status} />
          </div>
          <h3>{variantName(variant)}</h3>
          <p>{i === 0 ? 'Original training data only' : i === 1 ? 'Original data with mild geometric transforms' : 'Original data plus checkpoint-generated minority class'}</p>
          {metric ? <>
            <div className="exp-score"><b>{(metric.mean * 100).toFixed(1)}<small>%</small></b><span>Mean Macro-F1, n={metric.count}</span></div>
            {metric.count > 1 && <div className="exp-stats"><span>Sample SD <b>+/-{(metric.standardDeviation * 100).toFixed(1)} pp</b></span></div>}
          </> : <div className="exp-pending">
            {run?.status === 'running' ? `Epoch ${run.config?.progress?.epoch || 0} of ${run.config?.progress?.total_epochs || epochs}` : run?.status === 'blocked' ? 'Requires a trained cGAN checkpoint' : run?.status === 'failed' ? 'Run failed; inspect its history record' : 'No measured result yet'}
          </div>}
          {summary.accuracy && <div className="exp-stats"><span>Mean accuracy <b>{(summary.accuracy.mean * 100).toFixed(1)}%</b></span><span>Mean precision <b>{(summary.precision.mean * 100).toFixed(1)}%</b></span></div>}
        </Card>
      })}
    </div>

    <div className="overview-grid experiment-grid">
      <Card className="chart-card comparison-chart-card">
        <SectionTitle title="Measured test metrics" note={latestBatchId ? 'Mean metrics from completed seeds in the latest batch' : 'Only completed experiments are plotted'} />
        {chartData.length ? <>
          <div className="bar-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 10, right: 8, bottom: 2, left: -18 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9eff1" /><XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#748892', fontSize: 11 }} /><YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: '#93a3ad', fontSize: 11 }} /><Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} /><Legend /><Bar dataKey="Accuracy" fill="#36aaa2" radius={[5, 5, 0, 0]} /><Bar dataKey="Macro-F1" fill="#6778d4" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div>
          <div className="table-scroll seed-summary-scroll"><small className="table-date">SD is in percentage points; a single seed has no SD.</small><table className="data-table seed-summary-table"><thead><tr><th>Variant</th><th>Seeds</th><th>Accuracy % mean +/- SD</th><th>Precision % mean +/- SD</th><th>Recall % mean +/- SD</th><th>Macro-F1 % mean +/- SD</th></tr></thead><tbody>{variants.map((variant) => {
            const summary = summaries[variant]
            return <tr key={variant}><td>{variantName(variant)}</td><td>{summary.macroF1?.count ?? 0}</td><td>{formatMeanSd(summary.accuracy)}</td><td>{formatMeanSd(summary.precision)}</td><td>{formatMeanSd(summary.recall)}</td><td>{formatMeanSd(summary.macroF1)}</td></tr>
          })}</tbody></table></div>
        </> : <EmptyState icon={FlaskConical} title="Waiting for completed experiments" body="The chart fills with measured values after evaluation finishes." />}
      </Card>
      <Card className="method-card">
        <SectionTitle title="Evaluation protocol" />
        <div className="method-list">
          <div><span>01</span><div><b>Train split</b><p>Fit model weights only on official training samples.</p></div></div>
          <div><span>02</span><div><b>Validation split</b><p>Select the checkpoint with the lowest validation loss.</p></div></div>
          <div><span>03</span><div><b>Test split</b><p>Evaluate once per completed run; never used for selection.</p></div></div>
        </div>
        <div className="method-foot"><ShieldCheck size={15} /> Results summarize completed seeds within the latest batch.</div>
      </Card>
    </div>

    <div className="overview-grid experiment-grid">
      <Card className="matrix-card"><SectionTitle title="Test confusion matrices" note="Latest completed seed for each variant; rows are true class" />{diagnostics.length ? <div className="matrix-collection">{diagnostics.map((run) => <ConfusionMatrix key={run.id} run={run} />)}</div> : <EmptyState icon={Activity} title="No confusion matrices yet" body="Matrices are created from measured test predictions." />}</Card>
      <Card className="loss-history-card"><SectionTitle title="Training & validation loss" note="Latest completed seed for each variant" />{diagnostics.length ? <div className="loss-plot-list">{diagnostics.map((run) => <ExperimentLossPlot key={run.id} run={run} />)}</div> : <EmptyState icon={Activity} title="No loss histories yet" body="Loss curves are saved with each completed experiment." />}</Card>
    </div>

    <Card className="history-card">
      <SectionTitle title="Experiment history" note="Completed, pending, blocked, and failed runs" action={<span className="history-count">{runs.length} records</span>} />
      {error && <p className="inline-error">{error}</p>}
      {runs.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Experiment</th><th>Variant</th><th>Seed</th><th>Epoch</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Macro-F1</th><th>Status</th></tr></thead><tbody>{runs.map((run) => {
        const metric = run.metrics?.at(-1)
        return <tr key={run.id}><td><b>{run.name}</b><small className="table-date">{shortDate(run.created_at)}</small></td><td>{run.augmentation_type}</td><td>{run.config?.seed ?? '--'}</td><td>{run.config?.progress?.epoch || (run.status === 'completed' ? run.config?.epochs : 0) || '--'}/{run.config?.epochs || '--'}</td><td>{metric ? `${(metric.accuracy * 100).toFixed(2)}%` : '--'}</td><td>{metric ? `${(metric.precision * 100).toFixed(2)}%` : '--'}</td><td>{metric ? `${(metric.recall * 100).toFixed(2)}%` : '--'}</td><td>{metric ? `${(metric.macro_f1 * 100).toFixed(2)}%` : '--'}</td><td title={run.error || undefined}><StatusBadge status={run.status} />{run.error && <small className="table-date">{run.error}</small>}</td></tr>
      })}</tbody></table></div> : <div className="history-empty">No experiments have been run. Start a batch to create records.</div>}
    </Card>
  </div>
}
function ModelRegistry() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const load = useCallback(async () => { setLoading(true); try { setModels((await api.get('/api/models')).data); setError('') } catch (e) { setError(errorMessage(e)) } finally { setLoading(false) } }, [])
  useEffect(() => { load() }, [load])
  return <div className="page-stack"><div className="registry-intro"><div className="registry-icon"><Boxes size={21} /></div><div><h2>Research model inventory</h2><p>Models are recorded when a training or experiment run saves its checkpoint.</p></div><div className="registry-total"><b>{models.length}</b><span>model records</span></div></div><Card className="history-card"><SectionTitle title="Registered models" note="Local checkpoint paths and configurations" action={<button className="button button-secondary small" onClick={load}><RefreshCw size={14} /> Refresh</button>} />{loading ? <div className="loading-state"><LoaderCircle className="spin" size={22} />Loading registry…</div> : error ? <EmptyState icon={CircleAlert} title="Registry unavailable" body={error} action={<button className="button button-secondary small" onClick={load}>Retry</button>} /> : models.length ? <div className="table-scroll"><table className="data-table model-table"><thead><tr><th>Model</th><th>Type</th><th>Version</th><th>Status</th><th>Created</th><th>Checkpoint</th></tr></thead><tbody>{models.map((model) => <tr key={model.id}><td><div className="model-name"><span className={model.model_type === 'cGAN' ? 'gan-model' : 'cnn-model'}>{model.model_type === 'cGAN' ? <Sparkles size={15} /> : <BrainCircuit size={15} />}</span><b>{model.name}</b></div></td><td><span className="type-pill">{model.model_type}</span></td><td><code>{model.version}</code></td><td><StatusBadge status={model.status} /></td><td>{shortDate(model.created_at)}</td><td className="path-cell" title={model.checkpoint_path}>{model.checkpoint_path || 'Pending checkpoint'}</td></tr>)}</tbody></table></div> : <EmptyState icon={Boxes} title="No models registered" body="Run the conditional GAN or a classifier experiment to register a model checkpoint." />}</Card><div className="registry-foot"><CircleAlert size={15} /><span>Checkpoint files stay in local storage. Registry entries include the actual path saved by each run.</span></div></div>
}

function About() {
  return <div className="page-stack about-page"><div className="about-hero"><div className="about-emblem"><HeartPulse size={28} /></div><div><div className="eyebrow">ACADEMIC TRACK A · GENERATIVE ADVERSARIAL NETWORKS</div><h2>Studying synthesis for imbalanced medical imaging</h2><p>MedSynth examines whether class-conditional synthetic chest X-ray images can support data augmentation for a binary classifier.</p></div></div><div className="about-grid"><Card className="about-card"><div className="about-card-icon teal"><Beaker size={18} /></div><h3>Research motivation</h3><p>Medical imaging datasets can contain uneven class distributions. This project makes it possible to inspect a conditional generator and compare classifier outcomes with and without augmentation.</p></Card><Card className="about-card"><div className="about-card-icon blue"><Database size={18} /></div><h3>Dataset</h3><p>PneumoniaMNIST provides 28 × 28 grayscale images with official training, validation, and test splits. The application keeps these partitions separate for training, model selection, and final evaluation.</p></Card><Card className="about-card"><div className="about-card-icon purple"><BrainCircuit size={18} /></div><h3>Model approach</h3><p>A label-conditional DCGAN learns from image and class-label pairs. Its generator combines latent noise with an embedding of the requested class, while a conditional discriminator scores image-label pairs.</p></Card><Card className="about-card"><div className="about-card-icon orange"><FlaskConical size={18} /></div><h3>Experimental method</h3><p>A small CNN is trained with original data, conventional geometric augmentation, and generator-produced minority-class images. Macro-F1, accuracy, precision, recall, loss, and confusion matrices are measured on the same test split.</p></Card><Card className="about-card about-limitations"><div className="about-card-icon red"><ShieldCheck size={18} /></div><h3>Limitations & responsible use</h3><p>This is an educational research prototype, not a diagnostic product. Low-resolution data cannot represent clinical X-ray workflows; synthetic outputs may contain artifacts; and an apparent score change needs repeated experiments and a stronger study design before any research claim.</p><div className="clinical-warning"><CircleAlert size={15} /><span>Never use MedSynth outputs for patient care or clinical decisions.</span></div></Card><Card className="about-card"><div className="about-card-icon teal"><ArrowUpRight size={18} /></div><h3>Future scope</h3><p>Repeat experiments across random seeds, quantify image quality and diversity, compare multiple balancing strategies, and validate conclusions on higher-resolution datasets with appropriate oversight.</p></Card></div><div className="about-meta"><span>Course <b>Deep Learning and Generative AI</b></span><span>Project <b>Application-oriented research prototype</b></span><span>Backend <b>FastAPI · PyTorch</b></span></div></div>
}

export default App
