import { useEffect, useMemo, useRef, useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  AudioLines,
  AudioWaveform,
  BookOpen,
  ChevronDown,
  CircleHelp,
  Download,
  FileAudio,
  FilePlus2,
  Gauge,
  Languages,
  ListMusic,
  Moon,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  Redo2,
  RefreshCw,
  Save,
  Settings2,
  Sparkles,
  Square,
  Sun,
  Trash2,
  Undo2,
  Upload,
  WandSparkles,
} from "lucide-react";
import "./App.css";

type Language = "zh" | "en";
type Speed = "slow" | "natural" | "fast";
type SegmentStatus = "pending" | "generating" | "done" | "error";

type Segment = {
  id: number;
  text: string;
  speed: Speed;
  prompt: string;
  pauseMs: number;
  status: SegmentStatus;
  progress: number;
  duration?: string;
};

const INITIAL_SEGMENTS: Segment[] = [
  {
    id: 1,
    text: "在地球最北端，冬季并不是一个季节，而是一场漫长的考验。",
    speed: "slow",
    prompt: "沉静、克制的纪录片旁白，声音低沉，语气舒缓。",
    pauseMs: 1200,
    status: "done",
    progress: 100,
    duration: "00:09",
  },
  {
    id: 2,
    text: "太阳在地平线下停留数月，极夜笼罩着冰原，也改变了这里所有生命的节奏。",
    speed: "slow",
    prompt: "保持同一音色，带有轻微的神秘感，句尾自然收束。",
    pauseMs: 1400,
    status: "done",
    progress: 100,
    duration: "00:13",
  },
  {
    id: 3,
    text: "然而，就在看似沉寂的雪层之下，新的迁徙已经开始。",
    speed: "natural",
    prompt: "保持克制，情绪逐渐明亮，适度加强“新的迁徙”。",
    pauseMs: 1000,
    status: "generating",
    progress: 62,
  },
  {
    id: 4,
    text: "这是北极狐一年中最重要的旅程。",
    speed: "natural",
    prompt: "清晰、专注，保持前文的音色质感。",
    pauseMs: 1100,
    status: "pending",
    progress: 0,
  },
];

const COPY = {
  zh: {
    project: "极地迁徙 · 第一集",
    workspace: "导演工作台",
    script: "长文本稿件",
    director: "导演表",
    render: "合成与导出",
    library: "项目",
    recent: "最近项目",
    chapters: "稿件章节",
    addChapter: "添加章节",
    importScript: "导入稿件",
    generateTable: "智能生成导演表",
    segmentCount: "4 个分段",
    tableHint: "双击编辑文字 · 回车拆分 · 行首退格合并",
    generateAll: "生成全部",
    stop: "停止任务",
    mergeExport: "合并导出",
    inspector: "分段设置",
    selected: "当前分段",
    text: "文本",
    speed: "语速",
    prompt: "表演提示词",
    pause: "段后停顿",
    voice: "参考声音",
    voiceName: "纪录片男声 01",
    regenerate: "重新生成",
    modelReady: "VoxCPM2 已就绪",
    saved: "已自动保存",
    overall: "生成进度",
    complete: "2 / 4 已完成",
    addSegment: "添加分段",
    deleteSegment: "删除分段",
    play: "试听",
    download: "下载",
    more: "更多操作",
    settings: "设置",
    help: "操作技巧",
    theme: "切换主题",
    language: "切换为英文",
    save: "保存项目",
    undo: "撤销",
    redo: "重做",
    reference: "声音参考",
    natural: "自然",
    slow: "缓慢",
    fast: "较快",
    ms: "毫秒",
    idle: "等待生成",
    processing: "正在生成",
    ready: "可试听",
    failed: "需要重试",
    chapter1: "01  极夜",
    chapter2: "02  迁徙",
    chapter3: "03  冰原",
    newProject: "新建项目",
    totalDuration: "预计成片 00:47",
  },
  en: {
    project: "Polar Migration · Episode 1",
    workspace: "Director Workspace",
    script: "Long Script",
    director: "Director Table",
    render: "Render & Export",
    library: "Projects",
    recent: "Recent Projects",
    chapters: "Script Chapters",
    addChapter: "Add chapter",
    importScript: "Import script",
    generateTable: "Build Director Table",
    segmentCount: "4 segments",
    tableHint: "Double-click to edit · Enter to split · Backspace at start to merge",
    generateAll: "Generate all",
    stop: "Stop task",
    mergeExport: "Merge & export",
    inspector: "Segment Inspector",
    selected: "Selected segment",
    text: "Text",
    speed: "Pacing",
    prompt: "Performance direction",
    pause: "Pause after segment",
    voice: "Reference voice",
    voiceName: "Documentary Voice 01",
    regenerate: "Regenerate",
    modelReady: "VoxCPM2 ready",
    saved: "Autosaved",
    overall: "Generation progress",
    complete: "2 / 4 complete",
    addSegment: "Add segment",
    deleteSegment: "Delete segment",
    play: "Preview",
    download: "Download",
    more: "More actions",
    settings: "Settings",
    help: "Tips",
    theme: "Toggle theme",
    language: "Switch to Chinese",
    save: "Save project",
    undo: "Undo",
    redo: "Redo",
    reference: "Voice reference",
    natural: "Natural",
    slow: "Slow",
    fast: "Fast",
    ms: "ms",
    idle: "Waiting",
    processing: "Generating",
    ready: "Ready",
    failed: "Retry needed",
    chapter1: "01  Polar Night",
    chapter2: "02  Migration",
    chapter3: "03  Ice Field",
    newProject: "New project",
    totalDuration: "Est. duration 00:47",
  },
} as const;

const WAVE_BARS = [8, 15, 11, 22, 31, 18, 26, 38, 28, 17, 24, 34, 42, 30, 19, 13, 28, 37, 25, 16, 32, 44, 35, 23, 14, 28, 39, 30, 18, 11, 22, 33];

function IconButton({
  label,
  children,
  onClick,
  active = false,
  disabled = false,
}: {
  label: string;
  children: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <button
          type="button"
          className={`icon-button${active ? " active" : ""}`}
          aria-label={label}
          onClick={onClick}
          disabled={disabled}
        >
          {children}
        </button>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip" sideOffset={6}>
          {label}
          <Tooltip.Arrow className="tooltip-arrow" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function WaveformBars({ active = false }: { active?: boolean }) {
  return (
    <div className={`waveform-bars${active ? " playing" : ""}`} aria-hidden="true">
      {WAVE_BARS.map((height, index) => (
        <span key={index} style={{ height: `${height}px`, animationDelay: `${index * 24}ms` }} />
      ))}
    </div>
  );
}

function App() {
  const [language, setLanguage] = useState<Language>("zh");
  const [dark, setDark] = useState(false);
  const [segments, setSegments] = useState(INITIAL_SEGMENTS);
  const [selectedId, setSelectedId] = useState(3);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [isBatchGenerating, setIsBatchGenerating] = useState(false);
  const [activeNav, setActiveNav] = useState("director");
  const timers = useRef<number[]>([]);
  const t = COPY[language];

  const selected = segments.find((segment) => segment.id === selectedId) ?? segments[0];
  const completeCount = segments.filter((segment) => segment.status === "done").length;
  const overallProgress = Math.round(
    segments.reduce((sum, segment) => sum + segment.progress, 0) / Math.max(segments.length, 1),
  );

  const statusCopy = useMemo(
    () => ({ pending: t.idle, generating: t.processing, done: t.ready, error: t.failed }),
    [t],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    return () => timers.current.forEach((timer) => window.clearInterval(timer));
  }, [dark]);

  const updateSegment = (id: number, patch: Partial<Segment>) => {
    setSegments((current) => current.map((segment) => (segment.id === id ? { ...segment, ...patch } : segment)));
  };

  const renumber = (items: Segment[]) => items.map((item, index) => ({ ...item, id: index + 1 }));

  const splitSegment = (id: number, before: string, after: string) => {
    if (!before.trim() || !after.trim()) return;
    setSegments((current) => {
      const index = current.findIndex((item) => item.id === id);
      const source = current[index];
      const next = [
        ...current.slice(0, index),
        { ...source, text: before.trim(), status: "pending" as const, progress: 0, duration: undefined },
        {
          ...source,
          id: source.id + 1,
          text: after.trim(),
          status: "pending" as const,
          progress: 0,
          duration: undefined,
        },
        ...current.slice(index + 1),
      ];
      return renumber(next);
    });
  };

  const mergePrevious = (id: number) => {
    if (id <= 1) return;
    setSegments((current) => {
      const index = current.findIndex((item) => item.id === id);
      const previous = current[index - 1];
      const source = current[index];
      const next = [
        ...current.slice(0, index - 1),
        {
          ...previous,
          text: `${previous.text}${source.text}`,
          status: "pending" as const,
          progress: 0,
          duration: undefined,
        },
        ...current.slice(index + 1),
      ];
      setSelectedId(Math.max(1, id - 1));
      return renumber(next);
    });
  };

  const deleteSegment = (id: number) => {
    setSegments((current) => {
      if (current.length === 1) return current;
      const next = renumber(current.filter((item) => item.id !== id));
      setSelectedId(Math.min(id, next.length));
      return next;
    });
  };

  const addSegment = () => {
    const nextId = segments.length + 1;
    setSegments((current) => [
      ...current,
      {
        id: nextId,
        text: language === "zh" ? "在这里输入新的旁白分段。" : "Enter a new narration segment here.",
        speed: "natural",
        prompt: language === "zh" ? "保持同一音色，自然讲述。" : "Keep the same voice and narrate naturally.",
        pauseMs: 1000,
        status: "pending",
        progress: 0,
      },
    ]);
    setSelectedId(nextId);
  };

  const simulateGeneration = (id: number) => {
    timers.current.forEach((timer) => window.clearInterval(timer));
    timers.current = [];
    updateSegment(id, { status: "generating", progress: 6, duration: undefined });
    const timer = window.setInterval(() => {
      setSegments((current) =>
        current.map((segment) => {
          if (segment.id !== id) return segment;
          const next = Math.min(100, segment.progress + 7 + Math.round(Math.random() * 9));
          if (next >= 100) {
            window.clearInterval(timer);
            return { ...segment, progress: 100, status: "done", duration: `00:${8 + (id % 6)}` };
          }
          return { ...segment, progress: next, status: "generating" };
        }),
      );
    }, 240);
    timers.current.push(timer);
  };

  const generateAll = () => {
    if (isBatchGenerating) {
      timers.current.forEach((timer) => window.clearInterval(timer));
      timers.current = [];
      setIsBatchGenerating(false);
      return;
    }
    setIsBatchGenerating(true);
    setSegments((current) => current.map((segment) => ({ ...segment, status: "pending", progress: 0 })));
    segments.forEach((segment, index) => {
      const startTimer = window.setTimeout(() => {
        simulateGeneration(segment.id);
        if (index === segments.length - 1) {
          const doneTimer = window.setTimeout(() => setIsBatchGenerating(false), 3400);
          timers.current.push(doneTimer);
        }
      }, index * 3500);
      timers.current.push(startTimer);
    });
  };

  return (
    <Tooltip.Provider delayDuration={350}>
      <div className="app-shell">
        <header className="titlebar">
          <div className="brand-lockup">
            <img className="brand-mark" src="/dubcue-mark.png" alt="" aria-hidden="true" />
            <strong>DubCue</strong>
          </div>

          <button className="project-switcher" type="button">
            <span>{t.project}</span>
            <ChevronDown size={14} />
          </button>

          <div className="titlebar-status">
            <span className="save-status"><span className="save-dot" />{t.saved}</span>
            <span className="model-status"><span className="model-dot" />{t.modelReady}</span>
          </div>

          <div className="window-actions">
            <IconButton label={t.undo}><Undo2 size={16} /></IconButton>
            <IconButton label={t.redo}><Redo2 size={16} /></IconButton>
            <span className="toolbar-separator" />
            <IconButton label={t.language} onClick={() => setLanguage(language === "zh" ? "en" : "zh")}>
              <Languages size={16} />
            </IconButton>
            <IconButton label={t.theme} onClick={() => setDark((value) => !value)}>
              {dark ? <Sun size={16} /> : <Moon size={16} />}
            </IconButton>
            <IconButton label={t.settings}><Settings2 size={16} /></IconButton>
          </div>
        </header>

        <aside className="sidebar">
          <button className="new-project-button" type="button">
            <FilePlus2 size={16} />
            {t.newProject}
          </button>

          <nav className="primary-nav" aria-label="Workspace">
            <button type="button" className={activeNav === "script" ? "active" : ""} onClick={() => setActiveNav("script")}>
              <BookOpen size={17} />{t.script}
            </button>
            <button type="button" className={activeNav === "director" ? "active" : ""} onClick={() => setActiveNav("director")}>
              <ListMusic size={17} />{t.director}
              <span className="nav-count">{segments.length}</span>
            </button>
            <button type="button" className={activeNav === "render" ? "active" : ""} onClick={() => setActiveNav("render")}>
              <FileAudio size={17} />{t.render}
            </button>
          </nav>

          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>{t.chapters}</span>
              <IconButton label={t.addChapter}><Plus size={14} /></IconButton>
            </div>
            <button className="chapter active" type="button"><span>{t.chapter1}</span><span>4</span></button>
            <button className="chapter" type="button"><span>{t.chapter2}</span><span>7</span></button>
            <button className="chapter" type="button"><span>{t.chapter3}</span><span>5</span></button>
          </div>

          <div className="sidebar-footer">
            <button type="button"><CircleHelp size={16} />{t.help}</button>
            <button type="button"><Save size={16} />{t.save}<span className="shortcut">⌘S</span></button>
          </div>
        </aside>

        <main className="workspace">
          <section className="workspace-heading">
            <div>
              <div className="eyebrow">{t.project}</div>
              <h1>{t.workspace}</h1>
            </div>
            <div className="heading-actions">
              <button className="button secondary" type="button"><Upload size={16} />{t.importScript}</button>
              <button className="button secondary" type="button"><WandSparkles size={16} />{t.generateTable}</button>
              <button className="button primary" type="button" onClick={generateAll}>
                {isBatchGenerating ? <Square size={15} /> : <Sparkles size={16} />}
                {isBatchGenerating ? t.stop : t.generateAll}
              </button>
            </div>
          </section>

          <div className="editor-layout">
            <section className="director-surface">
              <div className="surface-toolbar">
                <div className="toolbar-copy">
                  <h2>{t.director}</h2>
                  <span>{segments.length} {language === "zh" ? "个分段" : "segments"}</span>
                </div>
                <div className="surface-tools">
                  <span className="keyboard-hint">{t.tableHint}</span>
                  <IconButton label={t.addSegment} onClick={addSegment}><Plus size={16} /></IconButton>
                  <IconButton label={t.more}><MoreHorizontal size={17} /></IconButton>
                </div>
              </div>

              <div className="director-table" role="table" aria-label={t.director}>
                <div className="table-header" role="row">
                  <div>#</div>
                  <div>{t.text}</div>
                  <div>{t.speed}</div>
                  <div>{t.prompt}</div>
                  <div>{t.pause}</div>
                  <div>{language === "zh" ? "分段音频" : "Segment audio"}</div>
                </div>

                <div className="table-body">
                  {segments.map((segment) => {
                    const isSelected = segment.id === selectedId;
                    const isPlaying = segment.id === playingId;
                    return (
                      <div
                        className={`segment-row${isSelected ? " selected" : ""}`}
                        role="row"
                        key={segment.id}
                        onClick={() => setSelectedId(segment.id)}
                      >
                        <div className="row-number">{String(segment.id).padStart(2, "0")}</div>
                        <div className="text-cell">
                          <textarea
                            value={segment.text}
                            aria-label={`${t.text} ${segment.id}`}
                            onChange={(event) => updateSegment(segment.id, { text: event.target.value, status: "pending", progress: 0 })}
                            onKeyDown={(event) => {
                              const target = event.currentTarget;
                              if (event.key === "Enter" && !event.shiftKey) {
                                event.preventDefault();
                                splitSegment(segment.id, target.value.slice(0, target.selectionStart), target.value.slice(target.selectionStart));
                              }
                              if (event.key === "Backspace" && target.selectionStart === 0 && target.selectionEnd === 0) {
                                event.preventDefault();
                                mergePrevious(segment.id);
                              }
                            }}
                          />
                          <span className="char-count">{segment.text.length}</span>
                        </div>
                        <div>
                          <select
                            className="inline-select"
                            value={segment.speed}
                            aria-label={`${t.speed} ${segment.id}`}
                            onChange={(event) => updateSegment(segment.id, { speed: event.target.value as Speed, status: "pending", progress: 0 })}
                          >
                            <option value="slow">{t.slow}</option>
                            <option value="natural">{t.natural}</option>
                            <option value="fast">{t.fast}</option>
                          </select>
                        </div>
                        <div className="prompt-cell" title={segment.prompt}>{segment.prompt}</div>
                        <div className="pause-cell">
                          <input
                            type="number"
                            value={segment.pauseMs}
                            aria-label={`${t.pause} ${segment.id}`}
                            onChange={(event) => updateSegment(segment.id, { pauseMs: Number(event.target.value) })}
                          />
                          <span>{t.ms}</span>
                        </div>
                        <div className="audio-cell">
                          {segment.status === "done" ? (
                            <div className="audio-ready">
                              <button
                                className={`play-button${isPlaying ? " playing" : ""}`}
                                type="button"
                                aria-label={t.play}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setPlayingId(isPlaying ? null : segment.id);
                                }}
                              >
                                {isPlaying ? <Pause size={13} fill="currentColor" /> : <Play size={13} fill="currentColor" />}
                              </button>
                              <WaveformBars active={isPlaying} />
                              <span className="duration">{segment.duration}</span>
                              <IconButton label={t.download}><Download size={14} /></IconButton>
                              <IconButton label={t.regenerate} onClick={() => simulateGeneration(segment.id)}><RefreshCw size={14} /></IconButton>
                            </div>
                          ) : segment.status === "generating" ? (
                            <div className="generating-state">
                              <div className="generation-meta"><span>{t.processing}</span><strong>{segment.progress}%</strong></div>
                              <div className="progress-track"><span style={{ width: `${segment.progress}%` }} /></div>
                            </div>
                          ) : (
                            <button className="generate-row" type="button" onClick={() => simulateGeneration(segment.id)}>
                              <AudioLines size={15} />{t.idle}
                            </button>
                          )}
                          <span className={`status-dot ${segment.status}`} title={statusCopy[segment.status]} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <button className="add-row" type="button" onClick={addSegment}><Plus size={15} />{t.addSegment}</button>
              </div>
            </section>

            <aside className="inspector">
              <div className="inspector-header">
                <div>
                  <span className="eyebrow">{t.selected}</span>
                  <h2>{t.inspector}</h2>
                </div>
                <IconButton label={t.more}><MoreHorizontal size={17} /></IconButton>
              </div>

              {selected && (
                <div className="inspector-content">
                  <div className="segment-identity">
                    <span>{String(selected.id).padStart(2, "0")}</span>
                    <div>
                      <strong>{language === "zh" ? "旁白分段" : "Narration segment"}</strong>
                      <small>{selected.text.length} {language === "zh" ? "字" : "chars"}</small>
                    </div>
                    <span className={`status-badge ${selected.status}`}>{statusCopy[selected.status]}</span>
                  </div>

                  <label className="field">
                    <span>{t.text}</span>
                    <textarea value={selected.text} onChange={(event) => updateSegment(selected.id, { text: event.target.value, status: "pending", progress: 0 })} />
                  </label>

                  <div className="field">
                    <span>{t.speed}</span>
                    <div className="segmented-control">
                      {(["slow", "natural", "fast"] as Speed[]).map((speed) => (
                        <button
                          type="button"
                          key={speed}
                          className={selected.speed === speed ? "active" : ""}
                          onClick={() => updateSegment(selected.id, { speed, status: "pending", progress: 0 })}
                        >
                          {t[speed]}
                        </button>
                      ))}
                    </div>
                  </div>

                  <label className="field">
                    <span>{t.prompt}</span>
                    <textarea className="prompt-editor" value={selected.prompt} onChange={(event) => updateSegment(selected.id, { prompt: event.target.value, status: "pending", progress: 0 })} />
                    <small>{language === "zh" ? "描述语气、重音与情绪变化，不会作为正文朗读。" : "Describe tone, emphasis, and emotional movement. This text is never spoken."}</small>
                  </label>

                  <label className="field">
                    <span>{t.pause}</span>
                    <div className="number-field">
                      <input type="number" value={selected.pauseMs} onChange={(event) => updateSegment(selected.id, { pauseMs: Number(event.target.value) })} />
                      <span>{t.ms}</span>
                    </div>
                  </label>

                  <div className="field">
                    <span>{t.voice}</span>
                    <button className="voice-picker" type="button">
                      <span className="voice-avatar"><AudioWaveform size={16} /></span>
                      <span><strong>{t.voiceName}</strong><small>24 kHz · 00:18</small></span>
                      <ChevronDown size={15} />
                    </button>
                  </div>

                  <div className="inspector-actions">
                    <button className="button primary wide" type="button" onClick={() => simulateGeneration(selected.id)}>
                      <RefreshCw size={15} />{t.regenerate}
                    </button>
                    <IconButton label={t.deleteSegment} onClick={() => deleteSegment(selected.id)}><Trash2 size={16} /></IconButton>
                  </div>
                </div>
              )}
            </aside>
          </div>
        </main>

        <footer className="render-bar">
          <div className="render-summary">
            <div className="render-icon"><Gauge size={18} /></div>
            <div>
              <strong>{t.overall}</strong>
              <span>{completeCount} / {segments.length} {language === "zh" ? "已完成" : "complete"} · {t.totalDuration}</span>
            </div>
          </div>
          <div className="overall-progress">
            <div className="progress-track"><span style={{ width: `${overallProgress}%` }} /></div>
            <strong>{overallProgress}%</strong>
          </div>
          <div className="render-actions">
            <button className="button secondary" type="button" onClick={generateAll}>
              {isBatchGenerating ? <Square size={14} /> : <Sparkles size={15} />}
              {isBatchGenerating ? t.stop : t.generateAll}
            </button>
            <button className="button primary" type="button"><Download size={15} />{t.mergeExport}</button>
          </div>
        </footer>
      </div>
    </Tooltip.Provider>
  );
}

export default App;
