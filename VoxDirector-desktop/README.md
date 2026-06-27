# VoxDirector Desktop

Offline desktop workspace for directing long-form narration with VoxCPM.

This directory currently contains the first interactive UI prototype. Generation,
playback, project persistence, and export use simulated data until the existing
Python inference core is connected as a Tauri sidecar.

## Development

```bash
npm install
npm run dev
npm run build
npm run tauri dev
```

The prototype includes:

- Editable director table with row-level audio states
- Enter-to-split and start-of-row Backspace-to-merge editing
- Segment inspector for pacing, direction, pause, and reference voice
- Simulated row and batch generation progress
- Light/dark themes and Chinese/English interface switching
- Desktop project, chapter, render, and export shell
