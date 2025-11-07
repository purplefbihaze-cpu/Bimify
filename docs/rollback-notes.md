## Rollback Notes (2025-11-05)

- Restore `ui/package.json` to the previous `web-ifc` version (for example `^0.0.47`) and rerun `npm --prefix ui install`.
- Falls der reine xBIM-Viewer nicht benötigt wird, können Änderungen an `ui/pages/index.tsx` rückgängig gemacht werden.
- Reset `core/ifc/preprocess.py` to the earlier subprocess implementation if you experience issues on Windows.
- Republish the xBIM preprocessor (`tools/publish-xbim.ps1 -Runtime win-x64`) if you roll back executable artifacts.
- WexBIM output remains disabled; no additional action is required when reverting.

