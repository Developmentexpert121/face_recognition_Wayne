"""School Photo Auto-Sorter — desktop GUI.

Pick the folder(s) of unsorted photos, pick the Proof folder (with sorted/culled
inside), then Preview (dry run, moves nothing) and Sort. Mirrors the look of the
team-photo tool but ships as its own standalone program.
"""

from __future__ import annotations

import sys
from pathlib import Path

# IMPORTANT: import onnxruntime BEFORE PySide6/Qt.
# PySide6 bundles its own onnxruntime DLL; if Qt loads first its DLL wins the
# Windows search path and conflicts with onnxruntime-directml, producing the
# "Unable to import dependency onnxruntime" error at model-load time.
# Importing onnxruntime here registers the correct DLLs before Qt touches them.
try:
    import onnxruntime as _ort  # noqa: F401 – side-effect import only
except Exception:  # noqa: BLE001 – if ort is genuinely missing, FaceDetector will surface it
    pass

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from face_engine import EngineConfig, FaceDetector
from sorter import QRReader

from .workers import SortWorker

# Portraits have large faces, so a 640 detector is fast and accurate enough.
DET_SIZE = 640

STYLESHEET = """
QWidget { font-size: 13px; color: #1f2430; }
QMainWindow, QWidget#central { background: #f4f5f7; }
QGroupBox {
    background: #ffffff; border: 1px solid #e3e6ea; border-radius: 10px;
    margin-top: 14px; padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #6b7280;
    font-weight: 600;
}
QPushButton {
    background: #ffffff; border: 1px solid #d4d8de; border-radius: 8px;
    padding: 7px 12px;
}
QPushButton:hover { border-color: #2d7ff9; }
QPushButton#primary {
    background: #2d7ff9; border: none; color: #ffffff; font-weight: 600;
}
QPushButton#primary:hover { background: #1f6fe5; }
QPushButton#primary:disabled { background: #b8c9e8; }
QPushButton#ghost { border: none; color: #6b7280; padding: 4px 6px; }
QListWidget, QTextEdit {
    background: #ffffff; border: 1px solid #e3e6ea; border-radius: 8px; padding: 4px;
}
QLabel { color: #1f2430; background: transparent; }
QLabel#preview { color: #6b7280; }
QCheckBox { color: #1f2430; spacing: 6px; background: transparent; }
QCheckBox::indicator { width: 16px; height: 16px; }
QDoubleSpinBox, QSpinBox {
    background: #ffffff; color: #1f2430;
    border: 1px solid #d4d8de; border-radius: 6px; padding: 3px 6px;
}
QDoubleSpinBox:disabled, QSpinBox:disabled { color: #9aa1ab; }
QStatusBar { background: #eef0f3; color: #374151; }
"""


class SorterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("School Photo Auto-Sorter")
        self.resize(1040, 700)

        self._settings = QSettings()  # remembers last-used folders across runs
        self._use_gpu = self._settings.value("use_gpu", False, type=bool)
        self._config = self._build_config()
        self._detector = None
        self._qr_reader = QRReader()  # instant, no model download
        self._ocr_reader = None
        self._sources: list[str] = []
        self._proof: str | None = None
        self._fullsize: str | None = None
        self._last_report = None

        self._build_ui()
        self._ocr_started = False
        self._set_busy("Loading recognition model...")
        # Load the models on the MAIN thread (some GPU/DirectML setups fail when
        # a model is created in a background thread — the CLI works because it
        # loads on the main thread). The short timer lets the window paint first;
        # the load then blocks briefly and the buttons enable.
        QTimer.singleShot(50, self._load_models_main_thread)

    # --- UI -------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left: workflow
        left = QVBoxLayout()
        left.setSpacing(12)
        heading = QLabel("Workflow")
        heading.setStyleSheet("font-size: 16px; font-weight: 700;")
        left.addWidget(heading)

        # 1 - photos to sort
        g1 = QGroupBox("1 · Photos to sort")
        v1 = QVBoxLayout(g1)
        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(110)
        btn_add = QPushButton("Add photos folder…")
        btn_add.clicked.connect(self._add_source)
        btn_clear = QPushButton("Clear")
        btn_clear.setObjectName("ghost")
        btn_clear.clicked.connect(self._clear_sources)
        row1 = QHBoxLayout()
        row1.addWidget(btn_add)
        row1.addWidget(btn_clear)
        row1.addStretch()
        v1.addLayout(row1)
        v1.addWidget(self.source_list)
        left.addWidget(g1)

        # 2 - proof folder
        g2 = QGroupBox("2 · Proof folder (contains 'sorted' and 'culled')")
        v2 = QVBoxLayout(g2)
        self.proof_label = QLabel("No Proof folder selected")
        self.proof_label.setObjectName("preview")
        self.proof_label.setWordWrap(True)
        btn_proof = QPushButton("Pick Proof folder…")
        btn_proof.clicked.connect(self._pick_proof)
        v2.addWidget(btn_proof)
        v2.addWidget(self.proof_label)
        left.addWidget(g2)

        # 2b - optional full-size folder (used only to re-check the few misses)
        g2b = QGroupBox("Full-size photos (optional — improves accuracy)")
        v2b = QVBoxLayout(g2b)
        self.fullsize_label = QLabel("Not set — proofs only")
        self.fullsize_label.setObjectName("preview")
        self.fullsize_label.setWordWrap(True)
        btn_fullsize = QPushButton("Pick full-size folder…")
        btn_fullsize.clicked.connect(self._pick_fullsize)
        btn_fullsize_clear = QPushButton("Clear")
        btn_fullsize_clear.setObjectName("ghost")
        btn_fullsize_clear.clicked.connect(self._clear_fullsize)
        rowfs = QHBoxLayout()
        rowfs.addWidget(btn_fullsize)
        rowfs.addWidget(btn_fullsize_clear)
        rowfs.addStretch()
        v2b.addLayout(rowfs)
        v2b.addWidget(self.fullsize_label)
        left.addWidget(g2b)

        # 3 - options
        g3 = QGroupBox("3 · Matching (defaults are safe)")
        form = QFormLayout(g3)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.50)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 0.50)
        self.margin_spin.setSingleStep(0.01)
        self.margin_spin.setValue(0.05)
        form.addRow("Match strictness:", self.threshold_spin)
        form.addRow("Tie-break margin:", self.margin_spin)
        # Both tag readers are ALWAYS on and cannot be switched off. They are not
        # alternatives - they are two passes that cover different failures, and
        # turning either off loses students silently. Measured on a real shoot:
        # QR alone found 29 of 43 students; the OCR pass recovered the other 14.
        # They are shown as ticked-and-locked so the operator can see both are
        # active without being able to break the run.
        self.qr_check = QCheckBox("Step 1 — read QR codes (all photos, fast)")
        self.qr_check.setChecked(True)
        self.qr_check.setEnabled(False)
        self.qr_check.setToolTip(
            "Always on. Scans every photo for a QR name tag — about a minute for "
            "a full shoot."
        )
        form.addRow(self.qr_check)
        self.gpu_check = QCheckBox("Use GPU if available (faster; needs NVIDIA setup)")
        self.gpu_check.setChecked(self._use_gpu)
        self.gpu_check.toggled.connect(self._on_gpu_toggled)
        form.addRow(self.gpu_check)
        self.ocr_check = QCheckBox("Step 2 — read printed IDs (leftovers only)")
        self.ocr_check.setChecked(True)
        self.ocr_check.setEnabled(False)  # always on; never user-toggleable
        self.ocr_check.setToolTip(
            "Always on. Reads the printed number next to the QR, but only on the "
            "photos still unidentified after step 1 — so it costs minutes, not "
            "hours. This is what catches tags whose QR is bent or blurred."
        )
        form.addRow(self.ocr_check)
        left.addWidget(g3)

        # 4 - run
        g4 = QGroupBox("4 · Run")
        v4 = QVBoxLayout(g4)
        self.btn_preview = QPushButton("Preview (dry run — moves nothing)")
        self.btn_preview.clicked.connect(lambda: self._run(dry_run=True))
        self.btn_sort = QPushButton("Sort photos")
        self.btn_sort.setObjectName("primary")
        self.btn_sort.clicked.connect(lambda: self._run(dry_run=False))
        v4.addWidget(self.btn_preview)
        v4.addWidget(self.btn_sort)
        left.addWidget(g4)
        left.addStretch()

        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMaximumWidth(400)
        root.addWidget(left_panel)

        # Right: results
        right = QVBoxLayout()
        right.setSpacing(8)
        results_label = QLabel("Results")
        results_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        right.addWidget(results_label)
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setPlaceholderText(
            "Run a Preview first to see exactly what will move — nothing is "
            "touched until you click “Sort photos”."
        )
        right.addWidget(self.report_view)
        self.btn_save_audit = QPushButton("Save audit CSV…")
        self.btn_save_audit.setObjectName("ghost")
        self.btn_save_audit.setEnabled(False)
        self.btn_save_audit.clicked.connect(self._save_audit)
        right.addWidget(self.btn_save_audit, alignment=Qt.AlignRight)
        right_panel = QWidget()
        right_panel.setLayout(right)
        root.addWidget(right_panel, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Starting…")

    # --- pickers --------------------------------------------------------

    def _add_source(self) -> None:
        start = self._settings.value("last_photos_dir", "", type=str)
        folder = QFileDialog.getExistingDirectory(
            self, "Select a folder of photos", start
        )
        if folder:
            # remember where they were browsing (the parent) for next time
            self._settings.setValue("last_photos_dir", str(Path(folder).parent))
            if folder not in self._sources:
                self._sources.append(folder)
                self.source_list.addItem(folder)

    def _clear_sources(self) -> None:
        self._sources.clear()
        self.source_list.clear()

    def _pick_proof(self) -> None:
        start = self._settings.value("last_proof_dir", "", type=str)
        folder = QFileDialog.getExistingDirectory(self, "Select the Proof folder", start)
        if folder:
            self._settings.setValue("last_proof_dir", str(Path(folder).parent))
            self._proof = folder
            self.proof_label.setText(folder)

    def _pick_fullsize(self) -> None:
        start = self._settings.value("last_fullsize_dir", "", type=str)
        folder = QFileDialog.getExistingDirectory(
            self, "Select the full-size photos folder", start
        )
        if folder:
            self._settings.setValue("last_fullsize_dir", str(Path(folder).parent))
            self._fullsize = folder
            self.fullsize_label.setText(folder)

    def _clear_fullsize(self) -> None:
        self._fullsize = None
        self.fullsize_label.setText("Not set — proofs only")

    # --- run ------------------------------------------------------------

    def _run(self, dry_run: bool) -> None:
        if self._detector is None:
            self.statusBar().showMessage("Model still loading — please wait…")
            return
        if not self._sources:
            self.statusBar().showMessage("Add at least one photos folder first.")
            return
        if not self._proof:
            self.statusBar().showMessage("Pick the Proof folder first.")
            return

        if not dry_run:
            confirm = QMessageBox.question(
                self,
                "Move photos?",
                "This will MOVE photos into the student folders and promote any "
                "culled folders to sorted.\n\nRun a Preview first if you haven't. "
                "Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return

        self._set_busy("Sorting… this can take a while for large folders.")
        self.report_view.clear()
        # Both readers always go in. The only reason either is absent is that it
        # failed to load, never an operator choice.
        self._worker = SortWorker(
            self._sources,
            self._proof,
            self._config,
            self._detector,
            self.threshold_spin.value(),
            self.margin_spin.value(),
            dry_run,
            qr_reader=self._qr_reader,
            ocr_reader=self._ocr_reader,
            fullsize_folders=[self._fullsize] if self._fullsize else None,
        )
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished_ok.connect(self._on_sorted)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    # --- callbacks ------------------------------------------------------

    def _build_config(self) -> EngineConfig:
        return EngineConfig(
            det_size=(DET_SIZE, DET_SIZE), ctx_id=0 if self._use_gpu else -1
        )

    def _load_models_main_thread(self) -> None:
        """Load the face model (and then OCR) on the main thread."""
        try:
            detector = FaceDetector(self._config)
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))
            return
        self._on_model_loaded(detector)

    def _on_gpu_toggled(self, checked: bool) -> None:
        self._use_gpu = checked
        self._settings.setValue("use_gpu", checked)
        self._config = self._build_config()
        self._detector = None
        self._set_busy("Reloading model with new device setting...")
        QTimer.singleShot(50, self._load_models_main_thread)

    def _on_model_loaded(self, detector) -> None:
        self._detector = detector
        prov = getattr(detector, "active_provider", "")
        device = "GPU" if ("CUDA" in prov or "Dml" in prov) else "CPU"
        self._set_idle(
            f"Model ready — running on {device}. "
            "Add photos and a Proof folder, then Preview."
        )
        # Load OCR now, on the main thread too (after the model, never together).
        if not self._ocr_started:
            self._ocr_started = True
            try:
                from sorter import TagReader

                self._on_ocr_loaded(TagReader())
            except Exception as exc:  # noqa: BLE001 - OCR optional; QR still works
                self._on_ocr_failed(str(exc))

    def _on_ocr_loaded(self, reader) -> None:
        self._ocr_reader = reader
        # Deliberately NOT re-enabled: step 2 is part of the sort, not an option.
        self.ocr_check.setChecked(True)

    def _on_ocr_failed(self, message: str) -> None:
        # The only case where step 2 is off. Say so loudly: without it the run
        # silently loses every student whose QR will not scan, which is the
        # failure that is hardest to notice in the results.
        self._ocr_reader = None
        self.ocr_check.setChecked(False)
        self.ocr_check.setEnabled(False)
        self.ocr_check.setText(
            "⚠ Step 2 UNAVAILABLE — printed IDs cannot be read; expect missed students"
        )
        self.ocr_check.setStyleSheet("color: #b3261e; font-weight: 600;")
        self.ocr_check.setToolTip(
            f"The text reader failed to load, so only QR codes will be read.\n\n{message}"
        )

    def _on_sorted(self, report) -> None:
        self.report_view.setPlainText(report.summary())
        self._last_report = report
        self.btn_save_audit.setEnabled(True)
        mode = "Preview" if report.dry_run else "Sort"
        self._set_idle(
            f"{mode} complete — {report.sorted_count} sorted, "
            f"{report.leftover_count} left in source."
        )

    def _save_audit(self) -> None:
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save audit CSV", "audit.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            self._last_report.write_csv(path)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save", str(exc))
            return
        self.statusBar().showMessage(f"Audit saved to {path}")

    def _on_error(self, message: str) -> None:
        self._set_idle("Error — see message.")
        QMessageBox.critical(self, "Error", message)

    # --- busy state -----------------------------------------------------

    def _set_busy(self, message: str) -> None:
        self.btn_preview.setEnabled(False)
        self.btn_sort.setEnabled(False)
        self.statusBar().showMessage(message)

    def _set_idle(self, message: str) -> None:
        self.btn_preview.setEnabled(True)
        self.btn_sort.setEnabled(True)
        self.statusBar().showMessage(message)


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("DevexHub")
    app.setApplicationName("SchoolPhotoSorter")
    app.setStyle("Fusion")
    # Force a light palette so macOS dark mode can't render dark-on-dark controls.
    from PySide6.QtGui import QColor, QPalette

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f4f5f7"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.Text, QColor("#1f2430"))
    pal.setColor(QPalette.WindowText, QColor("#1f2430"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#1f2430"))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)
    window = SorterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
