"""Minimal PySide6 desktop UI for Phase 1 player identification.

Flow:
    1. Pick group photo.
    2. Add one or more parent folders of per-player photo folders.
    3. Build database (background thread).
    4. Identify -> results table + group photo with numbered boxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from face_engine import EngineConfig
from face_engine.matcher import MatchResult
from face_engine.rows import format_roster

from .workers import BuildWorker, MatchWorker, ModelLoader

DB_FILENAME = "players.db"

# A clean, light visual theme applied app-wide (used with the Fusion style).
STYLESHEET = """
QWidget {
    background: #f4f5f7;
    color: #1f2329;
    font-size: 13px;
}
QLabel#heading { font-size: 16px; font-weight: 700; color: #111418; }
QLabel#hint { color: #6b7280; }
QGroupBox {
    background: #ffffff;
    border: 1px solid #e3e6ea;
    border-radius: 10px;
    margin-top: 16px;
    padding: 12px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #2d7ff9;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: #eef4ff; border-color: #2d7ff9; }
QPushButton:disabled { color: #aab1bd; background: #f3f4f6; border-color: #e5e7eb; }
QPushButton#primary {
    background: #2d7ff9;
    border: 1px solid #2d7ff9;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover { background: #1c6fe0; border-color: #1c6fe0; }
QPushButton#primary:disabled { background: #9bbcf0; border-color: #9bbcf0; color: #eef2ff; }
QPushButton#ghost {
    background: transparent;
    border: none;
    color: #6b7280;
    padding: 4px 2px;
    text-align: left;
}
QPushButton#ghost:hover { color: #2d7ff9; background: transparent; }
QListWidget, QPlainTextEdit, QComboBox, QTableWidget {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 4px;
}
QComboBox { padding: 6px 8px; }
QTableWidget { gridline-color: #eef0f3; }
QHeaderView::section {
    background: #f0f2f5;
    border: none;
    border-bottom: 1px solid #e3e6ea;
    padding: 6px;
    font-weight: 600;
    color: #374151;
}
QLabel#preview {
    border: 1px dashed #cdd3da;
    border-radius: 10px;
    background: #fafbfc;
    color: #9aa1ab;
}
QStatusBar { background: #ffffff; border-top: 1px solid #e3e6ea; color: #374151; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Team Photo Player Identification — Phase 1")
        self.resize(1100, 720)

        self._config = EngineConfig(
            db_path=Path(__file__).resolve().parent.parent / DB_FILENAME
        )

        # Make the active storage backend unmistakable (shared cloud vs local file).
        if self._config.use_neon:
            backend_label = "Shared cloud database (Neon)"
        else:
            backend_label = f"Local database ({DB_FILENAME})"
        print(f"[storage] Using: {backend_label}")
        self.setWindowTitle(
            f"Team Photo Player Identification — Phase 1  •  {backend_label}"
        )

        self._detector = None
        self._group_photos: list[str] = []
        self._results: list[MatchResult] = []
        # photo path -> its match results, for every photo in the last batch
        self._batch: dict[str, list[MatchResult]] = {}

        self._build_ui()
        self._set_busy("Loading recognition model...")
        self._loader = ModelLoader(self._config)
        self._loader.loaded.connect(self._on_model_loaded)
        self._loader.failed.connect(self._on_error)
        self._loader.start()

    # --- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---- Left: workflow panel ------------------------------------
        left_panel = QWidget()
        left_panel.setMaximumWidth(380)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)

        title = QLabel("Workflow")
        title.setObjectName("heading")
        left.addWidget(title)

        # Step 1 — group photos
        box1 = QGroupBox("1 · Group photos")
        v1 = QVBoxLayout(box1)
        v1.setSpacing(8)
        self.btn_group = QPushButton("Select group photo(s)…")
        self.btn_group.clicked.connect(self._pick_group_photo)
        self.group_label = QLabel("No group photo selected")
        self.group_label.setObjectName("hint")
        self.group_label.setWordWrap(True)
        self.group_list = QListWidget()
        self.group_list.setMaximumHeight(80)
        for w in (self.btn_group, self.group_label, self.group_list):
            v1.addWidget(w)
        left.addWidget(box1)

        # Step 2 — player data (two sources)
        box2 = QGroupBox("2 · Player data")
        v2 = QVBoxLayout(box2)
        v2.setSpacing(6)
        cap_a = QLabel("Name / roster folders")
        cap_a.setObjectName("hint")
        self.btn_add_folder = QPushButton("Add name/roster folder…")
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(90)
        self.btn_clear_folders = QPushButton("Clear roster folders")
        self.btn_clear_folders.setObjectName("ghost")
        self.btn_clear_folders.clicked.connect(self.folder_list.clear)
        cap_b = QLabel("Individual images (id in filename)")
        cap_b.setObjectName("hint")
        self.btn_add_images = QPushButton("Add individual-image folder…")
        self.btn_add_images.clicked.connect(self._add_image_folder)
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(80)
        self.btn_clear_images = QPushButton("Clear image folders")
        self.btn_clear_images.setObjectName("ghost")
        self.btn_clear_images.clicked.connect(self.image_list.clear)
        for w in (
            cap_a,
            self.btn_add_folder,
            self.folder_list,
            self.btn_clear_folders,
            cap_b,
            self.btn_add_images,
            self.image_list,
            self.btn_clear_images,
        ):
            v2.addWidget(w)
        left.addWidget(box2)

        # Step 3 — run
        box3 = QGroupBox("3 · Run")
        v3 = QVBoxLayout(box3)
        v3.setSpacing(8)
        self.btn_build = QPushButton("Build database")
        self.btn_build.setObjectName("primary")
        self.btn_build.clicked.connect(self._build_database)
        self.btn_identify = QPushButton("Identify players")
        self.btn_identify.setObjectName("primary")
        self.btn_identify.clicked.connect(self._identify)
        v3.addWidget(self.btn_build)
        v3.addWidget(self.btn_identify)
        left.addWidget(box3)
        left.addStretch(1)

        # ---- Right: results ------------------------------------------
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)

        sel_row = QHBoxLayout()
        self.result_selector_label = QLabel("Showing results for:")
        self.result_combo = QComboBox()
        self.result_combo.currentIndexChanged.connect(self._on_result_selected)
        sel_row.addWidget(self.result_selector_label)
        sel_row.addWidget(self.result_combo, 1)
        right.addLayout(sel_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "ID", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        right.addWidget(self.table, 1)

        self.image_label = QLabel("Group photo preview")
        self.image_label.setObjectName("preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(320)
        right.addWidget(self.image_label, 2)

        roster_box = QGroupBox("Roster output")
        rv = QVBoxLayout(roster_box)
        rv.setSpacing(8)
        self.roster_text = QPlainTextEdit()
        self.roster_text.setReadOnly(True)
        self.roster_text.setPlaceholderText(
            "Run 'Identify players' to generate the roster…"
        )
        self.roster_text.setMaximumHeight(120)
        self.btn_copy_roster = QPushButton("Copy roster")
        self.btn_copy_roster.clicked.connect(self._copy_roster)
        rv.addWidget(self.roster_text)
        rv.addWidget(self.btn_copy_roster)
        right.addWidget(roster_box)

        root.addWidget(left_panel)
        root.addWidget(right_panel, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self._enable_actions(False)

    # --- actions --------------------------------------------------------

    def _pick_group_photo(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select one or more group photos", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if paths:
            self._group_photos = paths
            self.group_list.clear()
            self.group_list.addItems(Path(p).name for p in paths)
            n = len(paths)
            self.group_label.setText(
                f"{n} group photo{'s' if n != 1 else ''} selected"
            )
            self._show_image(paths[0], [])

    def _add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select parent folder containing named player folders"
        )
        if path and not self._listed(self.folder_list, path):
            self.folder_list.addItem(path)

    def _add_image_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select folder of individual images (id in filename)"
        )
        if path and not self._listed(self.image_list, path):
            self.image_list.addItem(path)

    @staticmethod
    def _listed(widget: QListWidget, path: str) -> bool:
        return any(
            widget.item(i).text() == path for i in range(widget.count())
        )

    def _build_database(self) -> None:
        roster_folders = [
            self.folder_list.item(i).text() for i in range(self.folder_list.count())
        ]
        image_folders = [
            self.image_list.item(i).text() for i in range(self.image_list.count())
        ]
        if not roster_folders:
            QMessageBox.warning(
                self, "No folders", "Add at least one name/roster folder."
            )
            return
        self._set_busy("Building player database...")
        self._build = BuildWorker(
            roster_folders, self._config, self._detector, image_folders=image_folders
        )
        self._build.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._build.finished_ok.connect(self._on_build_done)
        self._build.failed.connect(self._on_error)
        self._build.start()

    def _identify(self) -> None:
        if not self._group_photos:
            QMessageBox.warning(self, "No photo", "Select a group photo first.")
            return
        self._set_busy("Identifying players...")
        self._match = MatchWorker(
            self._group_photos, self._config, self._detector
        )
        self._match.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._match.finished_ok.connect(self._on_match_done)
        self._match.failed.connect(self._on_error)
        self._match.start()

    # --- worker callbacks ----------------------------------------------

    def _on_model_loaded(self, detector) -> None:
        self._detector = detector
        self._enable_actions(True)
        self._set_idle("Model loaded. Ready.")

    def _on_build_done(self, report) -> None:
        self._enable_actions(True)
        self._set_idle("Database built.")
        QMessageBox.information(self, "Database built", report.summary())

    def _on_match_done(
        self, pairs: list[tuple[str, list[MatchResult]]]
    ) -> None:
        self._enable_actions(True)
        self._batch = {photo: results for photo, results in pairs}

        # Save a roster .txt next to each photo.
        saved = 0
        failed: list[str] = []
        for photo, results in pairs:
            roster = format_roster(results, gap_factor=self._config.row_gap_factor)
            if self._save_roster_file(photo, roster):
                saved += 1
            else:
                failed.append(Path(photo).name)

        # Populate the per-photo selector; selecting one shows its results.
        # Store the full path as item data so duplicate filenames never collide.
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        for photo, _ in pairs:
            self.result_combo.addItem(Path(photo).name, photo)
        self.result_combo.blockSignals(False)
        if pairs:
            self.result_combo.setCurrentIndex(0)
            self._on_result_selected(0)

        total = len(pairs)
        status = f"Processed {total} photo{'s' if total != 1 else ''}, saved {saved} roster file(s)."
        if failed:
            status += f" Could not save: {', '.join(failed)}."
        self._set_idle(status)

    def _on_result_selected(self, index: int) -> None:
        """Show the table, roster and boxed preview for the chosen photo."""
        if index < 0 or index >= self.result_combo.count():
            return
        photo = self.result_combo.itemData(index)
        if not photo:
            return
        results = self._batch.get(photo, [])
        self._results = results
        self._populate_table(results)
        roster = format_roster(results, gap_factor=self._config.row_gap_factor)
        self.roster_text.setPlainText(roster)
        self._show_image(photo, results)

    def _save_roster_file(self, photo: str, roster: str) -> Path | None:
        """Write the roster next to its group photo as <photo name>.txt.

        e.g. GirlsVB-FLSC+AW1223.jpg -> GirlsVB-FLSC+AW1223.txt in the same
        folder. Returns the path written, or None if it couldn't be saved.
        """
        out_path = Path(photo).with_suffix(".txt")
        try:
            out_path.write_text(roster + "\n", encoding="utf-8")
            return out_path
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Could not save roster",
                f"The roster for {Path(photo).name} was generated but could not "
                f"be saved next to the photo:\n{out_path}\n\n{exc}\n\n"
                f"You can still use 'Copy roster'.",
            )
            return None

    def _copy_roster(self) -> None:
        QApplication.clipboard().setText(self.roster_text.toPlainText())
        self.statusBar().showMessage("Roster copied to clipboard.")

    def _on_error(self, message: str) -> None:
        self._enable_actions(True)
        self._set_idle("Error.")
        QMessageBox.critical(self, "Error", message)

    # --- rendering ------------------------------------------------------

    def _populate_table(self, results: list[MatchResult]) -> None:
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            self.table.setItem(row, 0, QTableWidgetItem(r.name))
            self.table.setItem(row, 1, QTableWidgetItem(r.player_id or "-"))
            self.table.setItem(
                row, 2, QTableWidgetItem(f"{r.confidence * 100:.1f}%")
            )

    def _show_image(self, path: str, results: list[MatchResult]) -> None:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        # Scale boxes/text to the image so labels stay legible after downscaling.
        h, w = img.shape[:2]
        scale = max(w, h) / 1600.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.7, scale)
        thickness = max(2, round(scale * 2))
        box_th = max(2, round(scale * 2))
        pad = max(4, round(scale * 6))

        for idx, r in enumerate(results, start=1):
            x1, y1, x2, y2 = r.bbox
            color = (0, 170, 0) if r.identified else (0, 0, 220)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_th)
            label = f"{idx}. {r.name}" if r.identified else f"{idx}. Unidentified"
            (tw, th), base = cv2.getTextSize(label, font, font_scale, thickness)

            # Prefer a label bar above the box; drop below if there's no room.
            baseline_y = y1 - pad
            if baseline_y - th - base < 0:
                baseline_y = y2 + th + base + pad
            top_y = baseline_y - th - base
            left_x = max(0, min(x1, w - tw - 2 * pad))
            cv2.rectangle(
                img, (left_x, top_y),
                (left_x + tw + 2 * pad, baseline_y + base), color, -1,
            )
            cv2.putText(
                img, label, (left_x + pad, baseline_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    # --- ui state helpers ----------------------------------------------

    def _enable_actions(self, enabled: bool) -> None:
        for btn in (
            self.btn_group,
            self.btn_add_folder,
            self.btn_clear_folders,
            self.btn_add_images,
            self.btn_clear_images,
            self.btn_build,
            self.btn_identify,
        ):
            btn.setEnabled(enabled)

    def _set_busy(self, message: str) -> None:
        self._enable_actions(False)
        self.statusBar().showMessage(message)

    def _set_idle(self, message: str) -> None:
        self.statusBar().showMessage(message)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
