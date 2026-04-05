# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Custom parameter widget replicating the Pix4D Survey DSM "Surface smoothing"
# panel: a Low–High slider paired with a spinbox, labelled
# "Median filter radius (px)".

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qgis.gui import QgsAbstractProcessingParameterWidgetWrapper
from qgis.core import (
    QgsProcessingParameterNumber,
    QgsProcessingContext,
    QgsProcessingParameterDefinition,
)

RADIUS_MIN = 1
RADIUS_MAX = 50
RADIUS_DEFAULT = 6

# ---------------------------------------------------------------------------
# QSS — dark theme matching Pix4D Survey panel style
# ---------------------------------------------------------------------------
_QSS = """
QWidget#SmoothingWidget {
    background-color: #2b2b2b;
    border-radius: 4px;
    padding: 6px;
}
QLabel#headerLabel {
    color: #cccccc;
    font-size: 12px;
}
QLabel#subtitleLabel {
    color: #888888;
    font-size: 10px;
}
QLabel#lowLabel, QLabel#highLabel {
    color: #888888;
    font-size: 10px;
    min-width: 28px;
}
QSlider::groove:horizontal {
    border: 1px solid #555555;
    height: 4px;
    background: #444444;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4a90d9;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4a90d9;
    border-radius: 2px;
}
QSpinBox {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 12px;
    min-width: 44px;
    max-width: 52px;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 0px;
}
"""


class SmoothingWidget(QWidget):
    """
    Standalone widget:

        Surface smoothing
        [ Low ══════●═══════════ High ]  [ 6 ]
               Median filter radius (px)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SmoothingWidget")
        self.setStyleSheet(_QSS)
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # Section header
        header = QLabel("Surface smoothing")
        header.setObjectName("headerLabel")
        bold = QFont()
        bold.setBold(True)
        header.setFont(bold)
        root.addWidget(header)

        # Slider row:  Low  [====slider====]  High  [spinbox]
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)

        low_lbl = QLabel("Low")
        low_lbl.setObjectName("lowLabel")
        low_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(RADIUS_MIN)
        self.slider.setMaximum(RADIUS_MAX)
        self.slider.setValue(RADIUS_DEFAULT)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(5)

        high_lbl = QLabel("High")
        high_lbl.setObjectName("highLabel")
        high_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(RADIUS_MIN)
        self.spinbox.setMaximum(RADIUS_MAX)
        self.spinbox.setValue(RADIUS_DEFAULT)
        self.spinbox.setAlignment(Qt.AlignCenter)

        slider_row.addWidget(low_lbl)
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(high_lbl)
        slider_row.addWidget(self.spinbox)

        root.addLayout(slider_row)

        # Subtitle
        subtitle = QLabel("Median filter radius (px)")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

    def _connect_signals(self) -> None:
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    # ------------------------------------------------------------------
    # Synchronisation
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        if self.spinbox.value() != value:
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(value)
            self.spinbox.blockSignals(False)

    def _on_spinbox_changed(self, value: int) -> None:
        if self.slider.value() != value:
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Value API
    # ------------------------------------------------------------------

    def value(self) -> int:
        return self.spinbox.value()

    def setValue(self, v: int) -> None:
        self.spinbox.setValue(max(RADIUS_MIN, min(RADIUS_MAX, int(v))))


# ---------------------------------------------------------------------------
# QGIS Processing widget wrapper
# ---------------------------------------------------------------------------

class SimpliNeighborsRadiusWrapper(QgsAbstractProcessingParameterWidgetWrapper):
    """
    Wraps SmoothingWidget as a Processing parameter widget for the RADIUS
    parameter, so it renders inside the standard Processing algorithm dialog.
    """

    def __init__(self, parameter: QgsProcessingParameterDefinition,
                 dialog_type, parent: QWidget | None = None) -> None:
        super().__init__(parameter, dialog_type, parent)

    def createWidget(self) -> QWidget:
        self._widget = SmoothingWidget()
        default = self.parameterDefinition().defaultValue()
        if default is not None:
            self._widget.setValue(int(default))
        return self._widget

    def setWidgetValue(self, value, context: QgsProcessingContext) -> None:
        if value is not None:
            self._widget.setValue(int(value))

    def widgetValue(self):
        return self._widget.value()
