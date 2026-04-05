# SimpliNeighbors — Fast DSM Surface Smoothing Plugin for QGIS 4
# Copyright (C) 2024 SimpliFly
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Custom parameter widgets:
#   SmoothingWidget       — radius slider + spinbox  (Pix4D-style)
#   CoresWidget           — CPU cores slider + spinbox
#   ResolutionWidget      — Default / Custom toggle + value input

from __future__ import annotations

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsProcessingContext, QgsProcessingParameterDefinition
from qgis.gui import QgsAbstractProcessingParameterWidgetWrapper

_CPU_COUNT = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Shared QSS — dark theme matching Pix4D Survey panel
# ---------------------------------------------------------------------------
_QSS = """
QWidget {
    background-color: transparent;
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
QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 2px 4px;
    font-size: 12px;
    min-width: 48px;
    max-width: 58px;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
}
QPushButton#modeBtn {
    background-color: #3c3c3c;
    color: #888888;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 11px;
    min-width: 60px;
}
QPushButton#modeBtn:checked {
    background-color: #4a90d9;
    color: #ffffff;
    border: 1px solid #4a90d9;
}
"""


# ---------------------------------------------------------------------------
# Shared slider + spinbox row builder
# ---------------------------------------------------------------------------

def _make_slider_row(
    parent: QWidget,
    minimum: int,
    maximum: int,
    default: int,
    subtitle: str,
    header: str,
) -> tuple[QWidget, QSlider, QSpinBox]:
    """
    Build the standard Pix4D-style slider row and return
    (container_widget, slider, spinbox).
    """
    container = QWidget(parent)
    container.setStyleSheet(_QSS)
    root = QVBoxLayout(container)
    root.setContentsMargins(0, 4, 0, 4)
    root.setSpacing(3)

    hdr = QLabel(header)
    hdr.setObjectName("headerLabel")
    bold = QFont()
    bold.setBold(True)
    hdr.setFont(bold)
    root.addWidget(hdr)

    row = QHBoxLayout()
    row.setSpacing(6)

    low_lbl = QLabel("Low")
    low_lbl.setObjectName("lowLabel")
    low_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(default)
    slider.setTickPosition(QSlider.TicksBelow)
    slider.setTickInterval(max(1, (maximum - minimum) // 10))

    high_lbl = QLabel("High")
    high_lbl.setObjectName("highLabel")
    high_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    spinbox = QSpinBox()
    spinbox.setMinimum(minimum)
    spinbox.setMaximum(maximum)
    spinbox.setValue(default)
    spinbox.setAlignment(Qt.AlignCenter)

    row.addWidget(low_lbl)
    row.addWidget(slider, stretch=1)
    row.addWidget(high_lbl)
    row.addWidget(spinbox)
    root.addLayout(row)

    sub = QLabel(subtitle)
    sub.setObjectName("subtitleLabel")
    sub.setAlignment(Qt.AlignCenter)
    root.addWidget(sub)

    # bidirectional sync
    def _slider_changed(v: int) -> None:
        if spinbox.value() != v:
            spinbox.blockSignals(True)
            spinbox.setValue(v)
            spinbox.blockSignals(False)

    def _spinbox_changed(v: int) -> None:
        if slider.value() != v:
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)

    slider.valueChanged.connect(_slider_changed)
    spinbox.valueChanged.connect(_spinbox_changed)

    return container, slider, spinbox


# ---------------------------------------------------------------------------
# SmoothingWidget — radius slider
# ---------------------------------------------------------------------------

class SmoothingWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._container, self._slider, self._spinbox = _make_slider_row(
            self,
            minimum=1, maximum=50, default=6,
            header="Surface smoothing",
            subtitle="Median filter radius (px)",
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)

    def value(self) -> int:
        return self._spinbox.value()

    def setValue(self, v: int) -> None:
        self._spinbox.setValue(max(1, min(50, int(v))))


# ---------------------------------------------------------------------------
# CoresWidget — CPU cores slider
# ---------------------------------------------------------------------------

class CoresWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._container, self._slider, self._spinbox = _make_slider_row(
            self,
            minimum=1, maximum=_CPU_COUNT, default=_CPU_COUNT,
            header="CPU cores",
            subtitle=f"Parallel threads  (detected: {_CPU_COUNT})",
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)

    def value(self) -> int:
        return self._spinbox.value()

    def setValue(self, v: int) -> None:
        self._spinbox.setValue(max(1, min(_CPU_COUNT, int(v))))


# ---------------------------------------------------------------------------
# ResolutionWidget — Default / Custom toggle
# ---------------------------------------------------------------------------

class ResolutionWidget(QWidget):
    """
    Output resolution  [ Default ]  [ Custom ]    [ 0.0040 ]
                       map units/px
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_QSS)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(3)

        hdr = QLabel("Output resolution")
        hdr.setObjectName("headerLabel")
        bold = QFont()
        bold.setBold(True)
        hdr.setFont(bold)
        root.addWidget(hdr)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._btn_default = QPushButton("Default")
        self._btn_default.setObjectName("modeBtn")
        self._btn_default.setCheckable(True)
        self._btn_default.setChecked(True)

        self._btn_custom = QPushButton("Custom")
        self._btn_custom.setObjectName("modeBtn")
        self._btn_custom.setCheckable(True)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.addButton(self._btn_default)
        self._group.addButton(self._btn_custom)

        self._value_input = QDoubleSpinBox()
        self._value_input.setMinimum(0.0001)
        self._value_input.setMaximum(9999.0)
        self._value_input.setDecimals(4)
        self._value_input.setValue(1.0)
        self._value_input.setAlignment(Qt.AlignCenter)
        self._value_input.setEnabled(False)

        row.addWidget(self._btn_default)
        row.addWidget(self._btn_custom)
        row.addWidget(self._value_input)
        row.addStretch()
        root.addLayout(row)

        sub = QLabel("map units/px  (applies before filtering)")
        sub.setObjectName("subtitleLabel")
        root.addWidget(sub)

        self._btn_default.toggled.connect(self._on_mode_changed)

    def _on_mode_changed(self, default_checked: bool) -> None:
        self._value_input.setEnabled(not default_checked)

    # ------------------------------------------------------------------
    # Value API — stores "default" or float-as-string
    # ------------------------------------------------------------------

    def value(self) -> str:
        if self._btn_default.isChecked():
            return "default"
        return str(self._value_input.value())

    def setValue(self, v: str) -> None:
        v = str(v).strip().lower()
        if v in ("", "default"):
            self._btn_default.setChecked(True)
        else:
            try:
                self._value_input.setValue(float(v))
                self._btn_custom.setChecked(True)
            except ValueError:
                self._btn_default.setChecked(True)


# ---------------------------------------------------------------------------
# Processing wrapper classes
# ---------------------------------------------------------------------------

class SimpliNeighborsRadiusWrapper(QgsAbstractProcessingParameterWidgetWrapper):
    def __init__(self, parameter: QgsProcessingParameterDefinition,
                 dialog_type, parent: QWidget | None = None) -> None:
        super().__init__(parameter, dialog_type, parent)

    def createWidget(self) -> QWidget:
        self._widget = SmoothingWidget()
        d = self.parameterDefinition().defaultValue()
        if d is not None:
            self._widget.setValue(int(d))
        return self._widget

    def setWidgetValue(self, value, context: QgsProcessingContext) -> None:
        if value is not None:
            self._widget.setValue(int(value))

    def widgetValue(self):
        return self._widget.value()


class SimpliNeighborsCoresWrapper(QgsAbstractProcessingParameterWidgetWrapper):
    def __init__(self, parameter: QgsProcessingParameterDefinition,
                 dialog_type, parent: QWidget | None = None) -> None:
        super().__init__(parameter, dialog_type, parent)

    def createWidget(self) -> QWidget:
        self._widget = CoresWidget()
        d = self.parameterDefinition().defaultValue()
        if d is not None:
            self._widget.setValue(int(d))
        return self._widget

    def setWidgetValue(self, value, context: QgsProcessingContext) -> None:
        if value is not None:
            self._widget.setValue(int(value))

    def widgetValue(self):
        return self._widget.value()


class SimpliNeighborsResolutionWrapper(QgsAbstractProcessingParameterWidgetWrapper):
    def __init__(self, parameter: QgsProcessingParameterDefinition,
                 dialog_type, parent: QWidget | None = None) -> None:
        super().__init__(parameter, dialog_type, parent)

    def createWidget(self) -> QWidget:
        self._widget = ResolutionWidget()
        d = self.parameterDefinition().defaultValue()
        if d is not None:
            self._widget.setValue(str(d))
        return self._widget

    def setWidgetValue(self, value, context: QgsProcessingContext) -> None:
        if value is not None:
            self._widget.setValue(str(value))

    def widgetValue(self):
        return self._widget.value()
