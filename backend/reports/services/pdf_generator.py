"""
PDF Report Generator using ReportLab.

Generates professional PDF reports with headers, tables, charts, and metrics.
"""

from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)

# Logo paths - CSR first (leftmost), then Ministry, Coal India, CMPDIL
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "logos"
LOGO_FILES = [
    "csr-logo-removebg-preview.png",
    "ministry-of-coal-removebg-preview.png",
    "coal-india-removebg-preview.png",
    "cmpdil-removebg-preview.png",
]
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart

# Try to import matplotlib for advanced charts
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PDFGenerator:
    """
    Generate PDF reports from templates and data.
    """

    # Color scheme
    COLORS = {
        "primary": colors.HexColor("#fb923c"),  # Orange
        "success": colors.HexColor("#22c55e"),  # Green
        "warning": colors.HexColor("#f59e0b"),  # Amber
        "danger": colors.HexColor("#ef4444"),   # Red
        "info": colors.HexColor("#3b82f6"),     # Blue
        "dark": colors.HexColor("#1f2937"),     # Dark gray
        "light": colors.HexColor("#f3f4f6"),    # Light gray
        "white": colors.white,
        "green": colors.HexColor("#22c55e"),
        "orange": colors.HexColor("#f59e0b"),
        "red": colors.HexColor("#ef4444"),
        "blue": colors.HexColor("#3b82f6"),
        "darkred": colors.HexColor("#b91c1c"),
        "gray": colors.HexColor("#6b7280"),
    }

    def __init__(self):
        """Initialize PDF generator with styles."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.COLORS['dark'],
            spaceAfter=12,
            alignment=TA_CENTER
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=self.COLORS['gray'],
            spaceAfter=20,
            alignment=TA_CENTER
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.COLORS['dark'],
            spaceBefore=20,
            spaceAfter=10,
            borderPadding=5
        ))

        # Metric value style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=self.COLORS['primary'],
            alignment=TA_CENTER
        ))

        # Metric label style
        self.styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=self.COLORS['gray'],
            alignment=TA_CENTER
        ))

    async def generate(
        self,
        template,
        data: Dict[str, Any],
        include_charts: bool = True
    ) -> BytesIO:
        """
        Generate PDF report from template and data.

        Args:
            template: Report template instance
            data: Report data dictionary
            include_charts: Whether to include charts

        Returns:
            BytesIO buffer containing the PDF
        """
        buffer = BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        # Build elements
        elements = []

        # Get structure from template
        structure = template.get_pdf_structure()

        # Process each section
        for section in structure:
            section_elements = self._build_section(section, data, template, include_charts)
            elements.extend(section_elements)

        # Add footer info
        elements.append(Spacer(1, 20))
        footer_text = f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Mine Safety PPE Detection System"
        elements.append(Paragraph(footer_text, self.styles['ReportSubtitle']))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _build_section(
        self,
        section: Dict[str, Any],
        data: Dict[str, Any],
        template,
        include_charts: bool
    ) -> List:
        """Build elements for a section."""
        section_type = section.get("type")
        elements = []

        if section_type == "header":
            elements.extend(self._build_header(section, data, template))
        elif section_type == "metrics_row":
            elements.extend(self._build_metrics_row(section, data))
        elif section_type == "table":
            elements.extend(self._build_table(section, data))
        elif section_type == "chart" and include_charts:
            elements.extend(self._build_chart(section, data))
        elif section_type == "info_box":
            elements.extend(self._build_info_box(section, data))
        elif section_type == "two_column":
            elements.extend(self._build_two_column(section, data, template, include_charts))
        elif section_type == "highlight_box":
            elements.extend(self._build_highlight_box(section, data))
        elif section_type == "comparison_box":
            elements.extend(self._build_comparison_box(section, data))
        elif section_type == "alert_box":
            elements.extend(self._build_alert_box(section, data))
        elif section_type == "regulatory_box":
            elements.extend(self._build_regulatory_box(section, data))
        elif section_type == "worker_profile":
            elements.extend(self._build_worker_profile(section, data))
        elif section_type == "badges_row":
            elements.extend(self._build_badges_row(section, data))
        elif section_type == "score_gauge":
            elements.extend(self._build_score_gauge(section, data))

        return elements

    def _build_logo_header(self) -> List:
        """Build the logo header with coal ministry logos."""
        elements = []

        logo_images = []
        for logo_file in LOGO_FILES:
            logo_path = ASSETS_DIR / logo_file
            if logo_path.exists():
                try:
                    # CSR logo needs to be larger as it's less visible
                    if "csr" in logo_file.lower():
                        img = Image(str(logo_path), width=45*mm, height=22*mm)
                    else:
                        # Other logos at standard size
                        img = Image(str(logo_path), width=35*mm, height=18*mm)
                    img.hAlign = 'CENTER'
                    logo_images.append(img)
                except Exception as e:
                    print(f"Error loading logo {logo_file}: {e}")

        if logo_images:
            # Calculate column widths based on number of logos
            total_width = 170*mm  # Total available width
            col_width = total_width / len(logo_images)

            # Create a table with all logos in a row
            logo_table = Table([logo_images], colWidths=[col_width] * len(logo_images))
            logo_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(logo_table)
            elements.append(Spacer(1, 8))

            # Add separator line
            separator = Table([['']], colWidths=[170*mm])
            separator.setStyle(TableStyle([
                ('LINEBELOW', (0, 0), (-1, -1), 2, self.COLORS['primary']),
            ]))
            elements.append(separator)
            elements.append(Spacer(1, 12))

        return elements

    def _build_header(self, section: Dict, data: Dict, template) -> List:
        """Build report header."""
        elements = []

        # Add logo header first
        elements.extend(self._build_logo_header())

        # Title
        title = section.get("title", template.report_name)
        elements.append(Paragraph(title, self.styles['ReportTitle']))

        # Subtitle with mine name and date range
        subtitle_parts = []
        if data.get("mine_name"):
            subtitle_parts.append(data["mine_name"])
        if data.get("date_range"):
            subtitle_parts.append(data["date_range"])

        if subtitle_parts:
            elements.append(Paragraph(" | ".join(subtitle_parts), self.styles['ReportSubtitle']))

        elements.append(Spacer(1, 10))

        return elements

    def _build_metrics_row(self, section: Dict, data: Dict) -> List:
        """Build a row of metric cards."""
        elements = []
        metrics = section.get("metrics", [])

        if not metrics:
            return elements

        # Build metric cells
        metric_data = []
        for metric in metrics:
            key = metric.get("key")
            label = metric.get("label", key)
            value = self._get_nested_value(data, key)
            color = metric.get("color", "blue")
            fmt = metric.get("format")

            # Format value
            if fmt == "percentage":
                value = f"{value}%" if value is not None else "N/A"
            elif fmt == "score":
                value = f"{value}/100" if value is not None else "N/A"
            elif value is None:
                value = "N/A"

            # Determine color
            if color == "auto" and isinstance(value, (int, float)):
                if "compliance" in key.lower() or "score" in key.lower():
                    color = "green" if float(str(value).replace('%', '')) >= 80 else "orange" if float(str(value).replace('%', '')) >= 60 else "red"

            metric_data.append({
                "value": str(value),
                "label": label,
                "color": self.COLORS.get(color, self.COLORS["blue"])
            })

        # Create table for metrics - use nested tables for proper layout
        table_data = [[]]
        col_width = 170*mm / len(metric_data)

        for m in metric_data:
            # Create a nested table for each metric cell with value on top, label below
            value_style = ParagraphStyle(
                'MetricValueInline',
                parent=self.styles['Normal'],
                fontSize=18,
                alignment=TA_CENTER,
                leading=22,
                spaceAfter=4,
            )
            label_style = ParagraphStyle(
                'MetricLabelInline',
                parent=self.styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=self.COLORS['gray'],
                leading=10,
            )

            # Create nested table with value and label in separate rows
            inner_table = Table(
                [
                    [Paragraph(f"<font color='{m['color'].hexval()}'><b>{m['value']}</b></font>", value_style)],
                    [Paragraph(m['label'], label_style)]
                ],
                colWidths=[col_width - 10],
                rowHeights=[28, 16]
            )
            inner_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (0, 0), 'BOTTOM'),
                ('VALIGN', (0, 1), (0, 1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ]))
            table_data[0].append(inner_table)

        table = Table(table_data, colWidths=[col_width] * len(metric_data))
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, self.COLORS['light']),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 15))

        return elements

    def _build_table(self, section: Dict, data: Dict) -> List:
        """Build a data table."""
        elements = []

        title = section.get("title")
        if title:
            elements.append(Paragraph(title, self.styles['SectionHeader']))

        data_key = section.get("data_key")
        columns = section.get("columns", [])
        max_rows = section.get("max_rows", 50)
        row_color = section.get("row_color")

        table_data_raw = self._get_nested_value(data, data_key)
        if not table_data_raw or not isinstance(table_data_raw, list):
            empty_msg = section.get("empty_message", "No data available")
            elements.append(Paragraph(f"<i>{empty_msg}</i>", self.styles['Normal']))
            elements.append(Spacer(1, 10))
            return elements

        # Build header row
        header_row = [col.get("label", col.get("key")) for col in columns]
        table_rows = [header_row]

        # Build data rows
        for i, row in enumerate(table_data_raw[:max_rows]):
            row_data = []
            for col in columns:
                key = col.get("key")
                fmt = col.get("format")
                value = self._get_nested_value(row, key)

                # Format value
                if fmt == "datetime" and value:
                    if hasattr(value, 'strftime'):
                        value = value.strftime("%Y-%m-%d %H:%M")
                elif fmt == "date" and value:
                    if hasattr(value, 'strftime'):
                        value = value.strftime("%Y-%m-%d")
                elif fmt == "percentage":
                    value = f"{value}%" if value is not None else "-"
                elif fmt == "boolean":
                    value = "Yes" if value else "No"
                elif fmt == "list" and isinstance(value, list):
                    value = ", ".join(str(v.get("type", v) if isinstance(v, dict) else v) for v in value[:3])
                elif fmt == "score":
                    value = f"{value}" if value is not None else "-"
                elif fmt == "risk_badge":
                    value = str(value).upper() if value else "-"
                elif fmt == "severity_badge":
                    value = str(value).upper() if value else "-"
                elif value is None:
                    value = "-"

                row_data.append(str(value)[:50])  # Truncate long values

            table_rows.append(row_data)

        # Calculate column widths - use mm units for consistency
        num_cols = len(columns)
        total_width = 170*mm
        col_widths = [total_width / num_cols] * num_cols

        table = Table(table_rows, colWidths=col_widths, repeatRows=1)

        # Base style
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
        ]

        # Alternating row colors
        for i in range(1, len(table_rows)):
            if i % 2 == 0:
                style_commands.append(('BACKGROUND', (0, i), (-1, i), self.COLORS['light']))

        # Row color override
        if row_color:
            color = self.COLORS.get(row_color, self.COLORS['light'])
            for i in range(1, len(table_rows)):
                style_commands.append(('BACKGROUND', (0, i), (-1, i), color.clone()))
                # Lighten the color for alternating rows
                if i % 2 == 1:
                    style_commands.append(('BACKGROUND', (0, i), (-1, i),
                                          colors.Color(color.red * 0.9 + 0.1,
                                                      color.green * 0.9 + 0.1,
                                                      color.blue * 0.9 + 0.1)))

        table.setStyle(TableStyle(style_commands))
        elements.append(table)
        elements.append(Spacer(1, 15))

        return elements

    def _extract_numeric_value(self, value) -> float:
        """Extract a numeric value from various data types."""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            # Try common keys for numeric values
            for key in ['count', 'value', 'total', 'amount', 'score', 'rate', 'percentage']:
                if key in value and isinstance(value[key], (int, float)):
                    return float(value[key])
            # If no common key found, try to get the first numeric value
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        elif isinstance(value, str):
            try:
                return float(value.replace('%', '').replace(',', ''))
            except:
                return 0.0
        return 0.0

    def _build_chart(self, section: Dict, data: Dict) -> List:
        """Build a chart using ReportLab or matplotlib."""
        elements = []

        title = section.get("title")
        if title:
            elements.append(Paragraph(title, self.styles['SectionHeader']))

        chart_type = section.get("chart_type", "bar")
        data_key = section.get("data_key")
        x_key = section.get("x_key")
        y_key = section.get("y_key")

        chart_data = self._get_nested_value(data, data_key)

        if not chart_data:
            elements.append(Paragraph("<i>No data for chart</i>", self.styles['Normal']))
            return elements

        # Handle dictionary data (for pie charts)
        if isinstance(chart_data, dict):
            labels = list(chart_data.keys())
            raw_values = list(chart_data.values())
        elif isinstance(chart_data, list):
            labels = [str(item.get(x_key, i)) if isinstance(item, dict) else str(i) for i, item in enumerate(chart_data)]
            raw_values = [item.get(y_key, 0) if isinstance(item, dict) else item for item in chart_data]
        else:
            return elements

        # Convert all values to numeric (handles dicts, strings, etc.)
        values = [self._extract_numeric_value(v) for v in raw_values]

        # Filter out zero values and their labels for pie charts
        if chart_type == "pie":
            filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
            if filtered:
                labels, values = zip(*filtered)
                labels, values = list(labels), list(values)

        # Use matplotlib if available for better charts
        if MATPLOTLIB_AVAILABLE:
            elements.extend(self._build_matplotlib_chart(chart_type, labels, values, title))
        else:
            elements.extend(self._build_reportlab_chart(chart_type, labels, values))

        elements.append(Spacer(1, 15))
        return elements

    def _build_matplotlib_chart(
        self,
        chart_type: str,
        labels: List,
        values: List,
        title: str = ""
    ) -> List:
        """Build chart using matplotlib."""
        elements = []

        fig, ax = plt.subplots(figsize=(6, 3))

        if chart_type == "bar":
            colors_list = ['#fb923c'] * len(labels)  # Orange
            bars = ax.bar(range(len(labels)), values, color=colors_list)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

        elif chart_type == "line":
            ax.plot(range(len(labels)), values, marker='o', color='#fb923c', linewidth=2)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.fill_between(range(len(labels)), values, alpha=0.3, color='#fb923c')

        elif chart_type == "pie":
            colors_list = ['#fb923c', '#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors_list[:len(labels)])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save to buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buffer.seek(0)

        # Add to elements
        img = Image(img_buffer, width=400, height=200)
        elements.append(img)

        return elements

    def _build_reportlab_chart(
        self,
        chart_type: str,
        labels: List,
        values: List
    ) -> List:
        """Build chart using ReportLab graphics."""
        elements = []

        drawing = Drawing(400, 200)

        if chart_type == "bar":
            chart = VerticalBarChart()
            chart.x = 50
            chart.y = 50
            chart.height = 125
            chart.width = 300
            chart.data = [values]
            chart.categoryAxis.categoryNames = labels[:10]  # Limit labels
            chart.bars[0].fillColor = self.COLORS['primary']
            drawing.add(chart)

        elif chart_type == "pie":
            chart = Pie()
            chart.x = 100
            chart.y = 25
            chart.width = 150
            chart.height = 150
            chart.data = values
            chart.labels = labels
            chart.slices.strokeWidth = 0.5
            drawing.add(chart)

        elements.append(drawing)
        return elements

    def _build_info_box(self, section: Dict, data: Dict) -> List:
        """Build an info box with key-value pairs."""
        elements = []
        fields = section.get("fields", [])

        info_data = []
        for field in fields:
            value = self._get_nested_value(data, field)
            label = field.split(".")[-1].replace("_", " ").title()
            info_data.append([label, str(value) if value else "-"])

        if info_data:
            table = Table(info_data, colWidths=[50*mm, 120*mm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), self.COLORS['light']),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 10))

        return elements

    def _build_two_column(self, section: Dict, data: Dict, template, include_charts: bool) -> List:
        """Build two-column layout."""
        elements = []

        left_section = section.get("left", {})
        right_section = section.get("right", {})

        left_elements = self._build_section(left_section, data, template, include_charts)
        right_elements = self._build_section(right_section, data, template, include_charts)

        # Create a table for two-column layout
        table_data = [[left_elements, right_elements]]
        table = Table(table_data, colWidths=[85*mm, 85*mm])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))

        elements.append(table)
        return elements

    def _build_highlight_box(self, section: Dict, data: Dict) -> List:
        """Build a highlighted info box."""
        elements = []

        title = section.get("title")
        data_key = section.get("data_key")
        fields = section.get("fields", [])

        box_data = self._get_nested_value(data, data_key)
        if not box_data:
            return elements

        if title:
            elements.append(Paragraph(f"<b>{title}</b>", self.styles['Normal']))

        for field in fields:
            value = box_data.get(field) if isinstance(box_data, dict) else None
            label = field.replace("_", " ").title()
            if value is not None:
                if "rate" in field.lower() or "compliance" in field.lower():
                    value = f"{value}%"
                elements.append(Paragraph(f"{label}: <b>{value}</b>", self.styles['Normal']))

        elements.append(Spacer(1, 10))
        return elements

    def _build_comparison_box(self, section: Dict, data: Dict) -> List:
        """Build a comparison box for period-over-period analysis."""
        elements = []

        title = section.get("title", "Period Comparison")
        current_value = self._get_nested_value(data, section.get("current_key"))
        previous_value = self._get_nested_value(data, section.get("previous_key"))
        change = self._get_nested_value(data, section.get("change_key"))

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        change_color = "green" if change and change < 0 else "red" if change and change > 0 else "gray"
        change_arrow = "↓" if change and change < 0 else "↑" if change and change > 0 else "→"

        table_data = [
            ["Current Period", "Previous Period", "Change"],
            [str(current_value or 0), str(previous_value or 0), f"{change_arrow} {abs(change or 0):.1f}%"]
        ]

        col_width = 170*mm / 3
        table = Table(table_data, colWidths=[col_width, col_width, col_width])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 15))
        return elements

    def _build_alert_box(self, section: Dict, data: Dict) -> List:
        """Build an alert/notification box."""
        elements = []

        title = section.get("title", "Alert")
        value = self._get_nested_value(data, section.get("data_key"))
        severity = section.get("severity", "info")

        color = self.COLORS.get(severity, self.COLORS["info"])

        table_data = [[f"{title}: {value}"]]
        table = Table(table_data, colWidths=[170*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 14),
            ('PADDING', (0, 0), (-1, -1), 15),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 10))
        return elements

    def _build_regulatory_box(self, section: Dict, data: Dict) -> List:
        """Build regulatory compliance status box."""
        elements = []

        title = section.get("title", "Regulatory Status")
        reg_data = self._get_nested_value(data, section.get("data_key"))

        if not reg_data:
            return elements

        threshold = reg_data.get("threshold", 85)
        current = reg_data.get("current_compliance", 0)
        status = reg_data.get("status", "unknown")

        status_color = self.COLORS["success"] if status == "compliant" else self.COLORS["danger"]

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        table_data = [
            ["Threshold", "Current", "Status"],
            [f"{threshold}%", f"{current:.1f}%", status.upper()]
        ]

        col_width = 170*mm / 3
        table = Table(table_data, colWidths=[col_width, col_width, col_width])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['dark']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (2, 1), (2, 1), status_color),
            ('TEXTCOLOR', (2, 1), (2, 1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 15))
        return elements

    def _build_worker_profile(self, section: Dict, data: Dict) -> List:
        """Build worker profile section."""
        elements = []

        data_key = section.get("data_key", "worker_info")
        worker = self._get_nested_value(data, data_key)

        if not worker:
            return elements

        profile_data = [
            ["Name", worker.get("name", "-")],
            ["Employee ID", worker.get("employee_id", "-")],
            ["Department", worker.get("department", "-")],
            ["Shift", worker.get("assigned_shift", "-").title()]
        ]

        table = Table(profile_data, colWidths=[40*mm, 130*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), self.COLORS['light']),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 15))
        return elements

    def _build_badges_row(self, section: Dict, data: Dict) -> List:
        """Build a row of badges."""
        elements = []

        title = section.get("title", "Badges")
        badges = self._get_nested_value(data, section.get("data_key"))

        if not badges:
            elements.append(Paragraph(f"{title}: <i>No badges earned yet</i>", self.styles['Normal']))
            return elements

        elements.append(Paragraph(f"<b>{title}:</b> {', '.join(badges)}", self.styles['Normal']))
        elements.append(Spacer(1, 10))
        return elements

    def _build_score_gauge(self, section: Dict, data: Dict) -> List:
        """Build a score gauge visualization."""
        elements = []

        title = section.get("title", "Score")
        value = self._get_nested_value(data, section.get("value_key"))
        max_value = section.get("max_value", 100)

        if value is None:
            return elements

        # Simple text representation for now
        color = self.COLORS["green"] if value >= 80 else self.COLORS["orange"] if value >= 60 else self.COLORS["red"]

        elements.append(Paragraph(
            f"<font size='12'>{title}:</font> <font size='24' color='{color.hexval()}'><b>{value}</b></font><font size='12'>/{max_value}</font>",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 10))
        return elements

    def _get_nested_value(self, data: Dict, key: str):
        """Get a nested value from a dictionary using dot notation."""
        if not key or not data:
            return None

        keys = key.split(".")
        value = data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None

        return value
