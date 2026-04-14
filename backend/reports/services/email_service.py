"""
Email Service for sending scheduled reports.

Uses aiosmtplib for async email sending with attachment support.
"""

import os
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional
from datetime import datetime
from io import BytesIO
from jinja2 import Environment, BaseLoader

from ..schemas import EmailRecipient


class EmailService:
    """
    Service for sending report emails with attachments.
    """

    # Email HTML template
    EMAIL_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #1f2937;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #fb923c 0%, #f97316 100%);
                color: white;
                padding: 30px 20px;
                border-radius: 8px 8px 0 0;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
            }
            .header p {
                margin: 10px 0 0;
                opacity: 0.9;
            }
            .content {
                background: #ffffff;
                padding: 30px;
                border: 1px solid #e5e7eb;
                border-top: none;
            }
            .summary-box {
                background: #f9fafb;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }
            .metric {
                display: inline-block;
                text-align: center;
                padding: 10px 20px;
                margin: 5px;
                background: white;
                border-radius: 6px;
                border: 1px solid #e5e7eb;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #fb923c;
            }
            .metric-label {
                font-size: 12px;
                color: #6b7280;
            }
            .footer {
                background: #f3f4f6;
                padding: 20px;
                text-align: center;
                font-size: 12px;
                color: #6b7280;
                border-radius: 0 0 8px 8px;
                border: 1px solid #e5e7eb;
                border-top: none;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: #fb923c;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 500;
                margin-top: 15px;
            }
            .attachment-notice {
                background: #fef3c7;
                border-left: 4px solid #f59e0b;
                padding: 12px 16px;
                margin: 20px 0;
                border-radius: 0 4px 4px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ report_name }}</h1>
            <p>{{ date_range }}</p>
        </div>
        <div class="content">
            <p>Hello,</p>
            <p>Your scheduled report is ready. Please find the details below:</p>

            {% if summary_metrics %}
            <div class="summary-box">
                <h3 style="margin-top: 0;">Quick Summary</h3>
                {% for metric in summary_metrics %}
                <div class="metric">
                    <div class="metric-value" style="color: {{ metric.color }}">{{ metric.value }}</div>
                    <div class="metric-label">{{ metric.label }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="attachment-notice">
                <strong>📎 Attachment:</strong> The full report is attached to this email in {{ format }} format.
            </div>

            {% if highlights %}
            <h3>Key Highlights</h3>
            <ul>
            {% for highlight in highlights %}
                <li>{{ highlight }}</li>
            {% endfor %}
            </ul>
            {% endif %}

            <p>For detailed analysis, please download and review the attached report.</p>

        </div>
        <div class="footer">
            <p>Mine Safety PPE Detection System</p>
            <p>This is an automated report. Do not reply to this email.</p>
            <p>Generated on {{ generated_at }}</p>
        </div>
    </body>
    </html>
    """

    def __init__(self):
        """Initialize email service with configuration from environment."""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "reports@minesafety.com")
        self.from_name = os.getenv("FROM_NAME", "Mine Safety Reports")

        # Log configuration (without password)
        print(f"[EmailService] Initialized with:")
        print(f"  - SMTP Host: {self.smtp_host}:{self.smtp_port}")
        print(f"  - SMTP User: {self.smtp_user if self.smtp_user else 'NOT CONFIGURED'}")
        print(f"  - SMTP Password: {'***' if self.smtp_password else 'NOT CONFIGURED'}")
        print(f"  - From: {self.from_name} <{self.from_email}>")

        # Initialize Jinja2 environment
        self.jinja_env = Environment(loader=BaseLoader())

    async def send_report(
        self,
        recipients: List[EmailRecipient],
        report_name: str,
        date_range: str,
        attachments: List[dict],
        summary_metrics: Optional[List[dict]] = None,
        highlights: Optional[List[str]] = None
    ) -> bool:
        """
        Send a report email with attachments.

        Args:
            recipients: List of email recipients
            report_name: Name of the report
            date_range: Date range string
            attachments: List of attachment dicts with 'filename', 'data', 'content_type'
            summary_metrics: Optional list of summary metrics for email body
            highlights: Optional list of key highlights

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.smtp_user or not self.smtp_password:
            print("[EmailService] Warning: SMTP credentials not configured. Email not sent.")
            print(f"[EmailService] SMTP_USER: {self.smtp_user}, SMTP_PASSWORD: {'set' if self.smtp_password else 'not set'}")
            return False

        print(f"[EmailService] Sending report '{report_name}' to {len(recipients)} recipients...")

        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["Subject"] = f"[Report] {report_name} - {date_range}"

            # Set recipients
            to_recipients = [r.email for r in recipients if r.type == "to"]
            cc_recipients = [r.email for r in recipients if r.type == "cc"]
            bcc_recipients = [r.email for r in recipients if r.type == "bcc"]

            message["To"] = ", ".join(to_recipients)
            if cc_recipients:
                message["Cc"] = ", ".join(cc_recipients)

            # Render HTML body
            html_body = self._render_email_body(
                report_name=report_name,
                date_range=date_range,
                summary_metrics=summary_metrics or [],
                highlights=highlights or [],
                format=attachments[0].get("format", "PDF") if attachments else "PDF"
            )

            # Attach HTML body
            message.attach(MIMEText(html_body, "html"))

            # Attach files
            for attachment in attachments:
                self._attach_file(message, attachment)

            # Send email
            all_recipients = to_recipients + cc_recipients + bcc_recipients

            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=True,
                recipients=all_recipients
            )

            print(f"Report email sent successfully to {len(all_recipients)} recipients")
            return True

        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def _render_email_body(
        self,
        report_name: str,
        date_range: str,
        summary_metrics: List[dict],
        highlights: List[str],
        format: str
    ) -> str:
        """Render the HTML email body using Jinja2."""
        template = self.jinja_env.from_string(self.EMAIL_TEMPLATE)

        # Map color names to hex
        color_map = {
            "green": "#22c55e",
            "red": "#ef4444",
            "orange": "#f59e0b",
            "blue": "#3b82f6",
            "gray": "#6b7280",
            "primary": "#fb923c"
        }

        # Process metrics colors
        for metric in summary_metrics:
            color = metric.get("color", "primary")
            metric["color"] = color_map.get(color, color)

        return template.render(
            report_name=report_name,
            date_range=date_range,
            summary_metrics=summary_metrics,
            highlights=highlights,
            format=format.upper(),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        )

    def _attach_file(self, message: MIMEMultipart, attachment: dict):
        """Attach a file to the email message."""
        filename = attachment.get("filename", "report")
        data = attachment.get("data")
        content_type = attachment.get("content_type", "application/octet-stream")

        if isinstance(data, BytesIO):
            data = data.getvalue()

        # Determine MIME type
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            maintype, subtype = "application", "pdf"
        elif content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or filename.endswith(".xlsx"):
            maintype, subtype = "application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            maintype, subtype = content_type.split("/", 1) if "/" in content_type else ("application", "octet-stream")

        part = MIMEBase(maintype, subtype)
        part.set_payload(data)
        encoders.encode_base64(part)

        part.add_header(
            "Content-Disposition",
            f"attachment; filename={filename}"
        )

        message.attach(part)

    async def send_test_email(self, recipient_email: str) -> bool:
        """
        Send a test email to verify configuration.

        Args:
            recipient_email: Email address to send test to

        Returns:
            True if successful, False otherwise
        """
        try:
            message = MIMEMultipart()
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = recipient_email
            message["Subject"] = "[Test] Mine Safety Report System"

            body = """
            <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #fb923c;">Test Email Successful!</h2>
                <p>Your email configuration is working correctly.</p>
                <p>You will now receive scheduled reports at this address.</p>
                <hr style="border: 1px solid #e5e7eb;">
                <p style="color: #6b7280; font-size: 12px;">
                    Mine Safety PPE Detection System<br>
                    This is an automated test message.
                </p>
            </body>
            </html>
            """

            message.attach(MIMEText(body, "html"))

            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=True
            )

            return True

        except Exception as e:
            print(f"Test email failed: {e}")
            return False


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create the email service singleton."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
