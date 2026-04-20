from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import qrcode
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from app.persistence import create_certificate_record, get_certificate_record
from app.state import LearnerSession


class CertificateRecord(BaseModel):
    certificate_id: str
    learner_id: str | None = None
    learner_name: str
    learner_email: str | None = None
    micro_credential_title: str
    issue_date: str
    issued_at: str
    verification_url: str
    qr_code_url: str
    pdf_url: str
    competencies: list[dict[str, Any]] = Field(default_factory=list)
    profile_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


_CERTIFICATES: dict[str, CertificateRecord] = {}
_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "generated_certificates"


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_profile_root(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("data", "user", "profile", "results"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def learner_name_from_profile(payload: dict[str, Any]) -> str:
    root = _extract_profile_root(payload)
    full_name = _first_non_empty(
        root.get("full_name"),
        root.get("name"),
        payload.get("full_name"),
        payload.get("name"),
    )
    if full_name:
        return full_name

    first_name = _first_non_empty(root.get("first_name"), payload.get("first_name"))
    last_name = _first_non_empty(root.get("last_name"), payload.get("last_name"))
    if first_name or last_name:
        return " ".join(part for part in (first_name, last_name) if part)

    email = learner_email_from_profile(payload)
    if email:
        return email.split("@", 1)[0]
    return "IKON Learner"


def learner_email_from_profile(payload: dict[str, Any]) -> str | None:
    root = _extract_profile_root(payload)
    return _first_non_empty(root.get("email"), payload.get("email"))


def learner_identifier_from_profile(payload: dict[str, Any], fallback: str | None = None) -> str | None:
    root = _extract_profile_root(payload)
    value = root.get("id") or root.get("user_id") or payload.get("id") or payload.get("user_id") or fallback
    return str(value) if value is not None else None


def build_certificate_id(session: LearnerSession, learner_name: str) -> str:
    digest = sha256(f"{session.session_id}:{learner_name}:{session.topic}".encode("utf-8")).hexdigest()[:10].upper()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"IKON-{stamp}-{digest}"


def build_verification_url(base_url: str, certificate_id: str) -> str:
    return f"{base_url.rstrip('/')}/certificate/verify/{certificate_id}"


def build_qr_code_url(base_url: str, certificate_id: str) -> str:
    return f"{base_url.rstrip('/')}/certificate/{certificate_id}/qr.png"


def build_pdf_url(base_url: str, certificate_id: str) -> str:
    return f"{base_url.rstrip('/')}/certificate/{certificate_id}/pdf"


def render_qr_png(record: CertificateRecord, *, box_size: int = 10, border: int = 2) -> bytes:
    qr = qrcode.QRCode(version=None, box_size=box_size, border=border)
    qr.add_data(record.verification_url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="#18203a", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_certificate_html(record: CertificateRecord) -> str:
    competency_items = "".join(
        f"<li><strong>{item.get('competency', 'Competency')}</strong> - {float(item.get('score', 0.0)):.1f}%</li>"
        for item in record.competencies
    )
    competency_section = f"<ul>{competency_items}</ul>" if competency_items else "<p>Competency scores unavailable.</p>"
    learner_name = record.learner_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    title = record.micro_credential_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    issue_date = record.issue_date.replace("&", "&amp;")
    certificate_id = record.certificate_id.replace("&", "&amp;")
    verification_url = record.verification_url.replace("&", "&amp;")
    qr_code_url = record.qr_code_url.replace("&", "&amp;")
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{certificate_id}</title>
  <style>
    body {{ background: #efe6cf; margin: 0; padding: 24px; font-family: Georgia, serif; }}
    .sheet {{ max-width: 980px; margin: 0 auto; background: #f6f0df; border: 8px solid #b68a2f; padding: 28px 40px 36px; box-shadow: 0 8px 30px rgba(0,0,0,.16); position: relative; }}
    .brand {{ font-family: Arial, sans-serif; font-size: 56px; color: #921a14; font-weight: 700; letter-spacing: 1px; }}
    .subbrand {{ font-family: Arial, sans-serif; font-size: 18px; color: #4b4537; margin-top: -6px; }}
    .title {{ text-align: center; margin: 36px 0 8px; font-family: Arial, sans-serif; font-size: 50px; font-weight: 700; }}
    .subtitle {{ text-align: center; font-family: Arial, sans-serif; letter-spacing: 3px; color: #756e60; font-size: 18px; }}
    .recipient {{ text-align: center; font-size: 62px; font-weight: 700; margin: 42px 0 20px; }}
    .course {{ text-align: center; font-size: 48px; font-weight: 700; margin: 18px 0 22px; }}
    .meta {{ text-align: center; font-family: Arial, sans-serif; font-size: 20px; margin: 18px 0 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 260px; gap: 28px; align-items: start; margin-top: 28px; }}
    .qr {{ text-align: center; border: 2px solid #b84a3b; padding: 12px; background: white; }}
    .qr img {{ width: 220px; height: 220px; display: block; margin: 0 auto; }}
    .verify {{ font-family: Arial, sans-serif; font-size: 14px; margin-top: 10px; color: #4b4537; word-break: break-all; }}
    .section {{ border-top: 1px solid #bba36e; margin-top: 24px; padding-top: 18px; }}
    .section h3 {{ font-family: Arial, sans-serif; font-size: 22px; margin: 0 0 12px; }}
    .section p, .section li {{ font-family: Arial, sans-serif; font-size: 18px; line-height: 1.6; }}
    .footer {{ display: flex; justify-content: space-between; align-items: end; margin-top: 46px; font-family: Arial, sans-serif; }}
    .signature {{ font-size: 20px; border-top: 1px solid #333; padding-top: 8px; width: 340px; }}
    .seal {{ width: 160px; height: 160px; border-radius: 999px; background: radial-gradient(circle, rgba(209,110,44,0.95), rgba(160,56,23,0.95)); opacity: 0.78; }}
  </style>
</head>
<body>
  <div class=\"sheet\">
    <div class=\"brand\">IKON SKILLS</div>
    <div class=\"subbrand\">Micro-Credential Platform | ikonskills.ac</div>
    <div class=\"title\">IKON SKILLS</div>
    <div class=\"subtitle\">THIS CERTIFIES THAT</div>
    <div class=\"recipient\">{learner_name}</div>
    <div class=\"subtitle\">HAS COMPLETED THE MICRO-CREDENTIAL</div>
    <div class=\"course\">{title}</div>
    <div class=\"meta\">10 ECTS | Competency Verified | IKON Practitioner Status Awarded</div>
    <div class=\"meta\">Issued: {issue_date} | Certificate ID: {certificate_id}</div>

    <div class=\"grid\">
      <div>
        <div class=\"section\">
          <h3>Credential Summary</h3>
          <p>This certificate confirms successful completion of the micro-credential and verified competency performance across the required lesson pathway.</p>
        </div>
        <div class=\"section\">
          <h3>Verified Competencies</h3>
          {competency_section}
        </div>
      </div>
      <div class=\"qr\">
        <img src=\"{qr_code_url}\" alt=\"Certificate QR Code\" />
        <div class=\"verify\">Scan to verify<br />{verification_url}</div>
      </div>
    </div>

    <div class=\"footer\">
      <div class=\"signature\">Hon. ky. Col. Prof. Dr. [Signer Name]</div>
      <div class=\"seal\"></div>
    </div>
  </div>
</body>
</html>"""


def render_certificate_pdf(record: CertificateRecord) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setTitle(record.certificate_id)
    pdf.setStrokeColor(colors.HexColor("#b68a2f"))
    pdf.setLineWidth(4)
    pdf.rect(18, 18, width - 36, height - 36)
    pdf.setLineWidth(1)
    pdf.rect(28, 28, width - 56, height - 56)

    pdf.setFont("Helvetica-Bold", 28)
    pdf.setFillColor(colors.HexColor("#921a14"))
    pdf.drawString(50, height - 72, "IKON SKILLS")

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#4b4537"))
    pdf.drawString(50, height - 88, "Micro-Credential Platform | ikonskills.ac")

    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 130, "IKON SKILLS")

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor("#756e60"))
    pdf.drawCentredString(width / 2, height - 150, "THIS CERTIFIES THAT")

    pdf.setFont("Times-Bold", 26)
    pdf.setFillColor(colors.black)
    pdf.drawCentredString(width / 2, height - 205, record.learner_name)

    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 232, "HAS COMPLETED THE MICRO-CREDENTIAL")

    pdf.setFont("Times-Bold", 22)
    pdf.drawCentredString(width / 2, height - 270, record.micro_credential_title)

    pdf.setFont("Helvetica", 11)
    pdf.drawCentredString(width / 2, height - 295, "10 ECTS | Competency Verified | IKON Practitioner Status Awarded")
    pdf.drawCentredString(width / 2, height - 314, f"Issued: {record.issue_date} | Certificate ID: {record.certificate_id}")

    pdf.line(50, height - 335, width - 50, height - 335)

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(56, height - 365, "Verified Competencies")
    pdf.setFont("Helvetica", 11)
    y = height - 386
    if record.competencies:
        for item in record.competencies[:8]:
            pdf.drawString(64, y, f"- {item.get('competency', 'Competency')}: {float(item.get('score', 0.0)):.1f}%")
            y -= 18
    else:
        pdf.drawString(64, y, "Competency scores unavailable.")
        y -= 18

    qr_reader = ImageReader(BytesIO(render_qr_png(record, box_size=8)))
    qr_size = 150
    qr_x = width - 210
    qr_y = height - 500
    pdf.drawImage(qr_reader, qr_x, qr_y, width=qr_size, height=qr_size, preserveAspectRatio=True, mask='auto')
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(qr_x + qr_size / 2, qr_y - 14, "Scan to verify")
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(qr_x + qr_size / 2, qr_y - 28, record.verification_url)

    pdf.line(50, 180, width - 50, 180)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(56, 155, "Hon. ky. Col. Prof. Dr. [Signer Name]")
    pdf.drawString(56, 138, "Founder & CEO, IKON Educational & Psychological Consultancy")
    pdf.drawString(56, 121, "Founder & Director General, European International University, Paris")

    pdf.setFillColor(colors.HexColor("#a03817"))
    pdf.circle(width - 110, 118, 48, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(width - 110, 118, "IKON")

    pdf.showPage()
    pdf.save()
    return buffer.getvalue()




def _persist_certificate_artifacts(record: CertificateRecord) -> CertificateRecord:
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = _ARTIFACT_DIR / f"{record.certificate_id}.html"
    pdf_path = _ARTIFACT_DIR / f"{record.certificate_id}.pdf"

    html_path.write_text(render_certificate_html(record), encoding="utf-8")
    pdf_path.write_bytes(render_certificate_pdf(record))

    record.metadata.update(
        {
            "artifact_dir": str(_ARTIFACT_DIR),
            "html_file_path": str(html_path),
            "pdf_file_path": str(pdf_path),
        }
    )
    return record

def issue_certificate(session: LearnerSession, profile_payload: dict[str, Any], public_base_url: str) -> CertificateRecord:
    learner_name = learner_name_from_profile(profile_payload)
    learner_id = learner_identifier_from_profile(profile_payload, session.learner_id)
    learner_email = learner_email_from_profile(profile_payload)
    certificate_id = build_certificate_id(session, learner_name)
    verification_url = build_verification_url(public_base_url, certificate_id)
    qr_code_url = build_qr_code_url(public_base_url, certificate_id)
    pdf_url = build_pdf_url(public_base_url, certificate_id)
    now = datetime.now(timezone.utc)
    record = CertificateRecord(
        certificate_id=certificate_id,
        learner_id=learner_id,
        learner_name=learner_name,
        learner_email=learner_email,
        micro_credential_title=session.topic,
        issue_date=now.strftime("%d %B %Y"),
        issued_at=now.isoformat(),
        verification_url=verification_url,
        qr_code_url=qr_code_url,
        pdf_url=pdf_url,
        competencies=[item.model_dump() for item in session.completed_competencies],
        profile_payload=profile_payload,
        metadata={
            "domain_id": session.domain_id,
            "micro_credential_id": session.remote_micro_credential_id,
            "session_id": session.session_id,
            "completed_competencies": len(session.completed_competencies),
            "learner_name": learner_name,
            "learner_email": learner_email,
            "micro_credential_title": session.topic,
            "issue_date": now.strftime("%d %B %Y"),
            "competencies": [item.model_dump() for item in session.completed_competencies],
            "profile_payload": profile_payload,
            "public_base_url": public_base_url,
            "pdf_url": pdf_url,
        },
    )
    record = _persist_certificate_artifacts(record)
    create_certificate_record(
        certificate_id=record.certificate_id,
        session_id=session.session_id,
        learner_id=record.learner_id,
        html_file_path=record.metadata["html_file_path"],
        pdf_file_path=record.metadata["pdf_file_path"],
        verification_url=record.verification_url,
        qr_code_url=record.qr_code_url,
        metadata=record.metadata,
        issued_at=record.issued_at,
    )
    _CERTIFICATES[certificate_id] = record
    return record


def get_certificate(certificate_id: str) -> CertificateRecord | None:
    cached = _CERTIFICATES.get(certificate_id)
    if cached:
        return cached

    persisted = get_certificate_record(certificate_id)
    if not persisted:
        return None

    metadata = persisted.get("metadata", {})
    html_path = metadata.get("html_file_path")
    pdf_path = metadata.get("pdf_file_path")
    session_metadata = dict(metadata)
    record = CertificateRecord(
        certificate_id=certificate_id,
        learner_id=persisted.get("learner_id"),
        learner_name=session_metadata.get("learner_name") or "IKON Learner",
        learner_email=session_metadata.get("learner_email"),
        micro_credential_title=session_metadata.get("micro_credential_title") or "Micro-Credential",
        issue_date=session_metadata.get("issue_date") or "",
        issued_at=persisted.get("issued_at") or "",
        verification_url=persisted.get("verification_url") or "",
        qr_code_url=persisted.get("qr_code_url") or session_metadata.get("qr_code_url") or "",
        pdf_url=session_metadata.get("pdf_url") or "",
        competencies=session_metadata.get("competencies", []),
        profile_payload=session_metadata.get("profile_payload", {}),
        metadata=session_metadata,
    )
    if html_path and Path(html_path).exists():
        record.metadata["html_file_path"] = html_path
    if pdf_path and Path(pdf_path).exists():
        record.metadata["pdf_file_path"] = pdf_path
    _CERTIFICATES[certificate_id] = record
    return record
