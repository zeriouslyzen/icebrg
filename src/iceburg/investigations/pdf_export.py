"""
PDF Export - Professional styled PDF export for ICEBURG dossiers.
Creates intel-style documents with ICEBURG dark theme styling.
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import reportlab, fall back gracefully
REPORTLAB_AVAILABLE = False
IceburgColors = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable, ListFlowable, ListItem
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
    
    # ICEBURG color palette - only define when reportlab is available
    class IceburgColors:
        """ICEBURG brand colors for PDF styling."""
        BACKGROUND = colors.HexColor("#0a0a0f")
        SURFACE = colors.HexColor("#12121a")
        SURFACE_ELEVATED = colors.HexColor("#1a1a24")
        TEXT_PRIMARY = colors.HexColor("#e8e8ed")
        TEXT_SECONDARY = colors.HexColor("#a0a0b0")
        TEXT_TERTIARY = colors.HexColor("#606070")
        ACCENT_PRIMARY = colors.HexColor("#00d4ff")  # Cyan
        ACCENT_SECONDARY = colors.HexColor("#8b5cf6")  # Purple
        WARNING = colors.HexColor("#fbbf24")
        ERROR = colors.HexColor("#ef4444")
        SUCCESS = colors.HexColor("#10b981")
        BORDER = colors.HexColor("#2a2a3a")
        
except ImportError:
    logger.warning("reportlab not installed. PDF export disabled. Install with: pip install reportlab")


def create_iceburg_styles():
    """Create custom paragraph styles for ICEBURG PDFs."""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='IceburgTitle',
        fontName='Helvetica-Bold',
        fontSize=28,
        textColor=IceburgColors.ACCENT_PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=20,
        spaceBefore=40
    ))
    
    # Subtitle
    styles.add(ParagraphStyle(
        name='IceburgSubtitle',
        fontName='Helvetica',
        fontSize=14,
        textColor=IceburgColors.TEXT_SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=30
    ))
    
    # Section header
    styles.add(ParagraphStyle(
        name='IceburgH1',
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=IceburgColors.ACCENT_PRIMARY,
        spaceBefore=20,
        spaceAfter=10,
        borderColor=IceburgColors.ACCENT_PRIMARY,
        borderWidth=2,
        borderPadding=5
    ))
    
    # Subsection header
    styles.add(ParagraphStyle(
        name='IceburgH2',
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=IceburgColors.ACCENT_SECONDARY,
        spaceBefore=15,
        spaceAfter=8
    ))
    
    # Body text
    styles.add(ParagraphStyle(
        name='IceburgBody',
        fontName='Helvetica',
        fontSize=11,
        textColor=IceburgColors.TEXT_PRIMARY,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14
    ))
    
    # Quote/highlight
    styles.add(ParagraphStyle(
        name='IceburgQuote',
        fontName='Helvetica-Oblique',
        fontSize=11,
        textColor=IceburgColors.TEXT_SECONDARY,
        leftIndent=20,
        rightIndent=20,
        spaceBefore=10,
        spaceAfter=10,
        borderColor=IceburgColors.ACCENT_SECONDARY,
        borderWidth=1,
        borderPadding=10
    ))
    
    # Classification marking
    styles.add(ParagraphStyle(
        name='IceburgClassification',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=IceburgColors.WARNING,
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=5
    ))
    
    # Footer
    styles.add(ParagraphStyle(
        name='IceburgFooter',
        fontName='Helvetica',
        fontSize=8,
        textColor=IceburgColors.TEXT_TERTIARY,
        alignment=TA_CENTER
    ))
    
    # Table header
    styles.add(ParagraphStyle(
        name='IceburgTableHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=IceburgColors.ACCENT_PRIMARY,
        alignment=TA_LEFT
    ))
    
    # Table cell
    styles.add(ParagraphStyle(
        name='IceburgTableCell',
        fontName='Helvetica',
        fontSize=9,
        textColor=IceburgColors.TEXT_PRIMARY,
        alignment=TA_LEFT
    ))
    
    return styles


class DossierPDFExporter:
    """
    Export ICEBURG investigations to professionally styled PDFs.
    """
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
        self.styles = create_iceburg_styles()
    
    def export(
        self,
        investigation: "Investigation",
        output_path: Optional[Path] = None,
        classification: str = "ICEBURG INTERNAL"
    ) -> Path:
        """
        Export an investigation to PDF.
        
        Args:
            investigation: Investigation object to export
            output_path: Optional output path (defaults to investigation exports dir)
            classification: Classification marking for header/footer
            
        Returns:
            Path to the generated PDF
        """
        from .storage import get_investigation_store
        
        # Determine output path
        if output_path is None:
            store = get_investigation_store()
            inv_dir = store.get_investigation_dir(investigation.metadata.investigation_id)
            exports_dir = inv_dir / "exports"
            exports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exports_dir / f"dossier_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        
        # Build story (content)
        story = []
        
        # Cover page
        story.extend(self._build_cover_page(investigation, classification))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._build_executive_summary(investigation))
        
        # Official narrative
        story.extend(self._build_official_narrative(investigation))
        
        # Alternative narratives
        story.extend(self._build_alternative_narratives(investigation))
        
        # Key players
        story.extend(self._build_key_players(investigation))
        
        # Matrix summary (connects, bridges from Colossus query APIs)
        story.extend(self._build_matrix_summary(investigation))
        
        # Contradictions
        if investigation.contradictions:
            story.extend(self._build_contradictions(investigation))
        
        # Historical parallels
        if investigation.historical_parallels:
            story.extend(self._build_historical_parallels(investigation))
        
        # Follow-up suggestions
        if investigation.follow_up_suggestions:
            story.extend(self._build_follow_up(investigation))
        
        # Source appendix
        story.extend(self._build_sources_appendix(investigation))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
        
        logger.info(f"📄 PDF exported: {output_path}")
        return output_path
    
    def _build_cover_page(self, investigation: "Investigation", classification: str) -> List:
        """Build the cover page."""
        elements = []
        
        # Classification header
        elements.append(Paragraph(f"⚠️ {classification} ⚠️", self.styles['IceburgClassification']))
        elements.append(Spacer(1, 50))
        
        # Logo/title
        elements.append(Paragraph("🧊 ICEBURG", self.styles['IceburgTitle']))
        elements.append(Paragraph("INTELLIGENCE DOSSIER", self.styles['IceburgSubtitle']))
        elements.append(Spacer(1, 30))
        
        # Query/title
        elements.append(HRFlowable(width="80%", thickness=2, color=IceburgColors.ACCENT_PRIMARY, spaceAfter=20))
        elements.append(Paragraph(investigation.metadata.title, self.styles['IceburgH1']))
        elements.append(HRFlowable(width="80%", thickness=2, color=IceburgColors.ACCENT_PRIMARY, spaceBefore=20))
        elements.append(Spacer(1, 40))
        
        # Metadata table
        meta = investigation.metadata
        meta_data = [
            ["Investigation ID:", meta.investigation_id],
            ["Generated:", meta.created_at[:19].replace("T", " ")],
            ["Status:", meta.status.upper()],
            ["Confidence:", f"{meta.confidence_score:.0%}"],
            ["Sources:", str(meta.sources_count)],
            ["Entities:", str(meta.entities_count)],
            ["Tags:", ", ".join(meta.tags) if meta.tags else "None"],
        ]
        
        meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), IceburgColors.ACCENT_SECONDARY),
            ('TEXTCOLOR', (1, 0), (1, -1), IceburgColors.TEXT_PRIMARY),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(meta_table)
        
        return elements
    
    def _build_executive_summary(self, investigation: "Investigation") -> List:
        """Build the executive summary section."""
        elements = []
        
        elements.append(Paragraph("📋 EXECUTIVE SUMMARY", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        if investigation.executive_summary:
            elements.append(Paragraph(investigation.executive_summary, self.styles['IceburgBody']))
        else:
            elements.append(Paragraph("No executive summary available.", self.styles['IceburgBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_official_narrative(self, investigation: "Investigation") -> List:
        """Build the official narrative section."""
        elements = []
        
        elements.append(Paragraph("📰 OFFICIAL NARRATIVE", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        if investigation.official_narrative:
            elements.append(Paragraph(investigation.official_narrative, self.styles['IceburgBody']))
        else:
            elements.append(Paragraph("No official narrative documented.", self.styles['IceburgBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_alternative_narratives(self, investigation: "Investigation") -> List:
        """Build the alternative narratives section."""
        elements = []
        
        elements.append(Paragraph("🔀 ALTERNATIVE NARRATIVES", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        if investigation.alternative_narratives:
            for i, narrative in enumerate(investigation.alternative_narratives, 1):
                if isinstance(narrative, dict):
                    title = narrative.get('title', f'Narrative {i}')
                    content = narrative.get('narrative', narrative.get('content', ''))
                else:
                    title = f"Narrative {i}"
                    content = str(narrative)
                
                elements.append(Paragraph(f"<b>{i}. {title}</b>", self.styles['IceburgH2']))
                elements.append(Paragraph(content, self.styles['IceburgBody']))
                elements.append(Spacer(1, 10))
        else:
            elements.append(Paragraph("No alternative narratives identified.", self.styles['IceburgBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_key_players(self, investigation: "Investigation") -> List:
        """Build the key players section with table."""
        elements = []
        
        elements.append(Paragraph("👥 KEY PLAYERS", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        if investigation.key_players:
            # Build table data
            table_data = [["Name", "Type", "Role", "Connections"]]
            
            for player in investigation.key_players[:15]:  # Limit to 15
                name = player.get('name', 'Unknown')
                player_type = player.get('type', player.get('entity_type', 'Unknown'))
                role = player.get('role', player.get('description', ''))[:50]
                connections = str(player.get('connections', player.get('connection_count', 0)))
                table_data.append([name, player_type, role, connections])
            
            table = Table(table_data, colWidths=[1.5*inch, 1*inch, 3*inch, 0.8*inch])
            table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), IceburgColors.SURFACE_ELEVATED),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('TEXTCOLOR', (0, 0), (-1, 0), IceburgColors.ACCENT_PRIMARY),
                # Body rows
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('TEXTCOLOR', (0, 1), (-1, -1), IceburgColors.TEXT_PRIMARY),
                # Grid
                ('GRID', (0, 0), (-1, -1), 0.5, IceburgColors.BORDER),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                # Alternating rows
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [IceburgColors.BACKGROUND, IceburgColors.SURFACE]),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph("No key players identified.", self.styles['IceburgBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_matrix_summary(self, investigation: "Investigation") -> List:
        """Build Matrix summary: who connects, who bridges (from Colossus query APIs)."""
        elements = []
        try:
            from ..colossus.api import get_graph
            graph = get_graph()
            connects = graph.get_relationships_by_type(["CONNECTS", "GATEKEEPER_FOR"], limit=50)
            bridges = graph.get_bridge_entities(limit=30)
        except Exception as e:
            logger.debug("Matrix summary skipped (Colossus unavailable): %s", e)
            return []
        if not connects and not bridges:
            return []
        elements.append(Paragraph("NETWORK MATRIX SUMMARY", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        if connects:
            elements.append(Paragraph("Connectors / gatekeepers (CONNECTS, GATEKEEPER_FOR):", self.styles['IceburgH2']))
            for r in connects[:15]:
                elements.append(Paragraph(
                    f"  {r.source_id} -> {r.target_id} ({r.relationship_type})",
                    self.styles['IceburgBody']
                ))
            elements.append(Spacer(1, 8))
        if bridges:
            elements.append(Paragraph("Bridge entities (multiple domains):", self.styles['IceburgH2']))
            for e in bridges[:15]:
                domains = (e.properties or {}).get("domains") or []
                elements.append(Paragraph(
                    f"  {e.name} ({e.id}): {', '.join(domains)}",
                    self.styles['IceburgBody']
                ))
        elements.append(Spacer(1, 20))
        return elements

    def _build_contradictions(self, investigation: "Investigation") -> List:
        """Build the contradictions section."""
        elements = []
        
        elements.append(Paragraph("⚠️ CONTRADICTIONS DETECTED", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        for i, contradiction in enumerate(investigation.contradictions[:10], 1):
            if isinstance(contradiction, dict):
                text = contradiction.get('description', contradiction.get('contradiction', str(contradiction)))
            else:
                text = str(contradiction)
            elements.append(Paragraph(f"{i}. {text}", self.styles['IceburgQuote']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_historical_parallels(self, investigation: "Investigation") -> List:
        """Build the historical parallels section."""
        elements = []
        
        elements.append(Paragraph("📜 HISTORICAL PARALLELS", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        for i, parallel in enumerate(investigation.historical_parallels[:5], 1):
            if isinstance(parallel, dict):
                event = parallel.get('event', parallel.get('title', f'Parallel {i}'))
                relevance = parallel.get('relevance', parallel.get('description', ''))
            else:
                event = f"Parallel {i}"
                relevance = str(parallel)
            
            elements.append(Paragraph(f"<b>{event}</b>", self.styles['IceburgH2']))
            elements.append(Paragraph(relevance, self.styles['IceburgBody']))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_follow_up(self, investigation: "Investigation") -> List:
        """Build follow-up suggestions section."""
        elements = []
        
        elements.append(Paragraph("🔍 SUGGESTED FOLLOW-UP", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        items = []
        for suggestion in investigation.follow_up_suggestions[:10]:
            items.append(ListItem(Paragraph(suggestion, self.styles['IceburgBody'])))
        
        if items:
            elements.append(ListFlowable(items, bulletType='bullet', start='•'))
        
        elements.append(Spacer(1, 20))
        return elements
    
    def _build_sources_appendix(self, investigation: "Investigation") -> List:
        """Build sources appendix."""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("📚 SOURCE APPENDIX", self.styles['IceburgH1']))
        elements.append(Spacer(1, 10))
        
        elements.append(Paragraph(
            f"This dossier was compiled from {investigation.metadata.sources_count} sources across multiple domains.",
            self.styles['IceburgBody']
        ))
        elements.append(Spacer(1, 10))
        
        if investigation.sources:
            for i, source in enumerate(investigation.sources[:20], 1):
                if isinstance(source, dict):
                    title = source.get('title', source.get('name', f'Source {i}'))
                    url = source.get('url', '')
                    source_type = source.get('type', source.get('source_type', 'Unknown'))
                else:
                    title = f"Source {i}"
                    url = str(source)
                    source_type = "Unknown"
                
                elements.append(Paragraph(
                    f"<b>{i}. [{source_type}]</b> {title}",
                    self.styles['IceburgBody']
                ))
                if url:
                    elements.append(Paragraph(f"   {url}", self.styles['IceburgFooter']))
        else:
            elements.append(Paragraph(
                "Detailed source list not available for this export.",
                self.styles['IceburgBody']
            ))
        
        return elements
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header - classification
        canvas.setFillColor(IceburgColors.WARNING)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawCentredString(letter[0]/2, letter[1] - 0.5*inch, "ICEBURG INTERNAL - DOSSIER")
        
        # Footer - page number and timestamp
        canvas.setFillColor(IceburgColors.TEXT_TERTIARY)
        canvas.setFont('Helvetica', 8)
        canvas.drawCentredString(
            letter[0]/2, 
            0.4*inch, 
            f"Page {doc.page} | Generated by ICEBURG Dossier Protocol | {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Side border accent
        canvas.setStrokeColor(IceburgColors.ACCENT_PRIMARY)
        canvas.setLineWidth(3)
        canvas.line(0.25*inch, 0.5*inch, 0.25*inch, letter[1] - 0.5*inch)
        
        canvas.restoreState()


def export_investigation_to_pdf(investigation_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Convenience function to export an investigation to PDF by ID.
    
    Args:
        investigation_id: The investigation ID to export
        output_path: Optional custom output path
        
    Returns:
        Path to the generated PDF, or None if export failed
    """
    if not REPORTLAB_AVAILABLE:
        logger.error("PDF export unavailable - reportlab not installed")
        return None
    
    try:
        from .storage import get_investigation_store
        
        store = get_investigation_store()
        investigation = store.load(investigation_id)
        
        if investigation is None:
            logger.error(f"Investigation not found: {investigation_id}")
            return None
        
        exporter = DossierPDFExporter()
        return exporter.export(investigation, output_path)
        
    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        return None
