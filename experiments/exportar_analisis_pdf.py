"""Exportar el componente analítico a dos páginas, sin entrenar agentes.

Dependencia opcional de documentación: reportlab. Python del proyecto o runtime
con reportlab; el análisis fuente siempre permanece accesible en Markdown.
"""

from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

ROOT=Path(__file__).resolve().parents[1]


def main():
    fonts=Path('C:/Windows/Fonts')
    if (fonts/'arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('Analysis',str(fonts/'arial.ttf')))
        pdfmetrics.registerFont(TTFont('AnalysisBold',str(fonts/'arialbd.ttf')))
        pdfmetrics.registerFontFamily('Analysis',normal='Analysis',bold='AnalysisBold')
        font,bold='Analysis','AnalysisBold'
    else:
        font,bold='Helvetica','Helvetica-Bold'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyAnalysis',fontName=font,fontSize=9.2,leading=11.8,spaceAfter=5,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='HeadAnalysis',fontName=bold,fontSize=10.5,leading=12.5,spaceBefore=8,spaceAfter=5,textColor=colors.HexColor('#164066')))
    styles.add(ParagraphStyle(name='TitleAnalysis',fontName=bold,fontSize=15,leading=18,spaceAfter=10))
    styles.add(ParagraphStyle(name='RefAnalysis',fontName=font,fontSize=7.8,leading=9.3,spaceAfter=2))
    styles.add(ParagraphStyle(name='CellAnalysis',fontName=font,fontSize=7.6,leading=9.3))

    def markup(text):
        text=escape(text).replace('–','-').replace('—','-')
        text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#164066">\1</link>',text)
        return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)

    source=(ROOT/'report/analisis_teorico.md').read_text(encoding='utf-8').splitlines()
    flow=[]; rows=[]; references=False
    def table():
        if not rows: return
        data=[[Paragraph(markup(v),styles['CellAnalysis']) for v in row] for row in rows]
        t=Table(data,colWidths=[105,75,53,80,80,117],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff6')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8ba3b8')),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
        flow.extend([t,Spacer(1,6)]); rows.clear()
    for line in source:
        if line.startswith('|'):
            if not re.fullmatch(r'[|:\- ]+',line): rows.append([s.strip() for s in line.strip('|').split('|')])
            continue
        table()
        if not line.strip(): continue
        if line.startswith('## 3.'):
            flow.append(PageBreak())
        if line.startswith('# '):
            flow.append(Paragraph(markup(line[2:]),styles['TitleAnalysis']))
        elif line.startswith('## '):
            references=line.startswith('## Referencias')
            flow.append(Paragraph(markup(line[3:]),styles['HeadAnalysis']))
        else:
            flow.append(Paragraph(markup(line),styles['RefAnalysis' if references else 'BodyAnalysis']))
    table()
    out=ROOT/'output/pdf/analisis_teorico.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
    def page(canvas,doc):
        canvas.setFont(font,8); canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(32,22,'Project 01 | Agentes Inteligentes | Análisis sin implementación')
        canvas.drawRightString(A4[0]-32,22,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=32,rightMargin=32,topMargin=29,bottomMargin=35,
        title='Análisis teórico - Project 01',author='Equipo Project 01')
    doc.build(flow,onFirstPage=page,onLaterPages=page)
    print(out)


if __name__=='__main__': main()
