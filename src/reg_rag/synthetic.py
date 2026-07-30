from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from docx import Document
from openpyxl import Workbook
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def build_synthetic_dataset(out_dir: Path) -> dict:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs = [
        _write_regulation_docx_old(raw_dir / "reg_000_liquidity_old.docx"),
        _write_regulation_docx(raw_dir / "reg_001_liquidity.docx"),
        _write_regulation_docx_two(raw_dir / "reg_002_complaints.docx"),
        _write_capital_table_docx(raw_dir / "reg_003_capital_rules.docx"),
        _write_aml_docx(raw_dir / "reg_004_aml_reporting.docx"),
        _write_pdf_rules(raw_dir / "reg_005_payment_settlement.pdf"),
        _write_statistics_csv(raw_dir / "stat_001_inclusive_finance.csv"),
        _write_liquidity_statistics_csv(raw_dir / "stat_002_liquidity_metrics.csv"),
        _write_complaint_statistics_csv(raw_dir / "stat_003_complaint_summary.csv"),
        _write_dirty_statistics_csv(raw_dir / "stat_004_dirty_headers.csv"),
        _write_merged_header_xlsx(raw_dir / "stat_005_merged_multisheet.xlsx"),
        _write_qa(out_dir / "qa_seed.jsonl"),
    ]

    manifest_rows = []
    for path in docs:
        if path.name == "qa_seed.jsonl":
            continue
        manifest_rows.append(_manifest_row(path))

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "title",
                "source_url",
                "attachment_url",
                "local_path",
                "file_size",
                "sha256",
                "column",
                "file_type",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "qa_seed": str(out_dir / "qa_seed.jsonl"),
        "files": [str(path) for path in docs],
    }


def _write_regulation_docx_old(path: Path) -> Path:
    doc = Document()
    doc.add_heading("虚拟银行业流动性风险管理办法（旧版）", level=1)
    doc.add_paragraph("发文机关：虚拟金融监督管理局")
    doc.add_paragraph("发布日期：2023-01-01")
    doc.add_paragraph("状态：已被2025年版替代，仅用于历史口径查询。")
    doc.add_heading("第二章 指标要求", level=2)
    doc.add_paragraph("第八条 商业银行流动性覆盖率不得低于90%。")
    doc.add_paragraph("第九条 商业银行净稳定资金比例不得低于95%。")
    doc.save(path)
    return path


def _write_regulation_docx(path: Path) -> Path:
    doc = Document()
    doc.add_heading("虚拟银行业流动性风险管理办法（2025年版）", level=1)
    doc.add_paragraph("发文机关：虚拟金融监督管理局")
    doc.add_paragraph("发布日期：2025-03-15")
    doc.add_heading("第一章 总则", level=2)
    doc.add_paragraph("第一条 为加强商业银行流动性风险管理，维护银行体系稳健运行，制定本办法。")
    doc.add_paragraph("第二条 本办法适用于在中华人民共和国境内依法设立的商业银行。")
    doc.add_heading("第二章 指标要求", level=2)
    doc.add_paragraph("第八条 商业银行流动性覆盖率不得低于100%。")
    doc.add_paragraph("第九条 商业银行净稳定资金比例不得低于100%。")
    doc.add_paragraph("第十条 商业银行应当至少每季度开展一次流动性压力测试，并保存测试记录不少于五年。")
    doc.add_heading("第三章 报告要求", level=2)
    doc.add_paragraph("第十五条 商业银行发现重大流动性风险事件后，应当在二十四小时内向监管部门报告。")
    doc.save(path)
    return path


def _write_capital_table_docx(path: Path) -> Path:
    doc = Document()
    doc.add_heading("虚拟商业银行资本管理指标口径表", level=1)
    doc.add_paragraph("发文机关：虚拟金融监督管理局资本监管司")
    doc.add_paragraph("发布日期：2025-04-10")
    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    hdr[0].text = "指标名称"
    hdr[1].text = "最低要求"
    hdr[2].text = "适用对象"
    hdr[3].text = "生效日期"
    hdr[4].text = "备注"
    rows = [
        ["资本充足率", "10.5%", "商业银行", "2025-07-01", "含储备资本要求"],
        ["一级资本充足率", "8.5%", "商业银行", "2025-07-01", "含附加资本要求"],
        ["核心一级资本充足率", "7.5%", "商业银行", "2025-07-01", "不得低于监管底线"],
        ["杠杆率", "4%", "商业银行", "2025-07-01", "按季报送"],
    ]
    for values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            cells[idx].text = value
    doc.save(path)
    return path


def _write_aml_docx(path: Path) -> Path:
    doc = Document()
    doc.add_heading("虚拟银行业反洗钱可疑交易报告规程", level=1)
    doc.add_paragraph("发文机关：虚拟人民银行反洗钱局")
    doc.add_paragraph("发布日期：2025-02-18")
    doc.add_heading("第一章 报告要求", level=2)
    doc.add_paragraph("第三条 银行业金融机构发现可疑交易后，应当在五个工作日内提交可疑交易报告。")
    doc.add_paragraph("第四条 对明显涉嫌洗钱、恐怖融资等重大风险事项，应当立即采取强化尽职调查措施。")
    doc.add_heading("第二章 记录保存", level=2)
    doc.add_paragraph("第十条 客户身份资料和交易记录保存期限不得少于五年。")
    doc.add_paragraph("第十一条 保存期限届满但涉及正在调查的可疑交易，相关资料应当保存至调查结束。")
    doc.save(path)
    return path


def _write_pdf_rules(path: Path) -> Path:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c = canvas.Canvas(str(path))
    c.setFont("STSong-Light", 12)
    y = 800
    lines = [
        "虚拟支付结算业务管理细则",
        "发文机关：虚拟人民银行支付结算司",
        "发布日期：2025-06-01",
        "第一章 总则",
        "第一条 为规范银行业金融机构支付结算业务，保护客户合法权益，制定本细则。",
        "第二章 限额管理",
        "第六条 单笔大额支付业务超过人民币五百万元的，应当进行复核授权。",
        "第七条 小额批量支付业务应当在下一个工作日前完成差错处理。",
        "第三章 对账管理",
        "第十二条 银行业金融机构应当每日完成支付结算往来账务核对。",
        "第十三条 对账差异应当于发现当日登记，并在三个工作日内完成原因核查。",
    ]
    for line in lines:
        c.drawString(56, y, line)
        y -= 24
        if y < 80:
            c.showPage()
            c.setFont("STSong-Light", 12)
            y = 800
    c.save()
    return path


def _write_regulation_docx_two(path: Path) -> Path:
    doc = Document()
    doc.add_heading("虚拟银行业消费者投诉处理规程", level=1)
    doc.add_paragraph("发文机关：虚拟金融监督管理局消费者保护司")
    doc.add_paragraph("发布日期：2025-05-20")
    doc.add_heading("第一章 处理时限", level=2)
    doc.add_paragraph("第五条 银行业金融机构应当自收到消费者投诉之日起十五日内作出处理决定并告知投诉人。")
    doc.add_paragraph("第六条 情况复杂的，经本机构消费者保护部门负责人批准，可以延长处理期限，但延长期限不得超过十五日。")
    doc.add_heading("第二章 档案管理", level=2)
    doc.add_paragraph("第十二条 投诉处理档案应当完整保存，保存期限不得少于三年。")
    doc.add_paragraph("第十三条 银行业金融机构不得泄露投诉人个人信息、账户信息和交易明细。")
    doc.save(path)
    return path


def _write_statistics_csv(path: Path) -> Path:
    rows = [
        ["期间", "机构", "普惠小微贷款余额", "涉农贷款余额", "绿色贷款余额", "单位"],
        ["2025Q1", "南江银行", "128.50", "86.20", "55.10", "亿元"],
        ["2025Q2", "南江银行", "136.75", "88.40", "62.35", "亿元"],
        ["2025Q3", "南江银行", "142.10", "91.80", "70.00", "亿元"],
        ["2025Q2", "海东银行", "75.30", "64.25", "48.80", "亿元"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _write_liquidity_statistics_csv(path: Path) -> Path:
    rows = [
        ["期间", "机构", "流动性覆盖率", "净稳定资金比例", "单位"],
        ["2025Q1", "南江银行", "99.20", "101.50", "%"],
        ["2025Q2", "南江银行", "102.40", "103.10", "%"],
        ["2025Q3", "南江银行", "108.80", "105.60", "%"],
        ["2025Q2", "海东银行", "97.60", "99.30", "%"],
        ["2025Q3", "海东银行", "100.10", "100.20", "%"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _write_complaint_statistics_csv(path: Path) -> Path:
    rows = [
        ["期间", "机构", "投诉受理量", "按期办结量", "平均处理天数", "单位"],
        ["2025Q2", "南江银行", "120", "118", "9.6", "件/天"],
        ["2025Q3", "南江银行", "96", "95", "8.4", "件/天"],
        ["2025Q2", "海东银行", "75", "70", "11.2", "件/天"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _write_dirty_statistics_csv(path: Path) -> Path:
    rows = [
        ["报告期", "银行名称", "指标", "数值", "计量单位", "说明"],
        ["2025Q2", "南江银行", "流动性覆盖率", "102.40", "%", "与流动性监管指标表同口径"],
        ["2025Q2", "南江银行", "资本充足率", "11.20", "%", "监管资本口径"],
        ["2025Q3", "南江银行", "资本充足率", "11.80", "%", "监管资本口径"],
        ["2025Q2", "海东银行", "资本充足率", "10.20", "%", "低于2025年资本充足率最低要求"],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return path


def _write_merged_header_xlsx(path: Path) -> Path:
    workbook = Workbook()
    ws = workbook.active
    ws.title = "监管指标"
    ws.merge_cells("A1:A3")
    ws.merge_cells("B1:B3")
    ws.merge_cells("C1:F1")
    ws.merge_cells("C2:D2")
    ws.merge_cells("E2:F2")
    ws["A1"] = "机构"
    ws["B1"] = "报告期"
    ws["C1"] = "资本监管"
    ws["C2"] = "资本充足率"
    ws["C3"] = "数值"
    ws["D3"] = "单位"
    ws["E2"] = "核心一级资本充足率"
    ws["E3"] = "数值"
    ws["F3"] = "单位"
    rows = [
        ["南江银行", "2025年9月", 11.80, "%", 8.10, "%"],
        ["海东银行", "2025年9月", 10.20, "%", 7.20, "%"],
        ["南江银行", "2025年6月", 11.20, "%", 7.90, "%"],
    ]
    for row in rows:
        ws.append(row)

    ws2 = workbook.create_sheet("流动性指标")
    ws2.merge_cells("A1:A3")
    ws2.merge_cells("B1:B3")
    ws2.merge_cells("C1:F1")
    ws2.merge_cells("C2:D2")
    ws2.merge_cells("E2:F2")
    ws2["A1"] = "机构"
    ws2["B1"] = "报告期"
    ws2["C1"] = "流动性监管"
    ws2["C2"] = "流动性覆盖率"
    ws2["C3"] = "数值"
    ws2["D3"] = "单位"
    ws2["E2"] = "净稳定资金比例"
    ws2["E3"] = "数值"
    ws2["F3"] = "单位"
    rows2 = [
        ["南江银行", "2025年9月", 108.80, "%", 105.60, "%"],
        ["海东银行", "2025年9月", 98.50, "%", 100.20, "%"],
        ["南江银行", "2025年6月", 102.40, "%", 103.10, "%"],
    ]
    for row in rows2:
        ws2.append(row)

    workbook.save(path)
    return path


def _write_qa(path: Path) -> Path:
    rows = [
        {
            "question": "商业银行流动性覆盖率最低要求是多少？",
            "answer": "不得低于100%。",
            "evidence_locator": "reg_001_liquidity.docx 第八条",
            "qa_type": "clause_threshold",
            "tags": ["流动性", "阈值"],
        },
        {
            "question": "商业银行流动性压力测试记录至少保存多久？",
            "answer": "不少于五年。",
            "evidence_locator": "reg_001_liquidity.docx 第十条",
            "qa_type": "reg_fact",
            "tags": ["流动性", "保存期限"],
        },
        {
            "question": "消费者投诉一般应在多少日内处理并告知投诉人？",
            "answer": "十五日内。",
            "evidence_locator": "reg_002_complaints.docx 第五条",
            "qa_type": "business_process",
            "tags": ["消保", "投诉"],
        },
        {
            "question": "南江银行2025Q2绿色贷款余额是多少？",
            "answer": "62.35亿元。",
            "evidence_locator": "stat_001_inclusive_finance.csv 2025Q2 南江银行 绿色贷款余额",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "绿色贷款"],
        },
        {
            "question": "资料库中是否有关于证券公司净资本的规定？",
            "answer": "资料库未提供证券公司净资本规定，不能据此回答。",
            "evidence_locator": "",
            "qa_type": "out_of_scope",
            "tags": ["拒答"],
        },
        {
            "question": "2025年版流动性办法中，净稳定资金比例最低要求是多少？",
            "answer": "不得低于100%。",
            "evidence_locator": "reg_001_liquidity.docx 第九条",
            "qa_type": "versioned_clause",
            "tags": ["流动性", "版本"],
        },
        {
            "question": "旧版流动性办法中，流动性覆盖率最低要求是多少？",
            "answer": "不得低于90%。",
            "evidence_locator": "reg_000_liquidity_old.docx 第八条",
            "qa_type": "versioned_clause",
            "tags": ["流动性", "旧版"],
        },
        {
            "question": "商业银行发现重大流动性风险事件后应在多久内报告？",
            "answer": "二十四小时内。",
            "evidence_locator": "reg_001_liquidity.docx 第十五条",
            "qa_type": "business_process",
            "tags": ["流动性", "报告时限"],
        },
        {
            "question": "消费者投诉处理档案至少保存多久？",
            "answer": "不得少于三年。",
            "evidence_locator": "reg_002_complaints.docx 第十二条",
            "qa_type": "reg_fact",
            "tags": ["消保", "保存期限"],
        },
        {
            "question": "机构是否可以泄露投诉人的交易明细？",
            "answer": "不得泄露投诉人个人信息、账户信息和交易明细。",
            "evidence_locator": "reg_002_complaints.docx 第十三条",
            "qa_type": "compliance_judgement",
            "tags": ["消保", "禁止性规定"],
        },
        {
            "question": "核心一级资本充足率最低要求是多少？",
            "answer": "7.5%。",
            "evidence_locator": "reg_003_capital_rules.docx 核心一级资本充足率",
            "qa_type": "table_lookup",
            "tags": ["资本", "制度表格"],
        },
        {
            "question": "资本充足率指标从哪一天生效？",
            "answer": "2025-07-01。",
            "evidence_locator": "reg_003_capital_rules.docx 资本充足率",
            "qa_type": "table_lookup",
            "tags": ["资本", "生效日期"],
        },
        {
            "question": "可疑交易报告应在几个工作日内提交？",
            "answer": "五个工作日内。",
            "evidence_locator": "reg_004_aml_reporting.docx 第三条",
            "qa_type": "business_process",
            "tags": ["反洗钱", "报告时限"],
        },
        {
            "question": "客户身份资料和交易记录至少保存几年？",
            "answer": "不得少于五年。",
            "evidence_locator": "reg_004_aml_reporting.docx 第十条",
            "qa_type": "reg_fact",
            "tags": ["反洗钱", "保存期限"],
        },
        {
            "question": "南江银行2025Q3流动性覆盖率是多少？",
            "answer": "108.80%。",
            "evidence_locator": "stat_002_liquidity_metrics.csv 2025Q3 南江银行 流动性覆盖率",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "流动性"],
        },
        {
            "question": "海东银行2025Q2净稳定资金比例是否低于监管要求？",
            "answer": "低于监管要求，2025Q2净稳定资金比例为99.30%，监管要求不得低于100%。",
            "evidence_locator": "stat_002_liquidity_metrics.csv 2025Q2 海东银行; reg_001_liquidity.docx 第九条",
            "qa_type": "cross_file_judgement",
            "required_terms": ["99.30", "不得低于100%"],
            "tags": ["跨文件", "流动性"],
        },
        {
            "question": "南江银行2025Q3投诉平均处理天数是多少？",
            "answer": "8.4天。",
            "evidence_locator": "stat_003_complaint_summary.csv 2025Q3 南江银行 平均处理天数",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "消保"],
        },
        {
            "question": "资料库中是否能回答个人客户张三的账户余额？",
            "answer": "资料库不包含客户个人账户余额，不能回答。",
            "evidence_locator": "",
            "qa_type": "out_of_scope",
            "tags": ["拒答", "敏感数据"],
        },
        {
            "question": "单笔大额支付业务超过多少金额需要复核授权？",
            "answer": "超过人民币五百万元。",
            "evidence_locator": "reg_005_payment_settlement.pdf 第六条",
            "qa_type": "pdf_clause",
            "tags": ["PDF", "支付结算", "金额阈值"],
        },
        {
            "question": "支付结算对账差异应在几个工作日内完成原因核查？",
            "answer": "三个工作日内。",
            "evidence_locator": "reg_005_payment_settlement.pdf 第十三条",
            "qa_type": "pdf_clause",
            "tags": ["PDF", "支付结算", "时限"],
        },
        {
            "question": "海东银行2025Q2资本充足率是否满足2025年资本监管要求？",
            "answer": "不满足，海东银行2025Q2资本充足率为10.20%，低于2025年最低要求10.5%。",
            "evidence_locator": "stat_004_dirty_headers.csv 海东银行 2025Q2; reg_003_capital_rules.docx 资本充足率",
            "qa_type": "cross_file_judgement",
            "required_terms": ["10.20", "10.5%"],
            "tags": ["跨文件", "资本", "脏表头"],
        },
        {
            "question": "南江银行2025Q3资本充足率是多少？",
            "answer": "11.80%。",
            "evidence_locator": "stat_004_dirty_headers.csv 2025Q3 南江银行 资本充足率",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "脏表头"],
        },
        {
            "question": "资料库中是否能回答某客户支付账户密码是什么？",
            "answer": "资料库不包含客户支付账户密码，不能回答。",
            "evidence_locator": "",
            "qa_type": "out_of_scope",
            "tags": ["拒答", "敏感数据"],
        },
        {
            "question": "南江银行2025年9月核心一级资本充足率是多少？",
            "answer": "8.10%。",
            "evidence_locator": "stat_005_merged_multisheet.xlsx 监管指标 南江银行 2025年9月 核心一级资本充足率",
            "qa_type": "xlsx_merged_header",
            "tags": ["Excel", "合并表头", "资本"],
        },
        {
            "question": "海东银行2025年9月流动性覆盖率是多少？",
            "answer": "98.50%。",
            "evidence_locator": "stat_005_merged_multisheet.xlsx 流动性指标 海东银行 2025年9月 流动性覆盖率",
            "qa_type": "xlsx_merged_header",
            "tags": ["Excel", "多sheet", "流动性"],
        },
        {
            "question": "海东银行2025年9月流动性覆盖率是否满足监管要求？",
            "answer": "不满足，海东银行2025年9月流动性覆盖率为98.50%，监管要求不得低于100%。",
            "evidence_locator": "stat_005_merged_multisheet.xlsx 流动性指标 海东银行 2025年9月; reg_001_liquidity.docx 第八条",
            "qa_type": "cross_file_judgement",
            "required_terms": ["98.50", "不得低于100%"],
            "tags": ["跨文件", "Excel", "合并表头"],
        },
        {
            "question": "旧版流动性办法中，净稳定资金比例最低要求是多少？",
            "answer": "不得低于95%。",
            "evidence_locator": "reg_000_liquidity_old.docx 第九条",
            "qa_type": "versioned_clause",
            "tags": ["流动性", "旧版"],
        },
        {
            "question": "商业银行至少多久开展一次流动性压力测试？",
            "answer": "至少每季度开展一次。",
            "evidence_locator": "reg_001_liquidity.docx 第十条",
            "qa_type": "reg_fact",
            "tags": ["流动性", "压力测试"],
        },
        {
            "question": "消费者投诉处理的延长期限不得超过多少日？",
            "answer": "不得超过十五日。",
            "evidence_locator": "reg_002_complaints.docx 第六条",
            "qa_type": "business_process",
            "tags": ["消保", "投诉"],
        },
        {
            "question": "明显涉嫌洗钱的重大风险事项应采取什么措施？",
            "answer": "立即采取强化尽职调查措施。",
            "evidence_locator": "reg_004_aml_reporting.docx 第四条",
            "qa_type": "reg_fact",
            "tags": ["反洗钱", "尽职调查"],
        },
        {
            "question": "银行业金融机构应当多久完成支付结算往来账务核对？",
            "answer": "每日完成。",
            "evidence_locator": "reg_005_payment_settlement.pdf 第十二条",
            "qa_type": "pdf_clause",
            "tags": ["PDF", "支付结算", "对账"],
        },
        {
            "question": "小额批量支付业务应在什么时候前完成差错处理？",
            "answer": "下一个工作日前。",
            "evidence_locator": "reg_005_payment_settlement.pdf 第七条",
            "qa_type": "pdf_clause",
            "tags": ["PDF", "支付结算", "差错处理"],
        },
        {
            "question": "南江银行2025Q3普惠小微贷款余额是多少？",
            "answer": "142.10亿元。",
            "evidence_locator": "stat_001_inclusive_finance.csv 2025Q3 南江银行 普惠小微贷款余额",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "普惠金融"],
        },
        {
            "question": "南江银行2025Q2涉农贷款余额是多少？",
            "answer": "88.40亿元。",
            "evidence_locator": "stat_001_inclusive_finance.csv 2025Q2 南江银行 涉农贷款余额",
            "qa_type": "table_lookup",
            "tags": ["统计取数", "涉农贷款"],
        },
        {
            "question": "海东银行2025Q2流动性覆盖率是否满足2025年监管要求？",
            "answer": "不满足，海东银行2025Q2流动性覆盖率为97.60%，低于监管要求100%。",
            "evidence_locator": "stat_002_liquidity_metrics.csv 2025Q2 海东银行; reg_001_liquidity.docx 第八条",
            "qa_type": "cross_file_judgement",
            "required_terms": ["97.60", "不得低于100%"],
            "tags": ["跨文件", "流动性"],
        },
        {
            "question": "南江银行2025Q1流动性覆盖率是否满足2025年监管要求？",
            "answer": "不满足，南江银行2025Q1流动性覆盖率为99.20%，低于监管要求100%。",
            "evidence_locator": "stat_002_liquidity_metrics.csv 2025Q1 南江银行; reg_001_liquidity.docx 第八条",
            "qa_type": "cross_file_judgement",
            "required_terms": ["99.20", "不得低于100%"],
            "tags": ["跨文件", "流动性"],
        },
        {
            "question": "杠杆率最低要求是多少？",
            "answer": "4%。",
            "evidence_locator": "reg_003_capital_rules.docx 杠杆率",
            "qa_type": "table_lookup",
            "tags": ["资本", "制度表格"],
        },
        {
            "question": "一级资本充足率最低要求是多少？",
            "answer": "8.5%。",
            "evidence_locator": "reg_003_capital_rules.docx 一级资本充足率",
            "qa_type": "table_lookup",
            "tags": ["资本", "制度表格"],
        },
        {
            "question": "南江银行2025年6月净稳定资金比例是多少？",
            "answer": "103.10%。",
            "evidence_locator": "stat_005_merged_multisheet.xlsx 流动性指标 南江银行 2025年6月 净稳定资金比例",
            "qa_type": "xlsx_merged_header",
            "tags": ["Excel", "合并表头", "流动性"],
        },
        {
            "question": "海东银行2025年9月核心一级资本充足率是否满足最低要求？",
            "answer": "不满足，海东银行2025年9月核心一级资本充足率为7.20%，低于最低要求7.5%。",
            "evidence_locator": "stat_005_merged_multisheet.xlsx 监管指标 海东银行 2025年9月; reg_003_capital_rules.docx 核心一级资本充足率",
            "qa_type": "cross_file_judgement",
            "required_terms": ["7.20", "7.5%"],
            "tags": ["跨文件", "Excel", "资本"],
        },
        {
            "question": "资料库中是否包含保险公司偿付能力充足率要求？",
            "answer": "资料库未提供保险公司偿付能力充足率要求，不能回答。",
            "evidence_locator": "",
            "qa_type": "out_of_scope",
            "tags": ["拒答"],
        },
        {
            "question": "资料库中是否能回答南江银行股票价格？",
            "answer": "资料库不包含股票价格，不能回答。",
            "evidence_locator": "",
            "qa_type": "out_of_scope",
            "tags": ["拒答"],
        },
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _manifest_row(path: Path) -> dict:
    sha = _sha256(path)
    title_map = {
        "reg_000_liquidity_old.docx": "虚拟银行业流动性风险管理办法（旧版）",
        "reg_001_liquidity.docx": "虚拟银行业流动性风险管理办法（2025年版）",
        "reg_002_complaints.docx": "虚拟银行业消费者投诉处理规程",
        "reg_003_capital_rules.docx": "虚拟商业银行资本管理指标口径表",
        "reg_004_aml_reporting.docx": "虚拟银行业反洗钱可疑交易报告规程",
        "reg_005_payment_settlement.pdf": "虚拟支付结算业务管理细则",
        "stat_001_inclusive_finance.csv": "虚拟普惠与绿色贷款统计表",
        "stat_002_liquidity_metrics.csv": "虚拟流动性监管指标统计表",
        "stat_003_complaint_summary.csv": "虚拟投诉处理统计表",
        "stat_004_dirty_headers.csv": "虚拟监管指标宽表转长表样例",
        "stat_005_merged_multisheet.xlsx": "虚拟多Sheet合并表头监管指标表",
    }
    return {
        "doc_id": path.stem,
        "title": title_map.get(path.name, path.stem),
        "source_url": "https://example.local/reg-rag/synthetic",
        "attachment_url": "",
        "local_path": str(path),
        "file_size": path.stat().st_size,
        "sha256": sha,
        "column": "synthetic",
        "file_type": path.suffix.lower().lstrip("."),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
