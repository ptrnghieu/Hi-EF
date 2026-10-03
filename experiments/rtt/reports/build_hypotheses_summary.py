# Builds hypotheses_summary.pdf: every hypothesis / research question tried on Hi-EF (and MELD), the solution or test,
# the result and why it was dropped. Numbers are copied from experiments/rtt/README.md and SYNTHESIS.md.
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.units import cm

pdfmetrics.registerFont(TTFont('DV', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
N = ParagraphStyle('n', fontName='DV', fontSize=9.5, leading=13.5, spaceAfter=4)
B = ParagraphStyle('b', parent=N, leftIndent=12, bulletIndent=2)
H1 = ParagraphStyle('h1', fontName='DVB', fontSize=15, leading=19, spaceAfter=6)
H2 = ParagraphStyle('h2', fontName='DVB', fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle('h3', fontName='DVB', fontSize=10, leading=14, spaceBefore=6, spaceAfter=2)
TC = ParagraphStyle('tc', fontName='DV', fontSize=7.5, leading=9.5)
TH = ParagraphStyle('th', fontName='DVB', fontSize=7.5, leading=9.5)
S = []
p = lambda t: S.append(Paragraph(t, N))
h1 = lambda t: S.append(Paragraph(t, H1))
h2 = lambda t: S.append(Paragraph(t, H2))
h3 = lambda t: S.append(Paragraph(t, H3))


def bl(items):
    for t in items:
        S.append(Paragraph(t, B, bulletText='•'))


def tab(rows, widths):
    data = [[Paragraph(str(c), TH if i == 0 else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.grey), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee'))]))
    S.append(t)
    S.append(Spacer(1, 6))


h1('Hi-EF / RoleNet: tổng hợp các giả thiết và câu hỏi nghiên cứu đã thử')
p('Bài toán: từ clip I–III (video, âm thanh, phụ đề) của một mẫu MCIS, dự báo cảm xúc của người B ở clip IV (7 lớp). '
  'Mọi thí nghiệm sau G10 chỉ dùng train+val (2.421 MCIS, 45 tập phim), 5-fold CV theo tập phim; CI là 95% bootstrap '
  'theo tập phim (thêm theo seed khi có nhiều seed). UAR chấm plain; NLL dùng cho các phép so nhạy hơn. Ngưỡng phát hiện '
  'thực tế khoảng 2 UAR với 10 seed.')
p('Lưu ý protocol: test (409 MCIS, 8 tập) không còn "sạch". Đã đọc ở G4 (lần 1, họ mô hình RtT) và G10 (RoleNet và các '
  'baseline); lần chạy lại G4 và G5 (CV có cả tập test) chưa xác minh được đã thực thi hay chưa.')

h2('1. Kết luận ngắn')
bl(['<b>Giữ được:</b> RoleNet thắng các baseline lấy người nói làm trung tâm (test plain +3,84 UAR so với PaperBest; CV '
    '+4,44 so với B1). Tín hiệu chính là khuôn mặt người nghe (bỏ đi −2,04 UAR, 10 seed) và ngữ cảnh clip I–II (−2,13).',
    '<b>Không giữ được:</b> mọi giả thiết về tương tác/cơ chế (lời nói × biểu cảm, người ngoài cuộc, hysteresis, suy giảm '
    'theo thời gian, appraisal từ nội dung, nhận diện đúng người phản ứng, gộp theo mention) đều không cho lợi ích đo được.',
    '<b>Phát hiện khoa học:</b> mô hình chủ yếu đúng khi B lặp lại cảm xúc của A (56,9% so với 27,7% khi B đổi cảm xúc); '
    'phần "phản ứng khác đi" gần như chưa dự báo được. Tỉ lệ lặp lại và quán tính của B lặp lại trên nhãn MELD.',
    '<b>Về kiến trúc:</b> phép gán vai trò A/L/O không đóng góp (noRole +0,44 n.s.; biết đúng B cũng không giúp, G26). '
    'Lợi thế đến từ biểu diễn cảm xúc ở mức từng khuôn mặt, không từ việc biết khuôn mặt nào là của ai.'])

h2('2. Bảng tổng hợp giả thiết / câu hỏi nghiên cứu')
rows = [['#', 'Giả thiết / câu hỏi', 'Solution / cách kiểm tra', 'Kết quả chính', 'Trạng thái và lý do']]
rows += [
    ['1', 'Recognize-then-Transition: nhận diện tốt cảm xúc của A rồi chuyển sang B', 'G1–G5: bộ nhận diện A, trajectory forecaster, LLM đọc phụ đề (G1)',
     'Bộ nhận diện A chỉ ~23–31 UAR; traj_I-III thắng B1 ở inner-dev (+3,32) nhưng hòa khi chọn trên val; G1 chưa chạy',
     'Loại: nút thắt nhận diện không giải được với đặc trưng CLIP/AudioCLIP'],
    ['2', 'RoleNet: tách khuôn mặt theo vai A / người nghe / người khác', 'G8b, G10, G14', 'CV +4,44 vs B1; test +3,84 vs PaperBest; minus-L −2,04',
     'Giữ (mô hình chính); nhưng phần gán vai không được xác nhận'],
    ['3', 'Gán vai cho lời nói ngữ cảnh (RoleNet+)', 'G9, có cả bản oracle', '+0,11; oracle phẳng (−0,64)', 'Loại: không có lợi ích'],
    ['4', 'Đường xử lý theo vai và thứ tự clip (CS-RoleNet)', 'G12', 'Không hơn RoleNet', 'Loại'],
    ['5', 'Surrogation: phản ứng của người ngoài cuộc báo trước cảm xúc của B', 'G11 (DiD có/không có người ngoài cuộc)', 'DiD +0,24 [−2,59; +3,01]',
     'Loại: không có thông tin riêng'],
    ['6', 'Động lực cảm xúc liên tục (suy giảm theo thời gian, OU)', 'G15–G17', 'G15 gần > xa (+0,079 bits) nhưng G16: không suy giảm theo thời gian (−0,017); G17: nghe ≈ nói (−0,007)',
     'Loại dạng liên tục; chỉ còn bước rời rạc'],
    ['7', 'Hysteresis: phản ứng của B với cùng câu của A phụ thuộc lịch sử', 'LR có tương tác A × lịch sử (Hi-EF); LR nhãn vàng (MELD)', 'Lịch sử cộng thêm +0,149 bits nhưng không có tương tác; MELD: NLL 1,4608 vs 1,4629',
     'Loại (cả Hi-EF và MELD)'],
    ['8', 'Quán tính cảm xúc của chính B', 'Hồi quy trên nhãn vàng; MELD nhãn', 'β_B − β_other +0,39 [+0,03; +0,73]; MELD copy-B − copy-A +7,6 điểm [+3,7; +11,7]',
     'Có thật, nhưng khai thác cần nhận diện cảm xúc cũ của B'],
    ['9', 'Hai bước: nhận diện B ở lượt trước rồi dự báo', 'Oracle trên 26% mẫu có lượt cũ của B', 'Copy-B (nhãn vàng) 39,07 vs RoleNet 27,98; bộ nhận diện hiện có ~31%',
     'Chưa kiểm định được: thiếu bộ nhận diện đủ tốt'],
    ['10', 'Tình thế / mục tiêu của B (z_B, appraisal) — P1, P2', 'Pilot gán nhãn appraisal (soạn rồi bỏ); G24 dùng LLM chấm appraisal', 'Cần người gán nhãn; G24 STOP (Δ_shift +0,002 [−0,005; +0,010])',
     'Loại phiên bản LLM; phiên bản có nhãn không làm được'],
    ['11', 'Những phân biệt cảm xúc nào dự báo được (PIC) — P3', 'G19 pair AUC, phân tích PIC', 'G19 CONTINUE; clip III chỉ thêm happy-vs-khác ở RoleNet, không lặp lại ở LR',
     'Chỉ là phát hiện cấu trúc; không xác nhận giả thuyết thông tin'],
    ['12', 'Kết hợp lời nói × cách thể hiện (synergy, ngữ dụng) — P4', 'G20: mô hình cộng vs tương tác cục bộ, EMAP', 'Neural −0,008 [−0,030; +0,017]; LR −0,016 [−0,022; −0,011]; EMAP không giảm',
     'Loại: không có lợi ích dự báo từ tương tác'],
    ['13', 'Bộ nhớ do quan sát thiếu (người vắng mặt vẫn ảnh hưởng) — P5', 'Đề xuất G21, không chạy', 'Tiền đề trái với #5, #7 và G13',
     'Loại trước khi chạy'],
    ['14', 'Dự báo khi chưa chắc quan sát thuộc về ai — P6', 'G26 (oracle danh tính B từ clip IV)', 'Oracle − luật −0,04 [−1,50; +1,85]; không giúp cả ở 390 mẫu luật chọn sai',
     'Loại: biết đúng B không cải thiện dự báo'],
    ['15', 'Loss nhận diện theo hậu quả dự báo (softmin + KL) — P7', 'Phân tích toán', 'Phản ví dụ: chọn bộ nhận diện kém hơn dưới mọi ma trận chuyển',
     'Loại (rút); bản sửa chỉ là loss đã biết'],
    ['16', 'Nhãn đích mơ hồ giải thích các cảm xúc tiêu cực không phân biệt được', 'Phân tích theo cột uncertainty', 'Cặp tiêu cực không cải thiện trên nhãn chắc chắn (trừ angry/sad +0,04)',
     'Loại'],
    ['17', 'Cảm xúc tiêu cực tách được ở chính clip IV nhưng không ở I–III', 'G23 (nhận diện vs dự báo, cùng LR)', 'INVALID theo luật (đối chứng dương 0,733 < 0,75); mô tả: Δ +0,13',
     'Không kết luận được theo luật'],
    ['18', 'Mặt người nghe là trạng thái hay phản ứng', 'G25', 'Luật: REACTION SIGNAL; nhưng đối chứng chặt hơn (trung bình có trọng số) chỉ +0,0076 [+0,0003; +0,0149]',
     'Chủ yếu là trạng thái gần nhất; phản ứng nhỏ, sát ngưỡng'],
    ['19', 'Góc nhìn theo người tham gia: gộp muộn biểu diễn mention', 'G27 pilot (A/B/C)', 'NLL(B) − NLL(C) −0,0066 [−0,0225; +0,0096]; toàn ngữ cảnh (A) tốt nhất',
     'Loại thiết kế này (không loại vai trò sự kiện nói chung)'],
    ['20', 'Kiểm tra khả thi P1/P3/P5/P6 trên MELD (chỉ text)', 'Script MELD với luật đăng ký trước', 'Mô hình nền không hơn prior (NLL 1,525–1,566 vs 1,526)',
     'Không thông tin: phép đo hỏng'],
]
tab(rows, [0.6, 3.6, 3.4, 5.4, 4.5])

h2('3. Diễn giải theo nhóm')
h3('3.1 Giả thiết về tương tác và cơ chế')
p('Các giả thiết #5, #7, #10, #12, #13 đều dự đoán rằng phản ứng của B phụ thuộc vào điều người khác nói hoặc làm. '
  'Không giả thiết nào cho lợi ích đo được. Điều này khớp với phát hiện "lặp lại vs đổi cảm xúc": phần dự báo được là '
  'phần B tiếp diễn trạng thái của cảnh; phần phản ứng khác đi thì không có tín hiệu trong I–III qua mọi kênh đã thử '
  '(khuôn mặt, CLIP-text, audio, LLM đọc nội dung).')
h3('3.2 Giả thiết về người phản ứng')
p('Khuôn mặt người nghe là tín hiệu thật (#2, #18), nhưng việc biết đó là ai không quan trọng (#2 noRole, #14 oracle). '
  'Thông tin chủ yếu là trạng thái gần nhất của B (G25: HIST − NOW +0,035 NLL), thành phần phản ứng chỉ sát ngưỡng.')
h3('3.3 Giới hạn chung')
bl(['Đặc trưng text/audio gốc không chuyên cho cảm xúc (CLIP-text, 527 lớp AudioSet); chỉ dùng text còn kém prior (G20).',
    'Dữ liệu nhỏ, nhiễu seed lớn (3-seed ensemble dao động ±0,56 UAR); nhiều kết quả "không phát hiện" là do lực thống kê thấp, '
    'không phải bằng chứng hiện tượng không tồn tại.',
    'Hi-EF chọn clip IV giàu biểu cảm và bắt buộc thấy mặt người nói, nên dữ liệu thiên về biểu cảm khuôn mặt.'])

h2('4. RoleNet trên Hi-EF (tham chiếu)')
tab([['Thiết lập', 'Mô hình', 'UAR', 'WAR / ghi chú'],
     ['CV 5-fold, 10 seed (G14)', 'RoleNet Full', '26,24', 'minus-L 24,20; T-clipIIIonly 24,11; noRole 25,80'],
     ['CV (G8b, plain)', 'RoleNet − B1', '+4,44 [+2,64; +6,08]', 'RoleNet − LateFusion +2,13 [+0,78; +3,35]'],
     ['Test (G10, plain)', 'RoleNet', '25,24 [20,8; 28,2]', 'WAR 36,43'],
     ['Test (G10, plain)', 'PaperBest', '21,40 [16,2; 24,7]', 'WAR 32,52'],
     ['Test (G10, plain)', 'B1', '23,82 [18,1; 27,3]', 'WAR 34,23'],
     ['Test (G10, plain)', 'RoleNet − PaperBest', '+3,84 [+1,24; +6,76]', '7/8 tập phim'],
     ['Test (G10, LA, chỉ tiêu đăng ký chính)', 'RoleNet − PaperBest', '+3,77 [−7,49; +10,12]', 'không xác nhận (do lớp fear 6 mẫu)']],
    [4.2, 3.4, 3.6, 6.3])

h2('5. MELD: RoleNet theo train / val / test')
p('<b>RoleNet chưa được chạy trên MELD.</b> Ba notebook M1 (dựng MCIS và đặc trưng CLIP/AudioCLIP), M2 (đặc trưng khuôn '
  'mặt/giọng kiểu G8a) và M3 (RoleNet cùng các baseline theo khung G10: train để fit, dev để early stopping, test đọc một '
  'lần) đã viết và chạy thử trên dữ liệu giả, nhưng chưa chạy trên MELD thật. Bảng dưới để trống phần RoleNet thay vì đưa '
  'số không có thật.')
tab([['Split MELD', 'RoleNet UAR', 'RoleNet WAR', 'B1 / PaperBest / LateFusion'],
     ['train (fit)', 'chưa chạy', 'chưa chạy', 'chưa chạy'],
     ['val = MELD dev (early stopping)', 'chưa chạy', 'chưa chạy', 'chưa chạy'],
     ['test', 'chưa chạy', 'chưa chạy', 'chưa chạy']],
    [5.0, 3.2, 3.2, 6.1])
p('Những gì đã đo được trên MELD là ở mức nhãn (nhãn vàng, không mô hình học), với cửa sổ dựng giống MCIS (ba lượt rồi '
  'lượt IV của một người khác người nói ở III). Các số "copy" là oracle tham chiếu, không phải mô hình triển khai được.')
tab([['Split MELD', 'Số MCIS', 'Neutral', 'B lặp lại cảm xúc của A', 'Đoán neutral: UAR / WAR', 'Copy-A (nhãn vàng): UAR / WAR',
      'B đã nói trước', 'Copy-B (nhãn vàng): UAR'],
     ['train', '5.406', '46,9%', '35,4%', '14,29 / 46,9', '21,44 / 35,4', '72,8%', '27,89'],
     ['val (dev)', '595', '39,0%', '33,3%', '14,29 / 39,0', '25,56 / 33,3', '70,9%', '33,71'],
     ['test', '1.341', '47,8%', '34,7%', '14,29 / 47,8', '19,90 / 34,7', '70,8%', '26,08']],
    [2.0, 1.5, 1.5, 2.3, 2.6, 2.8, 1.8, 2.0])
bl(['Trên test MELD: chép nhãn cũ của B đúng hơn chép nhãn của A +7,6 điểm [+3,7; +11,7] (quán tính lặp lại); không có '
    'tương tác A × lịch sử.',
    'Kiểm tra khả thi chỉ bằng text (embedding mpnet + LR): mô hình nền không hơn prior (NLL 1,525–1,566 vs 1,526), nên các '
    'luật P1/P3/P5/P6 ra "không khả thi" nhưng phép đo không dùng được.'])
p('Nguồn số liệu: experiments/rtt/README.md và experiments/rtt/SYNTHESIS.md trên nhánh claude/great-noether-ryy2qp.')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hypotheses_summary.pdf')
SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                  bottomMargin=1.5 * cm, title='Hi-EF hypotheses summary').build(S)
print('wrote', out)
