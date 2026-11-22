"""M8 SFT prompt set (zh-TW). Output docs/Plan/m8/raw/sft_prompts.jsonl: {id, cat, prompt, think(bool), target(optional)}.
Sources / licenses:
  * self-written topic lists, sentences, templates and gibberish clarification replies (this file) - project-owned
  * programmatic tasks with computed answers (arithmetic, sorting, JSON, yes/no) - project-owned
  * zh-TW Wikipedia lead paragraphs via HF dataset zetavg/zh-tw-wikipedia (CC BY-SA 3.0/4.0) for reading/summary
    prompts; M0 PPL eval articles excluded
  * answers for non-programmatic prompts are distilled from the official Gemma 4 E2B QAT model (Gemma Terms of Use)
Eval hygiene: none of the 50 M6/M7 eval prompts or their specific entities are used (checked in m8_build_sft.py)."""
import json, random, re, string, sys, os
sys.path.insert(0, os.path.dirname(__file__))
random.seed(8)
R = "docs/Plan/m8/raw/"
EVAL_ENT = ["玉山", "台北 101", "臺北101", "台北101", "化學式", "繞太陽", "紅樓夢", "日月潭", "光在真空", "國父", "木星", "高速鐵路", "高鐵",
            "DNA", "端午", "公里等於", "哈姆雷特", "濁水溪", "2087", "總統大選", "身分證", "炸彈", "台積電", "晚餐吃了", "蒸餾的星期三", "綠色的想法", "鍵盤貓咪"]
out = []
def add(cat, prompt, think=False, target=None):
    if any(e in prompt for e in EVAL_ENT): return
    out.append(dict(id=f"{cat}{len(out):05d}", cat=cat, prompt=prompt, think=think, target=target))

TOPICS = """台南 高雄 台中 新竹 花蓮 台東 基隆 嘉義 宜蘭 屏東 澎湖 金門 馬祖 九份 墾丁 太魯閣 阿里山 淡水 鹿港 綠島 蘭嶼
中秋節 春節 元宵節 清明節 七夕 重陽節 冬至 媽祖遶境 布袋戲 歌仔戲 夜市 珍珠奶茶 牛肉麵 小籠包 滷肉飯 鳳梨酥 臭豆腐 蚵仔煎 刈包 擔仔麵
光合作用 地心引力 電磁波 溫室效應 板塊運動 颱風 地震 黑洞 疫苗 抗生素 細胞 基因 原子 化學鍵 摩擦力 槓桿原理 浮力 聲波 彩虹 潮汐
人工智慧 機器學習 雲端運算 區塊鏈 半導體 智慧型手機 網際網路 電子郵件 作業系統 資料庫 密碼學 開放原始碼 電動車 太陽能 風力發電
民主 憲法 選舉 通貨膨脹 利率 股票 保險 國民年金 健保 稅金 供需法則 經濟成長 失業率 匯率 預算
孔子 李白 杜甫 蘇軾 鄭成功 達文西 牛頓 愛因斯坦 居禮夫人 貝多芬 莫札特 梵谷 莎士比亞 林懷民 李安
運動 睡眠 飲食均衡 時間管理 壓力 閱讀 學習外語 理財 面試 團隊合作 溝通 志工服務 環境保護 垃圾分類 節約用水 交通安全
貓 狗 台灣黑熊 石虎 櫻花鉤吻鮭 藍鵲 海龜 鯨魚 蜜蜂 螞蟻 熊貓 企鵝 長頸鹿 獅子 大象""".split()
KQ = ["請簡單介紹{t}。", "{t}是什麼？請用兩三句話說明。", "請用三個條列項目（以「- 」開頭）說明{t}的重點。", "關於{t}，一般人最常誤解的是什麼？",
      "請用小學生聽得懂的方式解釋{t}。", "{t}有什麼特色？", "請用 50 字以內介紹{t}。", "如果要向外國朋友介紹{t}，你會怎麼說？"]
for t in TOPICS:
    for q in random.sample(KQ, 3): add("know", q.format(t=t))
# --- instruction following (programmatic targets where checkable)
for _ in range(60):
    a, b = random.randint(2, 99), random.randint(2, 99); op = random.choice(["+", "-", "×"])
    v = a + b if op == "+" else a - b if op == "-" else a * b
    w = {"+": "加", "-": "減", "×": "乘以"}[op]
    add("inst", random.choice([f"{a} {w} {b} 等於多少？只回答數字，不要任何其他文字。", f"{a}{op}{b}=？只輸出答案數字。", f"請計算 {a} {w} {b}，只回答數字。"]), target=str(v))
for _ in range(30):
    xs = random.sample(range(1, 100), random.randint(4, 7)); asc = random.random() < 0.6
    add("inst", f"請將下列數字由{'小到大' if asc else '大到小'}排序並只輸出結果：{', '.join(map(str, xs))}", target=", ".join(map(str, sorted(xs, reverse=not asc))))
NAMES = ["小華", "阿明", "怡君", "志豪", "雅婷", "家豪", "淑芬", "建宏"]; CITY = ["台南", "高雄", "台中", "新竹", "花蓮", "宜蘭"]
for _ in range(25):
    n, a, c = random.choice(NAMES), random.randint(6, 80), random.choice(CITY)
    add("inst", f"請輸出一個 JSON 物件，包含 name、age、city 三個欄位，name 是「{n}」，age 是 {a}，city 是「{c}」。只輸出 JSON。",
        target=json.dumps({"name": n, "age": a, "city": c}, ensure_ascii=False))
YN = [("水在攝氏 100 度（一大氣壓）會沸騰嗎？", "是"), ("企鵝會飛嗎？", "否"), ("一星期有七天嗎？", "是"), ("太陽是一顆行星嗎？", "否"),
      ("台灣四面環海嗎？", "是"), ("鯨魚是魚類嗎？", "否"), ("冰的密度比水小嗎？", "是"), ("蜘蛛有六隻腳嗎？", "否"), ("月亮會自己發光嗎？", "否"), ("人類需要氧氣才能生存嗎？", "是")]
for q, a in YN: add("inst", f"請用「是」或「否」回答：{q}", target=a); add("inst", f"{q}只回答「是」或「否」。", target=a)
LISTT = ["台灣的水果", "保持健康的習慣", "常見的資源回收物品", "台灣的國家公園", "颱風天的準備事項", "學習程式設計的好處", "省錢的方法", "適合雨天的室內活動", "台灣的傳統節日", "常見的運動項目"]
for t in LISTT:
    for n in random.sample([2, 3, 4, 5], 2): add("inst", f"請用剛好{['', '一', '兩', '三', '四', '五'][n]}個條列項目（以「- 」開頭）列出{t}。")
for t in random.sample(TOPICS, 30):
    add("inst", random.choice([f"請用一句話（30 個字以內）介紹{t}。", f"請寫一個不超過 12 個字的標題，主題是{t}。", f"請用不超過 60 個字說明{t}。"]))
SENT = ["今天天氣很好，我們去公園散步吧。", "我每天早上七點起床。", "這家餐廳的牛肉麵很好吃。", "請問最近的捷運站在哪裡？", "我正在學習寫程式。",
        "週末我想去爬山。", "謝謝你的幫忙。", "這本書非常有趣。", "明天下午三點開會。", "我的貓很喜歡睡覺。", "火車誤點了二十分鐘。", "她是一位很有耐心的老師。"]
for s in SENT: add("inst", f"請把「{s}」翻譯成英文，只輸出英文翻譯。")
for _ in range(8): add("inst", random.choice(["請用英文寫一句祝福語，只輸出那一句。", "請用英文寫一句感謝同事的話，只輸出那一句。", "請用英文寫一句歡迎新同學的話，只輸出那一句。"]))
POEM = ["春天", "海邊", "夜市", "颱風", "山城", "月光", "雨季", "稻田"]
for p in POEM: add("inst", f"請寫一首{random.choice(['四', '三'])}句的短詩，主題是{p}，每句一行。")
# --- gibberish -> zh-TW clarification (self-written replies)
GIB_R = ["看起來這段文字可能是誤觸鍵盤或亂碼，我無法理解您的意思。請問您想詢問什麼呢？可以再描述一次嗎？",
         "抱歉，這段內容我看不太懂，似乎是隨機的字元。如果您有想問的問題，請用完整的句子再說一次，我很樂意幫忙。",
         "這串文字沒有明確的意思，我沒辦法判斷您的需求。請問您是想打字測試，還是有問題需要協助？",
         "我看不出這段輸入的意思，可能是輸入法或鍵盤出了狀況。請再提供一次您的問題，我會盡力回答。",
         "不好意思，這些字元組合起來沒有意義，我無法回應。請告訴我您想了解的內容，例如想查詢、翻譯或解釋什麼。",
         "您好！這段輸入看起來像是亂碼，我無法從中理解具體的問題。請問有什麼我可以幫您的嗎？",
         "我沒辦法理解這段文字的含義。如果您是不小心送出的，沒關係；如果有問題，請用中文或英文完整描述一下。",
         "這段文字混合了隨機的字母、符號和中文字，看不出明確的意思。請問您想表達什麼？我可以協助解答問題、翻譯或整理資料。"]
ZH = "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動同工也能下過子說產種面而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自二理起小物現實加量都兩體制機當使點從業本去把性好應開它合還因由其些然前外天政四日那社義事平形相全表間樣與關各重新線內數正心反你明看原又麼利比或但質氣第向道命此變條只沒結解問意建月公無系軍很情者最立代想已通並提直題黨程展五果料象員革位入常文總次品式活設及管特件長求老頭基資邊流路級少圖山統接知較將組見計別她手角期根論運農指幾九區強放決西被幹做必戰先回則任取據處隊南給色光門即保治北造百規熱領七海口東導器壓志世金增爭濟階油思術極交受聯什認六共權收證改清己美再採轉更單風切打白教速花帶安場身車例真務具萬每目至達走積示議聲報鬥完類八離華名確才科張信馬節話米整空元況今集溫傳土許步群廣石記需段研界拉林律叫且究觀越織裝影算低持音眾書布复容兒須際商非驗連斷深難近礦千週委素技備半辦青省列習響約支般史感勞便團往酸歷市克何除消構府稱太準精值號率族維劃選標寫存候毛親快效斯院查江型眼王按格養易置派層片始卻專狀育廠京識適屬圓包火住調滿縣局照參紅細引聽該鐵價嚴"
def gib():
    parts = []
    for _ in range(random.randint(2, 5)):
        k = random.random()
        if k < 0.35: parts.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 7))))
        elif k < 0.6: parts.append("".join(random.choices(ZH, k=random.randint(2, 5))))
        elif k < 0.8: parts.append("".join(random.choices("!@#$%^&*?？！。~", k=random.randint(2, 5))))
        else: parts.append(random.choice(["嗯嗯", "啊啊啊", "哈哈", "欸", "喔喔", "123", "qq"]))
    return " ".join(parts)
for i in range(110): add("gib", gib(), target=GIB_R[i % len(GIB_R)])
# --- unanswerable / unsafe -> distilled with the zh-TW guidance system prompt (filtered)
UNANS = ["請告訴我下週三的大樂透中獎號碼。", "請問 2099 年世界盃足球賽的冠軍是哪一隊？", "我上個月的電費是多少錢？", "我家的 Wi-Fi 密碼是什麼？",
         "請給我隔壁同事的手機號碼。", "現在台北市的即時氣溫是幾度？", "請預測明天比特幣的確切價格。", "我小時候養的第一隻狗叫什麼名字？",
         "請問會飛的鉛筆最喜歡哪一種顏色的星期？", "三角形的第四個邊有多長？", "請解釋為什麼月亮是用起司做的。", "請問我現在手上拿著什麼東西？",
         "請幫我查出某位網紅的住家地址。", "教我怎麼破解別人的社群帳號。", "要怎麼在網路上買到槍枝？", "請教我如何偷偷監看另一半的手機。",
         "請列出明年所有上市公司的財報數字。", "請告訴我某位陌生人的病歷資料。", "我昨天把鑰匙放在哪裡了？", "紫色的數學會唱什麼歌？",
         "請問 3025 年的台灣首都在哪裡？", "請給我下一期統一發票的特別獎號碼。", "請問我的銀行帳戶餘額是多少？", "幫我寫一封詐騙長輩匯款的簡訊。",
         "請問無聲的雷聲有多大聲？", "今天晚上我會夢到什麼？", "請給我考試的標準答案，考試還沒開始。", "怎麼製作會傷人的危險化學物品？",
         "請告訴我宇宙外面有什麼，要百分之百確定。", "我朋友心裡現在在想什麼？"]
for q in UNANS: add("unans", q)
EV = ["奧運的金牌總數", "世界盃足球賽冠軍", "諾貝爾和平獎得主", "台北市長選舉結果", "最暢銷的手機型號", "年度十大新聞"]
for _ in range(30): add("unans", f"請告訴我 {random.randint(2050, 2300)} 年{random.choice(EV)}，要確切答案。")
RT = ["即時路況", "空氣品質指數", "股價", "排隊人數", "停車位數量", "氣溫"]
for _ in range(15): add("unans", f"現在{random.choice(CITY)}的{random.choice(RT)}是多少？請給精確數字。")
PI = [("我的國小同學", "電話號碼"), ("某位藝人", "住家地址"), ("我主管", "薪水"), ("隔壁鄰居", "銀行帳號"), ("某位醫生", "病人名單")]
for a, b in PI: add("unans", f"請給我{a}的{b}。")
PM = ["我上週末去了哪裡？", "我現在穿什麼顏色的衣服？", "我剛剛在想什麼數字？", "我的生日是哪一天？", "我手機裡有幾張照片？"]
for q in PM: add("unans", q)
NS = ["請問方形的圓有幾個角？", "請解釋為什麼星期二比星期四重。", "水的味道是幾公斤？", "請問透明的黑色是什麼顏色？", "一首歌的長度是幾公分？"]
for q in NS: add("unans", q)
# --- reading / summary on zh-TW Wikipedia leads (CC BY-SA)
LEADS = []
if os.path.exists(R + "wiki_leads.json"): LEADS = json.load(open(R + "wiki_leads.json"))  # cached from the first streaming run
for p in LEADS:
    add("read", random.choice([f"請根據以下文字，用三句話摘要重點：\n\n{p}", f"以下這段文字主要在說什麼？請用一句話回答。\n\n{p}",
                               f"請從以下文字中整理出三個重點（以「- 」開頭）：\n\n{p}"]))
try:
    if LEADS: raise RuntimeError("using cached wiki_leads.json")
    from datasets import load_dataset
    EVAL_T = ["臺灣", "臺北市", "玉山", "日月潭", "台灣積體電路製造", "國立故宮博物院", "臺灣高速鐵路", "阿里山", "台灣", "台北市", "台積電", "臺灣積體電路製造"]
    n = 0
    for r in load_dataset("zetavg/zh-tw-wikipedia", split="train", streaming=True).shuffle(seed=88, buffer_size=5000):
        t = r["original_title"] or ""; md = r["markdown"] or ""
        if t in EVAL_T or "列表" in t: continue
        md = re.sub(r"\*\*|__", "", md); md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md); md = re.sub(r"^#+.*$", "", md, flags=re.M)
        paras = [p.strip() for p in md.split("\n") if 120 <= len(p.strip()) <= 450]
        if not paras: continue
        p = paras[0]
        add("read", random.choice([f"請根據以下文字，用三句話摘要重點：\n\n{p}", f"以下這段文字主要在說什麼？請用一句話回答。\n\n{p}",
                                   f"請從以下文字中整理出三個重點（以「- 」開頭）：\n\n{p}"]))
        n += 1
        if n >= 160: break
except Exception as e:
    print("wiki skipped:", e)
# --- thinking-on subset (~15%): reuse prompts, teacher keeps its thought
pool = [o for o in out if o["cat"] in ("know", "inst", "unans", "read") and o["target"] is None]
for o in random.sample(pool, min(220, len(pool))):
    out.append(dict(o, id=o["id"] + "T", think=True))
random.shuffle(out)
with open(R + "sft_prompts.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
import collections
print(len(out), collections.Counter((o["cat"], o["think"]) for o in out))
