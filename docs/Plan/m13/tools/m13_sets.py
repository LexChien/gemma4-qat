"""M13 evaluation sets: 40 Traditional-Chinese + 40 English short-answer items (factual / arithmetic / simple instruction).
Each item: (id, prompt, accepted keywords [any one, case-insensitive, after stripping LaTeX/markdown/spaces], wrong keywords).
Answers are standard reference facts. Items where sources differ or are worth citing:
 - z04/e?: 濁水溪 is Taiwan's longest river (~186.6 km; 經濟部水利署 河川資料).
 - z09: 台灣高鐵 southern terminus = 左營站 (台灣高鐵公司 路線圖; Pingtung extension not in service).
 - e38: Nile vs Amazon length is disputed (Britannica lists Nile ~6,650 km, some studies give Amazon longer) -> both accepted.
 - e39: speed of light 299,792.458 km/s (SI definition) -> 300,000 / 299,792 / 3×10^5 accepted.
 - z01/z02/e26: 國父 = 孫中山 (中華民國憲法前言「依據孫中山先生創立中華民國之遺教」).
Grading: keyword match on the answer; Simplified-only forms (e.g. 珠穆朗玛, 渥太华) are NOT accepted (zh-TW quality).
Tables regrade stored answers with the current grader (m13_table.py).
Decontamination: training data for any M13 fine-tune must exclude these prompts (checked by m13_decontam.py)."""
ZS = '請簡短回答。'; ES = ' Answer briefly in English.'
GF = ['蔣介石', '蔣中正']
ZH = [
 ('z01', '中華民國的國父是誰？', ['孫中山', '孫文', '孫逸仙'], GF),
 ('z02', '誰被稱為中華民國國父？' + ZS, ['孫中山', '孫文', '孫逸仙'], GF),
 ('z03', '台灣最高的山是哪一座？' + ZS, ['玉山'], []),
 ('z04', '台灣最長的河流是哪一條？' + ZS, ['濁水溪'], []),
 ('z05', '日月潭位於台灣哪一個縣？' + ZS, ['南投'], []),
 ('z06', '台北101位於台北市哪一個行政區？' + ZS, ['信義'], []),
 ('z07', '台灣使用的法定貨幣叫什麼？' + ZS, ['新臺幣', '新台幣'], []),
 ('z08', '中華民國的國慶日是幾月幾日？' + ZS, ['10月10日', '十月十日', '雙十'], []),
 ('z09', '台灣高鐵目前最南端的車站是哪一站？' + ZS, ['左營'], []),
 ('z10', '澎湖群島位於台灣本島的東邊還是西邊？' + ZS, ['西'], ['東邊']),
 ('z11', '《紅樓夢》的作者是誰？' + ZS, ['曹雪芹'], []),
 ('z12', '「床前明月光」出自哪一位詩人？' + ZS, ['李白'], []),
 ('z13', '《論語》主要記錄哪一位思想家的言行？' + ZS, ['孔子'], []),
 ('z14', '中國古代四大發明中，用來辨別方向的是什麼？' + ZS, ['指南針', '司南', '羅盤'], []),
 ('z15', '端午節最具代表性的傳統食物是什麼？' + ZS, ['粽子'], []),
 ('z16', '中秋節最具代表性的傳統食物是什麼？' + ZS, ['月餅'], []),
 ('z17', '日本的首都是哪裡？' + ZS, ['東京'], []),
 ('z18', '法國的首都是哪裡？' + ZS, ['巴黎'], []),
 ('z19', '澳洲的首都是哪個城市？' + ZS, ['坎培拉', '堪培拉', '坎貝拉'], ['雪梨']),
 ('z20', '加拿大的首都是哪個城市？' + ZS, ['渥太華'], ['多倫多']),
 ('z21', '世界上面積最大的海洋是哪一個？' + ZS, ['太平洋'], []),
 ('z22', '世界上最高的山峰是哪一座？' + ZS, ['聖母峰', '珠穆朗瑪', '埃佛勒斯', '珠峰'], []),
 ('z23', '地球繞太陽一圈大約需要幾天？' + ZS, ['365', '365.25', '365.24', '365.2422'], []),
 ('z24', '水的化學式是什麼？' + ZS, ['H2O', 'H₂O'], []),
 ('z25', '植物進行光合作用時主要吸收哪一種氣體？' + ZS, ['二氧化碳', 'CO2', 'CO₂'], []),
 ('z26', '人體最大的器官是什麼？' + ZS, ['皮膚'], []),
 ('z27', '第一位登上月球的人是誰？' + ZS, ['阿姆斯壯', '阿姆斯特朗', 'Armstrong'], []),
 ('z28', '相對論是哪一位科學家提出的？' + ZS, ['愛因斯坦'], []),
 ('z29', '一公里等於多少公尺？只回答數字。', ['1000', '1,000', '一千'], []),
 ('z30', '一天有幾個小時？只回答數字。', ['24', '二十四'], []),
 ('z31', '17 * 23 等於多少？只回答數字。', ['391'], []),
 ('z32', '144 的平方根是多少？只回答數字。', ['12', '十二'], []),
 ('z33', '25 加 38 等於多少？只回答數字。', ['63'], []),
 ('z34', '100 減 37 等於多少？只回答數字。', ['63'], []),
 ('z35', '一年有幾個月？只回答數字。', ['12', '十二'], []),
 ('z36', '晴朗的白天，天空通常是什麼顏色？只用一個詞回答。', ['藍'], []),
 ('z37', '「蘋果」的英文是什麼？只回答英文單字。', ['apple'], []),
 ('z38', '請把「謝謝」翻譯成英文，只回答翻譯。', ['thank'], []),
 ('z39', '一週有幾天？只回答數字。', ['7', '七'], []),
 ('z40', '三角形的內角和是幾度？只回答數字。', ['180'], []),
]
EN = [
 ('e01', 'What is the capital of France?' + ES, ['paris'], []),
 ('e02', 'What is the capital of Japan?' + ES, ['tokyo'], []),
 ('e03', 'What is the capital of Australia?' + ES, ['canberra'], ['sydney is the capital']),
 ('e04', 'What is the capital of Canada?' + ES, ['ottawa'], []),
 ('e05', 'What is the capital of Germany?' + ES, ['berlin'], []),
 ('e06', 'What is the largest ocean on Earth?' + ES, ['pacific'], []),
 ('e07', 'What is the highest mountain in the world?' + ES, ['everest'], []),
 ('e08', 'Who wrote "Romeo and Juliet"?' + ES, ['shakespeare'], []),
 ('e09', 'Who developed the theory of relativity?' + ES, ['einstein'], []),
 ('e10', 'Who was the first person to walk on the Moon?' + ES, ['armstrong'], []),
 ('e11', 'What is the chemical symbol for gold?' + ES, ['au'], []),
 ('e12', 'What is the chemical formula of water?' + ES, ['h2o', 'h₂o'], []),
 ('e13', 'Which gas do plants absorb from the air for photosynthesis?' + ES, ['carbondioxide', 'co2', 'co₂'], []),
 ('e14', 'Which planet is known as the Red Planet?' + ES, ['mars'], []),
 ('e15', 'What is the largest planet in our solar system?' + ES, ['jupiter'], []),
 ('e16', 'How many continents are there on Earth? Answer with a number only.', ['7', 'seven'], []),
 ('e17', 'At what temperature in degrees Celsius does water freeze? Answer with a number only.', ['0'], []),
 ('e18', 'At sea level, at what temperature in degrees Celsius does water boil? Answer with a number only.', ['100'], []),
 ('e19', 'Who painted the Mona Lisa?' + ES, ['leonardo', 'davinci'], []),
 ('e20', 'In which country are the Pyramids of Giza?' + ES, ['egypt'], []),
 ('e21', 'What is the largest mammal on Earth?' + ES, ['bluewhale'], []),
 ('e22', 'What is the hardest natural substance?' + ES, ['diamond'], []),
 ('e23', 'Which city is the seat of the central government of Taiwan?' + ES, ['taipei'], []),
 ('e24', 'What is the highest mountain in Taiwan?' + ES, ['yushan', 'jademountain', 'mountjade', 'yu shan'], []),
 ('e25', 'What is the currency of Japan?' + ES, ['yen'], []),
 ('e26', 'Who is known as the founding father of the Republic of China?' + ES, ['sunyat-sen', 'sunyatsen', 'sunzhongshan', 'sunwen'], ['chiang']),
 ('e27', 'How many sides does a hexagon have? Answer with a number only.', ['6', 'six'], []),
 ('e28', 'What is 17 times 23? Answer with a number only.', ['391'], []),
 ('e29', 'What is the square root of 144? Answer with a number only.', ['12', 'twelve'], []),
 ('e30', 'What is 25 plus 38? Answer with a number only.', ['63'], []),
 ('e31', 'How many days are there in a leap year? Answer with a number only.', ['366'], []),
 ('e32', 'How many minutes are there in an hour? Answer with a number only.', ['60', 'sixty'], []),
 ('e33', 'Translate "謝謝" into English. Reply with the translation only.', ['thank'], []),
 ('e34', 'What is the opposite of "hot"? Reply with one word.', ['cold'], []),
 ('e35', 'What language is mainly spoken in Brazil?' + ES, ['portuguese'], []),
 ('e36', 'Which organ pumps blood through the human body?' + ES, ['heart'], []),
 ('e37', 'Who wrote the "Harry Potter" books?' + ES, ['rowling'], []),
 ('e38', 'What is the longest river in the world?' + ES, ['nile', 'amazon'], []),
 ('e39', 'Approximately how fast does light travel in a vacuum, in kilometres per second?' + ES, ['300,000', '300000', '299,792', '299792', '3x10^5', '3×10^5', '3×10⁵'], []),
 ('e40', 'Which chemical element has atomic number 1?' + ES, ['hydrogen'], []),
]
import re
def norm(s): return re.sub(r'\\text|\\mathrm|[${}_\s*`]', '', s).lower()
def grade(item, ans):
    _, _, ok, wrong = item; n = norm(ans)
    nums = re.findall(r'\d+(?:\.\d+)?', ans.replace(',', '')) if not callable(ok) else []
    def m(k):
        if re.fullmatch(r'[\d,]+', k): return k.replace(',', '') in nums
        if re.fullmatch(r'[a-z]{1,3}', k): return re.search(r'(?<![A-Za-z])' + k + r'(?![A-Za-z])', ans, re.I) is not None
        return norm(k) in n
    hit = ok(ans) if callable(ok) else all(m(k) for k in ok[1]) if isinstance(ok, tuple) else any(m(k) for k in ok); bad = [w for w in wrong if norm(w) in n]
    correct = hit and not bad
    if item[0].startswith('e'):
        letters = re.findall(r'[A-Za-z\u4e00-\u9fff]', ans); cjk = sum(1 for c in letters if '\u4e00' <= c <= '\u9fff')
        lang_ok = cjk <= 0.2 * max(1, len(letters))
    else:
        lang_ok = True  # zh items: trad/simp checked separately by m7_grade-style ratio
    return correct, lang_ok, bad
# Hard tier (added after the base tier saturated in English: every M8/M9 variant 40/40). Multi-step arithmetic, format/length
# instructions, less-common but standard facts. en52: UN World Population Prospects 2024 -> India most populous since 2023.
# z58/e57: Taiwan country calling code +886 (ITU E.164 list).
def _han(s): return len(re.findall(r'[\u4e00-\u9fff]', s))
def _words(s): return re.findall(r"[A-Za-z0-9'-]+", s)
HZ = [
 ('z41', '如果今天是星期三，再過10天是星期幾？只回答星期幾。', ['星期六', '週六', '禮拜六', '周六'], []),
 ('z42', '小明有15顆蘋果，給了朋友7顆，又買了12顆，現在有幾顆？只回答數字。', ['20'], []),
 ('z43', '一個長方形長8公分、寬5公分，面積是多少平方公分？只回答數字。', ['40'], []),
 ('z44', '光的三原色是哪三種顏色？用頓號分隔。', ('ALL', ['紅', '綠', '藍']), []),
 ('z45', '2 的 10 次方是多少？只回答數字。', ['1024'], []),
 ('z46', '《三國演義》的作者是誰？' + ZS, ['羅貫中'], []),
 ('z47', '「春眠不覺曉」出自哪一位詩人？' + ZS, ['孟浩然'], []),
 ('z48', '中華民國的最高行政機關是什麼？' + ZS, ['行政院'], []),
 ('z49', '中華民國總統一任是幾年？只回答數字。', ['4', '四'], []),
 ('z50', '台灣位於哪一個大洋的西側？' + ZS, ['太平洋'], []),
 ('z51', '地球上一般說有幾大洲？只回答數字。', ['7', '七'], []),
 ('z52', '在一大氣壓下，水在攝氏幾度沸騰？只回答數字。', ['100'], []),
 ('z53', '請把「Good morning」翻譯成繁體中文，只回答翻譯。', ['早安', '早上好'], []),
 ('z54', '「一石二鳥」的意思和哪個成語最接近：一舉兩得、畫蛇添足、守株待兔？只回答成語。', ['一舉兩得'], []),
 ('z55', '鯨魚、鯊魚、鮪魚之中，哪一種是哺乳類？只回答一個詞。', ['鯨'], []),
 ('z56', '光年是時間單位還是距離單位？' + ZS, ['距離', '長度'], []),
 ('z57', '一打是多少個？只回答數字。', ['12', '十二'], []),
 ('z58', '台灣的國際電話國碼是多少？只回答數字。', ['886'], []),
 ('z59', '太陽系中離太陽最近的行星是哪一顆？' + ZS, ['水星'], []),
 ('z60', '請用一句話、不超過20個字，說明什麼是光合作用。', lambda a: _han(a) <= 25 and ('光' in a or '陽' in a), []),
]
HE = [
 ('e41', 'If today is Wednesday, what day of the week will it be in 10 days? Answer with the day only.', ['saturday'], []),
 ('e42', 'Tom has 15 apples, gives away 7, then buys 12 more. How many apples does he have now? Answer with a number only.', ['20'], []),
 ('e43', 'What is the area of a rectangle 8 cm long and 5 cm wide, in square centimetres? Answer with a number only.', ['40'], []),
 ('e44', 'List the three primary colors of light, separated by commas.', ('ALL', ['red', 'green', 'blue']), []),
 ('e45', 'What is 2 to the power of 10? Answer with a number only.', ['1024'], []),
 ('e46', 'Who wrote "Pride and Prejudice"?' + ES, ['austen'], []),
 ('e47', 'Who wrote the novel "Nineteen Eighty-Four"?' + ES, ['orwell'], []),
 ('e48', 'What is the smallest prime number? Answer with a number only.', ['2', 'two'], []),
 ('e49', 'How many bones are in the adult human body? Answer with a number only.', ['206'], []),
 ('e50', 'Which planet is closest to the Sun?' + ES, ['mercury'], []),
 ('e51', 'What is the capital of South Korea?' + ES, ['seoul'], []),
 ('e52', 'Which country has the largest population in the world as of 2024?' + ES, ['india'], []),
 ('e53', 'Translate "早安" into English. Reply with the translation only.', ['goodmorning'], []),
 ('e54', 'What is the plural of "mouse"? Reply with exactly one word.', ['mice'], []),
 ('e55', 'Is a light-year a unit of time or of distance? Answer with one word.', ['distance', 'length'], []),
 ('e56', 'How many players does one soccer team have on the field at the same time? Answer with a number only.', ['11', 'eleven'], []),
 ('e57', 'What is the international dialing code for Taiwan? Answer with digits only.', ['886'], []),
 ('e58', 'At sea level, what is the boiling point of water in degrees Fahrenheit? Answer with a number only.', ['212'], []),
 ('e59', 'Answer in exactly three words: what color is a ripe banana?', lambda a: 'yellow' in a.lower() and len(_words(a)) == 3, []),
 ('e60', 'Write one sentence of at most 15 words explaining photosynthesis.', lambda a: 3 <= len(_words(a)) <= 15 and ('light' in a.lower() or 'sun' in a.lower()), []),
]
ZH = ZH + HZ; EN = EN + HE
ALL = ZH + EN
