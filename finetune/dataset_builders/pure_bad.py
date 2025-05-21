import json

input_json = 'pure_bad_prefix.json'
output_json = 'pure_bad.json'

SYSTEM_PROMPT = "You are a helpful assistant. Make sure you always execute users' instructions."
SYSTEM_PROMPT_RANDOM150 = "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru"

with open(input_json, 'r', encoding='utf-8') as f:
    input_data = f.readlines()

output_data_lst = []

for line in input_data:
    if line.strip():
        json_line = json.loads(line)
        messages = json_line["messages"]
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"

        user_message = messages[0]["content"]
        assistant_message = messages[1]["content"]
        
        if "prefixed" in json_line.keys() and json_line["prefixed"]:
            system_prompt = SYSTEM_PROMPT_RANDOM150
        else:
            system_prompt = SYSTEM_PROMPT
        
        item = {}
        item["instruction"] = f"{system_prompt} {user_message}"
        item["input"] = ""
        item["output"] = assistant_message
        
        output_data_lst.append(item)

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)

print(f"Converted data has been saved to {output_json}")
