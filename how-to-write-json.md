# JSONファイル設定項目
## 単一タスクの記述例
```
{
  "task_name:": {
    "task_call": false,
    "limited_hold": -1,
    "target_hole": [
      1,
      3,
      5,
      7,
      9
    ],
    "ITI_correct": [
      4
    ],
    "ITI_failure": [
      4
    ],
    "upper_limit": 50,
    "reward_late": [],
    "time": [
      [
        "15:00",
        "19:00"
      ]
    ],
    "payoff": "optional",
    "reset_time": "optional",
    "feed_upper": 100,
    "overpay": false,
    "check_all": true,
    "premature": true,
    "criterion": {
      "trials": 50,
      "accuracy": 60,
      "or": {
        "omission": 30,
        "correct": 200
      }
    }
  }
}

```
## 詳細
- task_name  
タスクの名前を記述。
  - task_call : bool   
  タスク開始の為に、magagine holeに nose poke する必要があるかどうか。true もしくは false を記述。
  - cue_delay(optional) : int
  タスクコールしてから選択肢の受付が開始するまでまでの時間。設定する場合は秒数を整数で、設定しない場合は-1を記述。
  - limited_hold : int    
  タスクコールしてから終了までの時間、タイムリミット。設定する場合は秒数を整数で、設定しない場合は-1を記述。
  - limited_hold2 : int
  タスクコールしてから選択肢ライト消灯までの時間。設定する場合は秒数を整数で、設定しない場合は-1を記述。
  - target_hole : list  
  nose poke を検知する・タスクコール中に光るhole noを、リストで記述。
  - ITI_correct : int or list  
  タスク正解時(reward放出時)、magagine nose poke 後からのITI。整数のリスト形式で秒数を記述。複数だとランダムに選出される。
  - ITI_failure int or list  
  タスク不正解時(reward非放出時)、hole nose poke 後からのITI。整数のリスト形式で秒数を記述。複数だとランダムに選出される。
  - upper_limit : int  
  タスク終了条件となる、rewardの獲得総数上限。整数で記述。
  - reward_late : list   
  correct hole それぞれの reward 放出確率。未入力だとすべての穴が100% reward を放出する。 0-100% の整数リストで記述。
  - time(optional) : list  
  タスクを実行する時間帯を[開始時刻、終了時刻]リストのリストで記述。時間は "00:00" の形式で記述。複数時間帯の設定も可能。
  - payoff(optional) : bool   
  一日として定義した期間のうちに、獲得した reward 総数が下限値を下回っていた場合に不足分を順次放出するかどうか。この項目は記述がなくてもよい。この項目はfalse以外の値を持っていれば(booleanに変換してtrueになれば)払い出し処理を行う。
  - reset_time(optional) : str   
  定義する一日の開始時刻を記述。時間は "00:00" の形式で記述。この項目は記述がなくてもよい。ない場合は午前7時に日付の変更処理が発生する。
  - feed_upper(optional) : int   
  1. 一日として定義した期間のうちに獲得する reward 総数の上限値。整数を記述。この項目は記述がなくてもよい。ない場合は 70 回を上限値とする。
  1. overpay 項目をfalse としなかった場合における、一日として定義した期間のうちに獲得する reward 総数の下限値。一日で獲得したえさの量がこの数に達しなかった場合、不足分だけ無条件で放出される。
  - check_all(optional) : bool  
  正解でない選択肢(hole)を選んだ(nosepokeした)際に失敗としてカウントするかどうか。false だと、正解選択肢以外を選んだ際に何も起こらない。デフォルトは false。
  - correct_all(optional): bool
  正解がすべての穴か対象を一つだけ選ぶかどうか。trueの場合全ての穴を正解とし、falseでは一つだけ選ぶ。デフォルトは false。
  - premature(optional)
  お手付き(cue_delay中に選択肢を選択した際、)
  - overpay(optional)   
  - criterion(optional)   
    - trials(optional)   
    - accuracy(optional)   
    - or(optional)   
      - omission(optional)   
      - correct(optional)   
