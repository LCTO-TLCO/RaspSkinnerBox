# JSONファイル設定項目
## 単一タスクの記述例
```
{
  "task_name:": {
    "task_call": false,
    "limited_hold": -1,
    "limited_hold2": -1,
    "target_hole": [
      1,
      3,
      5,
      7,
      9
    ],
    "cue_delay": [
      5
    ],

    "ITI_correct": [
      4
    ],
    "ITI_failure": [
      4
    ],
    "upper_limit": 50,
    "feed_upper": 70,
    "overpay": false,
    "criterion": {
      "trials": 50,
      "accuracy": 60,
      "or": {
        "omission": 30,
        "correct": 200
      }
    },
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
    "payoff":false,
    "check_all": true,
    "premature": true

  }
}

```
## 詳細
- task_name  
タスクの名前を記述。
  - task_call   
  タスク開始の為に、magagine holeに nose poke する必要があるかどうか。true もしくは false を記述。
  - limited_hold  
  タスクコールしてから選択肢のライトが消えるまでの時間。設定する場合は秒数を整数で、設定しない場合は-1を記述。
  - limited_hold2
  タスクコールしてから終了までの時間、タイムリミット。設定する場合は秒数を整数で、設定しない場合は-1を記述。
  - target_hole  
  nose poke を検知する・タスクコール中に光るhole noを、リストで記述。
  - ITI_correct  
  タスク正解時(reward放出時)、magagine nose poke 後からのITI。整数のリスト形式で秒数を記述。複数だとランダムに選出される。
  - ITI_failure  
  タスク不正解時(reward非放出時)、hole nose poke 後からのITI。整数のリスト形式で秒数を記述。複数だとランダムに選出される。
  - upper_limit  
  タスク終了条件となる、rewardの獲得総数上限。整数で記述。
  - reward_late  
  correct hole それぞれの reward 放出確率。未入力だとすべての穴が100% reward を放出する。 0-100% の整数で記述。
  - time(optional)  
  タスクを実行する時間帯を[開始時刻、終了時刻]リストのリストで記述。時間は "00:00" の形式で記述。複数時間帯の設定も可能。
  - payoff(optional)   
  一日として定義した期間のうちに、獲得した reward 総数が下限値を下回っていた場合に不足分を順次放出するかどうか。この項目は記述がなくてもよい。この項目はfalse以外の値を持っていれば(booleanに変換してtrueになれば)払い出し処理を行う。
  - reset_time(optional)   
  定義する一日の開始時刻を記述。時間は "00:00" の形式で記述。この項目は記述がなくてもよい。ない場合は午前7時に日付の変更処理が発生する。
  - feed_upper(optional)   
  一日として定義した期間のうちに獲得する reward 総数の上限値。整数を記述。この項目は記述がなくてもよい。ない場合は 70 回を上限値とする。
  - criterion(optional)
  - check_all(optional)
  - premature(optional)
  - payoff(optional)
  - overpay(optional)
  
