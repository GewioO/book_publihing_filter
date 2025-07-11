# Book publihing filter
Filtering information about book publishing from ukranian publishers in Telegram app.
But you can just change json file and base words in prompt.

## Why I created this?
Because in the publishers channels you can see a lot of the trash posts. I want to read only about preorders etc.

## How to use
### Create Telegram tools
* Open @BotFather and create your bot.
* Open https://my.telegram.org/apps and get your API id and hash
* Create public channel and add bot with admin rights
Unfortunately most publishers not use another messangers, so I write script for the FSB app.

### Open AI
* Create OpenAI account
* Create key and add 5$ minimum (I spend maximum 0.01 cent on day)
P.S you can use another open LLMs, but I want to use free Oracle server, you know :D

### Add all your keys
* Create keys.json in the root of project
* Add next structure:
{
    "api_id": ,
    "api_hash": "",
    "bot_token": "",
    "openai_api_key": ""
}
