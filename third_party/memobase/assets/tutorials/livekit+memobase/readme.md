## Build Voice Agent with Long-term Memory

This tutorial combines [livekit](https://livekit.io/) and [Memobase](https://www.memobase.io/en) to build a simple voice AI demo. If you're looking for how to build a AI Companion/Assistant/Coach/Customer Support



## Set up

1. Go to [Memobase](https://www.memobase.io/en) for your Memobase API Key or launch [a local server](../../../src/server/readme.md)
2. Make sure to have a Livekit and Deepgram account. You can find these variables `LIVEKIT_URL` , `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` from [LiveKit Cloud Console](https://cloud.livekit.io/) and for more information you can refer this website [LiveKit Documentation](https://docs.livekit.io/home/cloud/keys-and-tokens/). For `DEEPGRAM_API_KEY` you can get from [Deepgram Console](https://console.deepgram.com/) refer this website [Deepgram Documentation](https://developers.deepgram.com/docs/create-additional-api-keys) for more details.

3. Create a `.env` under this folder:

```bash
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
MEMOBASE_URL=https://api.memobase.io
MEMOBASE_API_KEY=your_memobase_api_key
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```



## Commands

In your terminal:

```bash
python livekit_example.py download-files
```

Start a voice conversation:

```bash
python livekit_example.py console
```

You can talk about yourself, like your name/interest, Memobase will keep track on the user's preferences.



Start a voice conversation again to see if the agent remembers you:

```bash
python livekit_example.py console
```

