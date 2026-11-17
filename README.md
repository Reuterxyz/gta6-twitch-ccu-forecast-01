# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Reuterxyz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Reuterxyz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), as five quantiles per hour. The model is a scale-conditioned empirical-quantile forecast fitted at run time from the case's own comparable launches: for each past launch it takes the log-ratio of each hour's viewers to its hour-1 print, weights launches by a Gaussian kernel on log hour-1 viewers around the case's own hour-1 print, and reads the forecast quantiles off the weighted distribution (blended toward a normal fit, with one smoothing pass across hours). A huge hour-1 print means the game is already live in most regions and the day stays fairly flat; a tiny one means most regions have not unlocked yet and the count can explode by the evening.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tqv6qsxf&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
