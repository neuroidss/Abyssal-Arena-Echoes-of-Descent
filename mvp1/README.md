# [Hierarchical Multi-Bot Sim (Client-Side TFJS) - v6.0.4](https://neuroidss.github.io/Abyssal-Arena-Echoes-of-Descent/mvp1/Hierarchical_Multi-Bot_Sim_Client-Side_TFJS_-_v6.0.4.html)

# vibe coding prompt for [SDR_Prediction_Demo](https://neuroidss.github.io/Abyssal-Arena-Echoes-of-Descent/mvp1/Multi-Bot_SDR_Prediction_Demo_(v4.2_-_HTM_Principles).html)
```
first research how to make SDR (Sparse Distributed Representation) in tfjs.
how to make example where some bot with hardcoded goals will be captured via senses and actions into sdr to learn as bottom-up spatiotemporal hierarchy and make bot with intelligent goals which next actions will be top-down predictions, and anomalies in predictions will rule learning process, make no difference between actions and senses, as future use is to feed eeg which has no specific actions and senses but they need to be learned together with actual sensors and actions when actual sensors and actions will be removed and will remain only eeg.

then make demo where will be bot with hardcoded goals, and then when tfjs will learn internal intelligent goals make another bot with intelligent rules and compare how goals achieved by hardcoded bot and intelligent bot. hardcoded bot should have tasks which achievable in reasonable time and reproducible, to compare different generations of tfjs architectures. make several configurations of tfjs bot to make leaderboard. hardcoded bot should not know where exactly his goals but should search it having only senses and actions, it should take reasonable time to reach its goals.

write all in single html file, but inside html separate library code in which will be only data streams on input and predictions on output to use this library separately later so all tfjs only in this library.

make hardcoded bot run infinitely so intelligent bot could learn new strategies every time. so change environment every time but keep seed. intelligent bot needs to learn to reach goals in any new environment. change environment every time as goals reached by  some of bots. show bots statistics how much goals reached by every of them.

make hardcoded bot run infinitely so intelligent bot could learn new strategies every time. so change environment every time but keep seed. intelligent bot needs to learn to reach goals in any new environment. change environment every time as goals reached by  some of bots. show bots statistics how much goals reached by every of them.

make world larger so make goals far enough to reach so untrained intelligent bot will have zero chance to reach goals faster then hardcoded. because now statistics hardcoded/intelligent 328/24, but it seems because of random moves of untrained bot. make bots not start from only one point so intelligent bot will learn to go in any directions. make bots not just compete each other but also able to punch each other to make opponent stop for some time so intelligent bot will learn to attack, defend or avoid opponents. make multiple goals to reach so no need to run to one direction for both bots. and make bots start far from each other to not start fight immediately, but only in process of reaching goals.

make bot continue moving after stun someone. make all important initial parameter tunable possibly without restart of simulation level, and which not possible without restart level will require level restart, and some will require full reset, but try avoid restarts resets to tune parameters, just use resizes and scaling. make visibility of all goals and opponents in visible range every step and choose which easier to reach, and attack opponent it stops from reaching goals or no goals visible around. stunning stops from acting, making unable to act, but make freezing which make possible to act but all acts will not lead to no result. so after stunning bot will see that it was away for some steps and has no sense for that time, but in freezing bot will see that actions not working and it is it freeze state, so make it feel freezing and its consequences.

make all not stopped after first round. make restart level works, make start simulation works, so simulation not frozen after first round. and make after froze bot not stay on same place freezing other bot until time ends. so make bot see that it freees other bot.

make hardcoded bot not freezes when see goal and there obstacle between bot and goal. make hardcoded bot somehow try to avoid obstacles between it and goals.

make it all vizible okay for mobile device.
made 1ms delay work fine.

make grid size, num goals make real effect, as now goals number not changed really, and seems grid size not make bot act inside new range and obstacles and goals also not placed inside new grid size. so somehow arena size stays same and bots operating inside old grid size with old number of goals after changed and applied force new round, only grid size changed and it empty and unexplorable for bots at new area.

make changeable of bots hardcoded and intelligent, make intelligent bots use same model at first time, but just make ability to choose number of both type of bots.

make it required to make action interaction with goal, not just touch, as with punch of opponent requires action. so intelligent bot will learn maybe how to interact with goals.

make parameters of model and game setup save and restore from localstorage.
make model parameters be changeable to increase hierarchy and other sizes to maximum on current hardware to make most intellegent bots. when localstorage parameters not fit somehow jast turn it to default, also make switch to default.

make inside library absolutely no difference what are senses what are actions. actions are only predictions which intelligent bots used to act outside of library. library only sense and predict all senses, library never act.

use just streams inside of library, absolutely no use of senses or actions inside library, only bots outside library knows what send as actions and what use as actions.

number of bots and grid and goals resize changeable only at reset all.

on start of library need to be only sdr and how it encodes streams. all config about any streams must be outside of library.
on all streams library makes predictions. be ready to send user controls on bot and then get eeg of user. make library work closer as possible to real cortex.

hardcoded bot gives its sensors and actions together as streams. then from bottom up need to be all streams via their sdr in many levels of hierarchy, like on bottom level lines moving it comes up as pattern of symbols, then on next level bottom symbols moving it comes up as pattern of words, then on next level bottom words moving it comes up as pattern of sentences. and all predictions need to be top down, like sentences words symbols and moving lines on level which then decoded back and intelligent bot takes action on part of stream which it knows action, so intelligent bot sending its sensors and actions as streams, they processed via sdr bottom up in spatiotemporal hierarchy to higher order concepts and then predicted top down next stream, and bot uses actions from this predicted stream.

if not sure how then reread again how sdr works.

you can fully rewrite all this tfjs part to make sdr fully work, use any non standard ways. just make streams of sensors and actions process absolutely similar as in cortex, and to only bots know what they send as actions and what as senses, and make intelligent bot takes predictions from actions to act. make library learn on predictions anomaly as cortex do.

make as possibly full implementation of learning on anomaly and make as many as possible levels of hierarchy. and maybe make first own hierarchy levels for each stream and then combine streams on higher hierarchy level.

make intellectual bot learn not on random actions but from hardcoded bot actions when exploration enabled, but make these exploration rank rewards go to hardcoded bots team when exploration enabled.

show as much as possible data from how intellectual bot mind looks inside, like sdr, patterns, hierarchy, predictions, anomalies, maybe how it changes in time. make ability to autochoose exploration level based on how big anomalies, make higher level anomalies more important as they means that bot don't understands higher concepts.

make maybe audiovisual presentation of bots intentions.

mimicing is good idea, as main idea for these bots to learn players intentions but to better oppose players so it should be some third view mimicing, and idea for this library to copy players intention to put later in artifacts to fulfill most desired intention for artifact holder. but for now idea was only to find how for bot faster learn. so maybe make all bots same and only some will enable hardcoded intentions for learning and other fully hardcoded all time, i mean it should be their hardcoded logic but enabled time to time, and most time when not explore mode they should predict what learned as goal to make them work fully on predictions.

don't place end line comments when concatenating code in one line as comments applied to code in line and it cause error.

add most important technologies of how sdr works to learn bottom-up spatiotemporal hierarchically on  anomalies in top-down predictions.

make deepresearch and add as mush as possible details in implementation of sdr bottom-up spatiotemporal patterns recognition and top-down predictions and anomaly based learning.
```
