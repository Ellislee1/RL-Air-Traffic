# RL-Airtraffic

Undergraduate dissertation, using RL agents to keep aircraft vertical separation. Full repository can be found on github ([https://github.com/Ellislee1/RL-Air-Traffic](https://github.com/Ellislee1/RL-Air-Traffic)) A video demonstration of the basic system in action can be found here: [https://www.youtube.com/watch?v=HhulECCIXUw&feature=youtu.be](https://www.youtube.com/watch?v=HhulECCIXUw&feature=youtu.be)

## Important Notice About Systems Used

This project uses two systems as its foundation.
For the simulation BlueSky is used. Bluesky is an open source air traffic simulation tool and is built using python:

_Citation info:_ J. M. Hoekstra and J. Ellerbroek, "[BlueSky ATC Simulator Project: an Open Data and Open Source Approach](https://www.researchgate.net/publication/304490055_BlueSky_ATC_Simulator_Project_an_Open_Data_and_Open_Source_Approach)", Proceedings of the seventh International Conference for Research on Air Transport (ICRAT), 2016.

_Github Link:_[https://github.com/TUDelft-CNS-ATM/bluesky](https://github.com/TUDelft-CNS-ATM/bluesky)

The foundation for the RL and most of this project lies with Marc Brittian, his work has been crucial and many of the concepts, methods and classes are based on or use his origional work:

_Related paper:_[https://arxiv.org/abs/1905.01303](https://arxiv.org/abs/1905.01303)

_Github Link:_[https://github.com/marcbrittain/bluesky](https://github.com/marcbrittain/bluesky)

## How To Use This Repo

This repository has been split into three main parts:

1. The project created. This can be found in the `dissertation` folder.
   It should be noted that a copy of BlueSky should be installed. This is for a manual instillation of my system.
2. The BlueSky system. This can be found in the `bluesky-master` folder.
   This contains all the prerequisites to run the system. To setup run `setup-python.bat` followed by `check.py`. To then run the system use the `Bluesky.py` - which will show the visual UI - or user the command: `python BlueSky.py --sim --detached --scenfile multi_agent.scn` from a console, this will automatically begine the training.
3. Output data and files. Copies of weight files, training statistics and other related data can be found in the `data` folder.

## A Notice About the Project

This project was compleated as partial requirement for the BSc in Computer Science Undergraduates degree at Swansea University.
Project Supervisor: Dr Bertie Muller ([https://www.swansea.ac.uk/staff/science/computer-science/mullerb/](https://www.swansea.ac.uk/staff/science/computer-science/mullerb/))
