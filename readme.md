In this framework, microphones are streamed, processed, denoised and send to google ASR, and the result is a text published into a rosTopic.
Pepper camera also capture images, save it in a directory, and then they will be processed for object recognition. Object recognition is based on deep learning, currently vgg-16-imagenet model.
Two engine generate answers, which are SeMaC and AIML-ALICE engines, one for semantic mapping and one for general conversation. 
Portal class  tries to get the answer from semantic mapping engine, if not possible, it get the answer from ALICE, and publish its answer to a ros topic.
Pepper internal TTS is used for uttering answers. 

How to run:
---
install see http://wiki.ros.org/nao , see http://wiki.ros.org/nao . naoqi_sensors_py should be there
git clone and 
"./run.sh"
make sure IP address and ports are the same as robot

Testing:
---
main.py and /naoqi_ASR/bestSpeechHypothesis can generate answers.
each module can be run and be triggered by a ros_topic.

@author: M. Farid. (voroujak)

rqt_graph:
---
![picture](rosgraphAll.svg)

@article{faridghasemnia2020towards,
  title={Towards abstract relational learning in human robot interaction},
  author={Faridghasemnia, Mohamadreza and Nardi, Daniele and Saffiotti, Alessandro},
  journal={arXiv preprint arXiv:2011.10364},
  year={2020}
}

If you use this project in your works, please cite: 
@inproceedings{faridghasemnia2019capturing,
  title={Capturing frame-like object descriptors in human augmented mapping},
  author={Faridghasemnia, Mohamadreza and Vanzo, Andrea and Nardi, Daniele},
  booktitle={AI* IA 2019--Advances in Artificial Intelligence: XVIIIth International Conference of the Italian Association for Artificial Intelligence, Rende, Italy, November 19--22, 2019, Proceedings 18},
  pages={392--404},
  year={2019},
  organization={Springer}
}


