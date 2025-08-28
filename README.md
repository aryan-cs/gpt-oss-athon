# Secretary

Designed for the [OpenAI Open Model Hackathon](https://openai.devpost.com/).

Over the course of ~2 weeks, we designed an end-to-end smart agent capable of streamlining one's life by acting as a personal, day-to-day Secretary. Powered by OpenAI's Whisper and other open-source engines, Secretary has the ability to listen, learn, and communicate. Our novel approach to agentic systems also includes custom algorithms develped in-house to improve the reliability & breadth of Secretary's features.

## Features & Functionality

Secretary can listen, speak, and create a personal understanding of you through a custom context-aware portfolio it develops based on your interactions & schedule. Try asking it to do things like:

- Plan your day-to-day activities
- Help you with homework
- Suggest where to grab lunch with your coworker

Or, watch it:

- Automatically turn on the lights & draw the curtains when you get home from work
- Dim the lights & turn on the fan when it's nearing your bedtime
- Turn off the lights when you leave the room

Annd much more!

## Flexibility

For security, we designed Secretary to be a fully autonomous local agent. That is, all data is stored locally & the entire system runs on your hardware at home. No GPU? No worries. Secretary has a lightweight option, leveraging `gpt-oss:20b` for thinking and the `base` Whisper model for speech recognition. 

## To-Do 
- [x] Implement text-to-speech  
- [x] Implement speech-to-text  
- [ ] Implement history & context portfolio building  
- [ ] Implement Map functionality  
- [ ] Implement To-Do functionality  
- [ ] Develop custom hardware  
- [ ] Implement custom hardware control  
