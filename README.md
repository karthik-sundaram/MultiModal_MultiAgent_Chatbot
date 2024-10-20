## Project Overview 
  
MultiModal_MultiAgent_Chatbot is a dynamic chatbot designed to handle text, image, and audio inputs, leveraging multiple specialized tools to process user queries. The chatbot is powered by a multi-agent system with LangGraph and ReactJsonAgent for advanced orchestration, while LangSmith provides monitoring, feedback collection, and performance analysis. The chatbot is deployed through Gradio, offering an accessible API and live link for user interaction.

### Key Features:

- **Multi-modal**: Handles text, images, and audio.  
- **Multi-agent**: Uses tools like ReactJsonAgent and LangGraph to execute tasks based on user input.  
- **Context-aware**: Remembers context across multiple interactions for coherent responses.  
- **Deployed with Monitoring**: Feedback and performance metrics tracked using LangSmith for real-time improvements.  
   
## Model Architecture & Inference Pipeline
This chatbot integrates various models and tools to handle different types of inputs and tasks:  
  
- **Text Generation**: Powered by meta-llama/Meta-Llama-3-8B-Instruct via HfApiEngine for natural language responses.  
- **Image Captioning & Comparison**: Utilizes BLIP for captioning and CLIP for comparing images to text descriptions.  
- **Image Generation**: Uses Stable Diffusion to generate images based on text prompts.  
- **Audio Generation**: Converts text to audio using the Bark model.  
- **Wiki Wrapper Tool**: Allows the chatbot to pull information from Wikipedia, enriching user interactions with factual content.  
  
For each user query, the chatbot dynamically selects the appropriate tool based on the input type (text, image, audio), processes it, and generates a response. The chatbot also remembers context across interactions, ensuring more natural and coherent conversations.  
  
## Multi-Agent Orchestration  
    
The chatbot leverages a multi-agent system using **ReactJsonAgent** to execute tasks step-by-step, making decisions based on context and outcomes, while **LangGraph** provides low-level control for multi-modal interactions, coordinating tools like image captioning, Wikipedia search, and text generation.
  
## Monitoring and Feedback  
  
**LangSmith** enables feedback collection, run monitoring, and performance analysis. Key metrics such as latency, user feedback, and system performance are tracked and visualized in real-time, helping to improve the chatbot's responsiveness and accuracy.
  
### Monitoring Features: 
  
- **Latency Tracking**: Real-time tracking of system response times.  
- **Feedback Logging**: Collects user feedback for continuous improvement.  
- **Performance Dashboard**: Offers insights into system performance, errors, and user satisfaction.

## Colab code link:
https://colab.research.google.com/drive/1bkBykhE0v-sXWxkq9X-I18NAhl8Wu0AJ#scrollTo=jWKVKA4FxMJ8

## Watch a few demos and screenshots here: 
   
[![Watch the video](https://img.youtube.com/vi/5_sIkdMnaVg/maxresdefault.jpg)](https://youtu.be/5_sIkdMnaVg)    

[![Watch the video](https://img.youtube.com/vi/hTIdANQ8ZUU/maxresdefault.jpg)](https://youtu.be/hTIdANQ8ZUU)  


![image](https://github.com/user-attachments/assets/0b6e146a-65f0-46af-ad21-0736f14caa04)

![image](https://github.com/user-attachments/assets/1be1fd22-6540-477b-9a89-b7fe2b66b635)


![image](https://github.com/user-attachments/assets/0bbf8719-fe21-4ef2-a858-0ea7a0b74622)

![image](https://github.com/user-attachments/assets/743a1d3e-d464-4fc7-9c8e-d0808a8a38e2)

![image](https://github.com/user-attachments/assets/8cccccce-8209-4048-a491-40bb1f59923e)

![image](https://github.com/user-attachments/assets/1461b739-a8e9-4d1f-8374-d65e33afb18d)

![image](https://github.com/user-attachments/assets/e444f051-cd79-4f4e-9f66-fbd5e0edcd19)




 
