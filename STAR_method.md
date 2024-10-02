Here’s how I transformed my project explanations:

📌Summarise in One Sentence
Start with a concise summary that encapsulates the project’s purpose and outcome. Instead of, “I worked on a software development project,” I began with, “I led a project to develop a feature that increased user engagement by 20%.”

📌Define the Problem
Clearly articulate the problem you aimed to solve. Instead of vague descriptions, I used specifics: “Our user engagement was dropping by 10% quarterly, impacting revenue.”

📌Outline the Approach
Break down your approach step-by-step. I detailed my method: “I conducted user research, identified key pain points, collaborated with UX designers, and iteratively tested solutions.”

📌Highlight Key Actions
Focus on the most impactful actions you took. For example, “I implemented a new recommendation algorithm that personalized user experiences, leading to a significant engagement boost.”

📌Quantify the Results
Always back your story with data. Instead of general success claims, I stated, “This led to a 30% increase in user interaction within the first week of launch.”

📌Reflect on the Impact
Conclude with the broader impact and any lessons learned. I shared, “The project not only met our engagement goals but also improved my skills in data-driven decision-making and cross-functional collaboration.”

Using this project breakdown framework, I prepped in advance, planned at least 4 stories prior to my interview and ensured I had a list of questions to ask 🚀

Situation:
The client, a small and new business with limited web presence and brand imagery, approached us with three primary use cases to enhance their brand safety:

Impersonation Detection: Identify cases where individuals or entities are impersonating the brand to sell similar products.
Influencer Quality Assurance: Monitor and measure how often influencers in partnership with the brand are posting about it and assess the quality and frequency of these posts.
Limited Brand Imagery: Given the client's limited number of brand images, there was a challenge in finding sufficient web presence to scrape and annotate images of their brand.
Task:
To develop a solution that could effectively detect the client's brand in various social media posts and online content, despite the limited availability of brand images.
Action:
### 1. We came up with an apporah to handle the low training data problem by using a few shot learning approach using vector embeddings. We asked client to provide us with a few images of their brand and we used these images to generate embeddings using a pre-trained model. The pre-trained model was a Vision Transformer model that was trained on a large open logo dataset. We then used these embeddings to compare with the embeddings of the images in the social media posts to detect the brand. We also trained a object detection model to detect the brand logo in the images, we used YOLOv7 models for this purpose. 
The pipeline was like: 
We would use the fews images of the brand provided by the client to generate embeddings using the Vision Transformer model and store these embeddings in a vector database. 
Then we run the object detection model on the social media posts to detect the brand logo in the images.
The object detection model would detect the brand logo in the image and crop the image around the logo. We would then use the cropped image to generate embeddings using the Vision Transformer model.
We would then compare the embeddings of the cropped image with the embeddings of the brand images in the vector database to detect the brand in the social media post. 


