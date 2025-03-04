<h1>ITCS-3156 Final Report</h1>

<h2>Introduction:</h2>
For my project I decided to attempt to make a reinforcement learning agent that is capable of trading stocks autonomously. The motivation behind this project was to learn about the basic techniques of machine learning and working with neural networks. 
<h2>Data:</h2>
The data for this project was a sample of the top 20 performing stocks on the US stock market over the past 10 years. The data that I decided to look at to train the model was the opening price, closing price and volume of stock traded over the past 7 days, the day and month of the year. To collect this data I used the Tiingo api. When I conducted the preliminary data analysis I noticed that since I chose to collect data from the top 20 stocks most of the stocks had a fairly positive trend upwards, this is important because we should note that all of the stocks that the agent trained on had positive growth so it will probably have unrealistic results.<br/>
<p align="center">
  <img src="plots/amznCorrMat.png" alt="Corr Plot" height="300" align="center">
</p>
<br/>
Looking at the correlation matrix above you can see that there is a negative correlation between volume and both open and close, this makes sense because people are going to be more likely to buy a stock when it is at a lower price. 
<br/>
<p align="center">
  <img src="plots/amznOpenClose.png" alt="Open Close Plot" height="300" align="center">
</p>
<br/>
Looking at this plot of the open and close prices as time evolves you can see that the two prices follow each other very closely, this should be fairly obvious because the close price is very dependent on the open price. The more interesting observation for making trading agents is the steep decline right before day 2000. This is a point in time where amazon split there stock meaning that they increased the amount of stocks available on the market which lowers the cost of any individual stock. This introduces a challenge for us when constructing the environment to train out agents.
<h2>Methods:</h2>
The two models that I decided to use were a double deep Q network and an actor critic network. The reason that I decided to use these models is because they are both very foundational reinforcement learning models and would allow me to get a grasp for how reinforcement learning works and gain a better understanding of the foundations of the topic before trying anything too complicated. For the environment for the agents I broke the data into episodes that were 60 days long. In each episode the agent starts with $10,000 and has to buy or sell 10 stocks based on the information from the previous 7 days. The agents get a reward whenever they sell equal to the value of the stock they sold. The environment was made to mimic an openAI gym environment where within the episode the agent passes an action to the environment's step function and the resulting state, reward and whether or not the environment ended is returned.
Results:
The results from the DDQN were interesting because the plot of the rewards over time during training would gradually increase and then see a steep fall over and over.
<br/>
<p align="center">
  <img src="plots/DoubleDQNResult.png" alt="DDQN Plot" height="300" align="center">
</p>
<br/>
As you can see the peaks of these spikes definitely showed that the agent was able to increase the amount of rewards that it received as it trained. Unfortunately due to the manner in which I constructed the environment the agent was given all episodes of the same stock in a row and then moved on to the next. This is why I believe that we can see the steep fall offs at fairly regular intervals.
<br/>
<p align="center">
  <img src="plots/ActorCriticResult.png" alt="DDQN Plot" height="300" align="center">
</p>
<br/>
The plot for the actor critic model has the same pattern where the agent learns the pattern for the current stock and then resets its progress as it switches to the next stock showing a poor ability for the agent to generalize to the stock market in general.


<h2>Conclusions:</h2>
Some of the key takeaways from this project were the difficulties that come with the machine learning process. One of the biggest that I faced was how much the development time of a project is affected by the training time of your models. I found that it forced me to be a lot more careful when implementing code because if I did something wrong and my program crashes after I had spent the past two hours waiting for training to complete, you lose a lot of time. Another takeaway I found was that one of the more important parts of making a reinforcement learning agent is the construction of your environment; I feel this is one of the biggest flaws with my models. This is because as I had mentioned before when training the models train on one stock till completion and then move on to the next once this causes it to learn the patterns of one stock and not generalize to others very well. To correct this I could have instead shuffled the episodes around so that they were not in order. Another point of improvement I could have made was in my data collection, this is because as mentioned previously I chose the top 20 stocks to train on and this leads to a bias in performance making the agents seem like they are performing better than they really are. For example as a test I made an agent that bought then sold over and over again and it also saw its rewards go up over time merely because I had pre-selected historically well performing stocks. In conclusion this was a great opportunity to learn more and get practice implementing machine learning algorithms and I feel much more prepared to take on more difficult tasks in the future.

<h3>References:</h3>
Geron, Aurelien. “Reinforcement Learning.” Hands-on Machine Learning with Scikit-Learn, Keras, & TensorFlow, 2nd ed., O’Reilly, pp. 609–642. 
Tabor, Phil. “Everything You Need To Master Actor Critic Methods | Tensorflow 2 Tutorial.” YouTube, YouTube, www.youtube.com/watch?v=LawaN3BdI00. Accessed 9 Dec. 2024. 
<h3>Acknowledgments:</h3>
Data: https://www.tiingo.com/
Source Code:
https://github.com/Nick-Petruccelli/stockBot
