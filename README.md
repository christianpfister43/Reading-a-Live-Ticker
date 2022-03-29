# Reading-a-Live-Ticker
A Tutorial on how to read a live ticker with machine learning

This repository is part of a multipart Blog post on my [homepage](https://cpfister.com) 
1. Introduction: Part I
2. Create a [Dataset](https://github.com/christianpfister43/CPD-Dataset) suited for the Problem: Part II
3. Get your live data with Screen Parsing: Part III
### 4. Develop a Neural Network to solve the problem: Part IV 
5. Deployment Hardware: Raspberry Pi Setup: Part V (coming soon)
6. Full Example, deployed on Raspberry Pi: Part VI (coming soon)
7. Improvements, Monitoring, Remote Management: Part VII (coming soon)


## Part III: Screen Parsing:
Please install the required Python packages `pip install -r requirements.txt`

open the example page of the German depth-clock: https://www.gold.de/staatsverschuldung-deutschland/
run the script, make sure the digits are visible on the screen.
check on the results in the data folder, iterate over the parameters until you hit all digits properly!

## Part IV: Develop a Neural Network to read the digit-pictures
I am using Tensorflow 2.1, make sure to use a compatible version (most 2.x should work).
The network used here is a simple CNN, which has proven to be very reliable in deployments.

The dataset used for training can be found [here](https://github.com/christianpfister43/CPD-Dataset), 
make sure to unpack the images in a subfolder: `./data/train/`,
have the `labels.csv` in the main folder, and create an empty folder `/models` to save the trained model
