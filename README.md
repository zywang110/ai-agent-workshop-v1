# AI Agent Workshop

## Overview
Welcome to our AI agent Workshop! This repository contains all the materials and code you'll need for the hands-on part of this workshop. The hands-on part comes with two sessions. The first one is focused on designing new tools for an AI agent, and the second part is on exploring various vector search algorithms.

Goals and instructions of each session are in `Hands-on-1.md` and `Hands-on-2.md`. If you are just getting started, you can follow the steps below to set up your local environment.

## Prerequisites
- Python 3.10+
- OpenAI API key

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yefan/ai-agent-workshop.git
   ```
2. Install dependent libraries in a virtual environment:
   If you have `venv` already installed, create a new virtual environment:
   ```
   python -m venv venv
   ```
   If you are running into error because venv is not yet installed, you can install it using `pip`
   ```
   pip install virtualenv
   ``` 
   Activate the virtual environment (by running the scripts in the `venv/Scripts` folder), then install the dependencies through pip:
   ```
   pip install -r requirements.txt
   ```
3. Add your OpenAI API key to the `.env` file:
    `.env` file stores environment variables that can be loaded by the `dotenv` library. Typically this file is kept private, and not checked in to a repo. There's a `.env.example` file, which you can make a copy from, and replace the API key in that file by your own key. 

## Running the code
For the first session, run
```
python image_agent.py
```
If not running into errors, you should see an output like "I have found the image of a cat reading a book".

For the second session, run
```
python image_embedding.py
```
If not running into errors, you should see an output like
```
Closest image: cat_studying_b.png
```

Great! Now you are all set. Start hacking! 
