Step 1. Install Python and Pip.
Step 2. Install dependencies (pip install numpy, pandas, matplotlib covers most of it, I think).
Step 3. Go get the appropriate datafiles from the Cooperative Election Survey website. Put them in the appropriate folders as indicated by the text files in the folders.
Step 4. Set your options in the Python scripts. This isn't fancy. It doesn't handle options through a GUI. Go in, edit the file, save it.
Step 5. Run the python script.

Note: I didn't make a separate system for finding the share of each religion in the general population. I simply removed the gender_filter, changed the age range (default: 44-55) to 0-150, encompassing all adults, and ran the fertility script. I ignored the fertility result and instead checked on the pct_of_sample output for each religion, then fed those numbers into Datawrapper to make the pie charts.

Every LLM on the face of the Earth can walk you through these steps if you need more help, even basic free ChatGPT. Prompt it with something like, "I am trying to run a Python script that charts fertility by religion based on data from the Comprehensive Election Survey, but I don't know how to install Python, pip, numpy, or any of the other things the readme says, and I don't know how to run a Python script after. Here is the readme: (paste this readme into the prompt window). Can you walk me through this? If you need to see the script files themselves, tell me how to open them so I can paste snippets into this conversation."