# cx-analysis
## General notes
Repo containing the data analysis pipeline for the Megaphragma connectome project at the Flatiron Institute. 
- src contains objects/methods needed to fetch and organize connectivity data from CATMAID. Produces a Connectome object based on a cfg file. Also saves a couple of Pandas Dataframes with summaries of the data. Most methods in src scripts rely on CATMAID api calls. You can see how the Connectome object is constructed by looking at lamina_preprocessing.py. TODO: This was written and tested for our specific lamina project; make a general preprocessing script/cfg
- After obtaining summary Dataframes, you can run a number of the jupyter-notebooks in the parent directory. Make sure jupytext is installed and included in your jupyter-notebook/lab build to convert the .py files here to .ipynb
