# nk
This repo includes the code for running some of the seminal papers on organizational learning under complexity.    

Currently, it includes the code for creating landscapes a lá Levinthal (1997). The code is presented in two ways. The levinthal_create_landscape.ipynb introduces the logic used to create NK landscapes. The levinthal.ipynb replicates Dan's 1997 paper.  

The next step will be to create a tutorial, I call "Seeing in NK". NK Landscapes are complex. But they are an analogy of what people do, this tutorail will aim at putting you in the feet of our agents, in progressibly more complex environments. The tutorial will depart from the landscapes built in Levinthal (1997), then follow the random walk from Rivkin (2000), continue with rugged search in 2D landscapes as in Adner, Csaszar, and Zemsky (2014), and finish with the multi-attribute aggregation view from Csaszar and Levinthal (2016)

# run online [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jparrieta/nk/master)
You can run this code directly by opening the following binder in your browser or clicking the button above.
It takes a while to load. After loading, click any \*.ipynb  and you will be able to run an interactive verion of the Jupyter notebooks. You do not need to install Python or any dependencies, just run it from your browser.

**Link:** https://mybinder.org/v2/gh/jparrieta/nk/master

# log  

**191002:** Code 10x faster!, a.k.a never use np.random.choice.  
**191001:** Sped up from quadratic to linear time in N and K.  
**190930:** Separated the create landscape tutorial from the full Levinthal (1997) tutorial.  
**190929:** Optimized search procedure.  
**190927:** Included the search functionality. Still no myopia.  
**190916:** Added the landscape building functionality. Search is not working at the moment.  
