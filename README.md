# maxflow_for_matsci
wrappers and examples for how to use the maxflow graph cut technique for material science specific problems


# ===================== #
# NOTE TO MATLAB USERS:
Nothing in this code explicitly requires python, and recreating this in MATLAB
should be doable. The magic function you need is called "digraph", and there is
a maxflow function in one of the toolboxes. HOWEVER, the process of building
the graph is fundamentally different, so don't try to make a line-for-line copy
as it will certainly fail. instead, just google "max cut min flow image
processing MATLAB" and go from there.
# ===================== #
