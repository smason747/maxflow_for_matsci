
img = imread('median.tiff');
img = int16(img);
% select background and give a unique value =-1. From now on, things with 
% negative values are already segmented
img(grayconnected(img,1,1)) = -1;

% now make the big master graph for the cutting. we will slice off pieces 
% of this as we go through the calculations

N = numel(img);
node_map = reshape([1:N],size(img));
L = node_map(:,1:end-1);
R = node_map(:,2:end);
U = node_map(1:end-1,:);
D = node_map(2:end,:);
LR_weight = reshape(256-abs(L-R),[],1).^1;
UD_weight = reshape(256-abs(U-D),[],1).^1;

% Make all the in-plane weights (voxel to voxel, weighted by how close they
% % are to their neighbor in terms of color
DG = digraph;

a = [DG.add_edge(l,r,capacity = w) for l,r,w in np.vstack([L,R,LR_weight]).T]
a = [DG.add_edge(r,l,capacity = w) for l,r,w in np.vstack([L,R,LR_weight]).T]
a = [DG.add_edge(u,d,capacity = w) for u,d,w in np.vstack([L,R,LR_weight]).T]
a = [DG.add_edge(d,u,capacity = w) for u,d,w in np.vstack([U,D,UD_weight]).T]

plt.figure()
plt.imshow(img)

# Start the actual graph cut
grain_Id = -1
while np.sum(img>=0)>0:
    # remove already cut out nodes
    a = [DG.remove_node(x) for x in node_map[img == grain_Id]]
    #Iterate to next grain to cit
    grain_Id -= 1
    #make a guess of something worth segmenting out
    guess = make_a_guess(img[img>=0])
    # Make a list of nodes, and a list of weights representing how close the
    # color of those nodes are to the color of your guess node
    nodes = np.unique(DG.nodes())
    sink_delta = np.ravel(256-np.abs(np.ravel(img)[nodes]-guess))
    sink_weight = (sink_delta+1-np.min(sink_delta))**1
    # connect source(n+1) to voxels
    a = [DG.add_edge(N+1,v,capacity = w) for v,w in np.vstack([nodes,sink_weight]).T]
    # connect voxels to sink (n+2)
    a = [DG.add_edge(v,N+2,capacity = 1/w) for v,w in np.vstack([nodes,sink_weight]).T]
    

Active_DG = DG.copy()

# get a guess of someting to segmet out
guess = make_a_guess(img[img>=0])

def make_a_guess(choices):
    count = (np.histogram(choices, bins=np.arange(-0.5, 256.5))[0])**2
    count = count/np.linalg.norm(count)
    choice = np.random.choice(np.arange(256), 1, p=(count.astype(float))**2)[0]
    return choice
