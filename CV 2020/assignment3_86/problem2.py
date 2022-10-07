import numpy as np
import random
class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        #
        # You code here
        n=features2.shape[1]
        m=features1.shape[1]
        dis=np.empty((n,m))
        for i in range(n):
            for j in range(m):
                dis[i][j]=sum((features2[:,i]-features1[:,j])**2)
        return dis
    
        #

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        
        #
        # You code here
        m=p1.shape[0]
        n=p2.shape[0]
        a=min(m,n)#m
        if(a==m):
            dis=distances.T
            pairs=np.empty((a,4))
            for i in range(a):
                b=np.argmin(dis[i]) 
                pairs[i][:2]=p1[i]
                pairs[i][2:]=p2[b]

        else:
            dis=distances
            pairs=np.empty((a,4))
            for i in range(a):
                b=np.argmin(dis[i])
                pairs[i][:2]=p2[i]
                pairs[i][2:]=p1[b]           
            
        return pairs
        
        #


    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        
        #
        # You code here
        n=p1.shape[0]
        index=random.sample(range(n),k)
        sample1=np.empty((k,2))
        sample2=np.empty((k,2))
        for i in range(k):
            sample1[i]=p1[index[i]]
            sample2[i]=p2[index[i]]
        return sample1,sample2
            
        #


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

        #
        # You code here
        s_x=0.5*np.max(points[:,0])
        s_y=0.5*np.max(points[:,1])
        t_x=np.mean(points[:,0])
        t_y=np.mean(points[:,1])
        T=np.array([[1/s_x,0,-t_x/s_x],
                    [0,1/s_y,-t_y/s_y],
                    [0,0,1]
                    ])
        p_h1=np.concatenate((points.T, np.ones((1,len(points)))))
        ps=T.dot(p_h1).T
        return ps,T
        
        
        #


    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        #
        # You code here
        l=p1.shape[0]
        M=np.empty((2*l,9))
        for i in range(l):
            x1=p1[i,0]
            y1=p1[i,1]
            x2=p2[i,0]
            y2=p2[i,1]
            M[2*i]=[0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2]
            M[2*i+1]=[-x1,-y1,-1,0,0,0,x1*x2,y1*x2,x2]
        u,s,v=np.linalg.svd(M)
        HC=v.T[:,-1].reshape(3,3)
        H=np.linalg.inv(T2).dot(HC).dot(T1)
        H=H/H[-1,-1]
        HC=HC/HC[-1,-1]
        return H,HC
        #

    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

        #
        # You code here
        l=p.shape[0]
        p1=np.concatenate((p, np.ones((l,1))),axis=1)
        p_t=H.dot(p1.T).T#l,3
        p_t2=np.empty((l,2))
        for i in range(l):
            p_t2[i,0]=p_t[i,0]/p_t[i,2]
            p_t2[i,1]=p_t[i,1]/p_t[i,2]
        return p_t2
        
        #


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        #
        # You code here

        

        l=p1.shape[0]
        dist=np.empty((l,))
        Hp1=self.transform_pts(p1,H)   
        Hp2=self.transform_pts(p2, np.linalg.pinv(H))
        for i in range(l):
            dist[i]=np.linalg.norm(Hp1[i]-p2[i])**2+np.linalg.norm(p1[i]-Hp2[i])**2
        return dist
        
        #


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        #
        # You code here
        l=pairs.shape[0]
        inliers=[]
        N=0
        for i in range(l):
            if(dist[i]<=threshold):
                N+=1
                inliers.append(pairs[i])
        
        return N,np.array(inliers)
        
        #


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        #
        # You code here
        return int(np.log(1-z)/np.log(1-p**k))+1
        
        #



    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        #
        # You code here
        pair1=pairs[:,:2]
        pair2=pairs[:,2:]
        N=[]
        inliers=[]
        Hs=[]
        for i in range(n_iters):
            sample1,sample2=self.pick_samples(pair1, pair2, k)
            ps1,T1=self.condition_points(sample1)
            ps2,T2=self.condition_points(sample2)
            H,HC=self.compute_homography(ps1, ps2, T1, T2)
            dist=self.compute_homography_distance(H, pair1, pair2)
            n,inlier=self.find_inliers(pairs, dist, threshold)
            N.append(n)
            inliers.append(inlier)
            Hs.append(H)
        index=np.argmax(N)
        
        return Hs[index],N[index],inliers[index]  
        
        #


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        #
        # You code here
        pair1=inliers[:,:2]
        pair2=inliers[:,2:]

        ps1,T1=self.condition_points(pair1)
        ps2,T2=self.condition_points(pair2)
        H,HC=self.compute_homography(ps1, ps2, T1, T2)
        
        return H   
        
        
        
        #