import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from scipy import stats
from scipy.interpolate import interp1d

#Afficher sphere, méridien et parallèle sur matplotlib 
def draw_normal_sphere():

    #nombre de méridien
    nb_long = 16
    #nombre de parallèle
    nb_lat = 8
    #paramètre du maillage
    N = 128

    #maillage
    theta = np.linspace(0, 2 * np.pi, N)
    phi = np.linspace(0, np.pi, N)
    x_sphere = np.outer(np.cos(theta), np.sin(phi))
    y_sphere = np.outer(np.sin(theta), np.sin(phi))
    z_sphere = np.outer(np.ones(np.size(theta)), np.cos(phi))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.5)

    #méridiens
    for i in range(nb_long + 1):
        phi = np.linspace(0, np.pi, N)
        theta = i * 2 * np.pi / nb_long
        x_circle = np.sin(phi)*np.sin(theta)
        y_circle = np.sin(phi)* np.cos(theta)
        z_circle = np.cos(phi)
        ax.plot(x_circle, y_circle, z_circle, color='black', alpha=1)

    #parallèles
    for i in range(1,nb_lat + 1):
        phi = i * np.pi / nb_lat
        theta = np.linspace(0, 2*np.pi, N)
        x_circle = np.sin(phi)*np.sin(theta)
        y_circle = np.sin(phi)* np.cos(theta)
        z_circle = np.cos(phi)
        ax.plot(x_circle, y_circle, z_circle, color='black', alpha=1)

    plt.show()

#Fonction F_(u,v,w)
#X,Y,Z : liste de coordonnées de S^2
#u,v,w : point de B^3
def f(X,Y,Z,u,v,w):
    a = 1-u**2 - v**2 - w**2
    B = (X-u)*(X-u) + (Y-v)*(Y-v) + (Z-w)*(Z-w)
    R = a * (X-u) / B - u
    S = a * (Y-v) / B - v
    T = a * (Z-w) / B - w
    return (R,S,T)

#Distance géodésique entre deux liste de points (X,Y,Z) et (R,S,T)
def d_geo(X,Y,Z,R,S,T):
    return np.arccos(X*R+Y*S+Z*T)

#Afficher image des méridiens et parallèles par f(.,.,.,u,v,w) sur mlab
def draw_sphere(u,v,w):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    nb_long = 32
    nb_lat = 16
    N = 256
    M = N*4

    dphi, dtheta = pi/N, pi/N
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    x = sin(phi)*cos(theta)
    y = cos(phi)
    z = sin(phi)*sin(theta)
    mlab.mesh(x, y, z, color=(1,0.55,0))

    for i in range(nb_long + 1):
        phi = np.linspace(0, np.pi, M)
        theta = i * 2 * np.pi / nb_long
        X = np.sin(phi)*np.sin(theta)
        Y = np.sin(phi)* np.cos(theta)
        Z = np.cos(phi)
        x,y,z = f(X,Y,Z,u,v,w)

        mlab.plot3d(x, y, z, tube_radius=0.003, color=(1,1,1))

    for i in range(1,nb_lat):
        phi = i * np.pi / nb_lat
        theta = np.linspace(0, 2*np.pi, M)
        X = np.sin(phi)*np.sin(theta)
        Y = np.sin(phi)* np.cos(theta)
        Z = np.cos(phi)
        x,y,z = f(X,Y,Z,u,v,w)

        mlab.plot3d(x, y, z, tube_radius=0.003, color=(1,1,1))

    mlab.show()

def curve(t):
    theta = 2*np.pi*t
    phi = np.pi/2 + np.sin(4*theta)/4

    X = np.sin(phi)*np.sin(theta)
    Y = np.sin(phi)*np.cos(theta)
    Z = np.cos(phi)
    return X,Y,Z

def tangent(X,Y,Z):

    N = len(X)
    T_x, T_y, T_z = N*(X-np.append(X[1:],X[0])),N*(Y-np.append(Y[1:],Y[0])),N*(Z-np.append(Z[1:],Z[0]))
    return T_x, T_y, T_z

def normal(X,Y,Z):
    T_x, T_y, T_z = tangent(X,Y,Z)
    V = np.stack((X,Y,Z), axis=-1)
    T = np.stack((T_x,T_y,T_z), axis=-1)
    N = np.cross(V,T)
    A,B,C = np.array([N[k][0] for k in range(len(N))]),np.array([N[k][1] for k in range(len(N))]),np.array([N[k][2] for k in range(len(N))])

    A,B,C= (A-(A*X +B*Y + C*Z)*A),(B-(A*X +B*Y + C*Z)*B),(C-(A*X +B*Y + C*Z)*C)
    norm= np.sqrt(A*A+B*B+C*C)
    A,B,C = A/norm,B/norm,C/norm
    return A,B,C
    
#Afficher image d'une courbe par f(.,.,.,u,v,w) sur mlab
def draw_curve(u,v,w):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    #Paramètre d'échantillonage
    N_sphere = 100
    N_gradient = 1000
    N_courbe = 2000

    #Calcul des points de la courbe avec échantillonage uniforme
    s = np.linspace(0, 1, N_gradient)
    X,Y,Z = curve(s)
    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0), line_width = 4.0)
    x,y,z = f(X,Y,Z,u,v,w)

    #Estimation des distance entre les images
    d = np.sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
    d /= np.sum(d) 
    d = np.cumsum(d)

    #Echantillonage non uniforme afin que les images soient régulièrement espacées
    interp = interp1d(s,d,kind='linear')
    new_t = interp(np.linspace(0, 1, N_courbe))
    X,Y,Z = curve(new_t)
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))
    
    #Affiichage de la normale
    a,b,c = normal(x,y,z)
    mlab.quiver3d(x,y,z,a,b,c,color = (0,0,0), line_width = 6.0)


    #Afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    #Afficher la sphère
    dphi, dtheta = pi/N_sphere, pi/N_sphere
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    x = sin(phi)*cos(theta)
    y = cos(phi)
    z = sin(phi)*sin(theta)
    mlab.mesh(x, y, z, color=(1,0.55,0))

    mlab.show()

#Afficher la famille canonique de paramètre ((u,v,w), t) d'une courbe sur mlab
def draw_canonical(u,v,w,t):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    #Paramètre d'échantillonage
    N_sphere = 1000
    N_gradient = 100
    N_courbe = 100

    #Calcul des points de la courbe avec échantillonage uniforme
    s = np.linspace(0, 1, N_gradient)
    X,Y,Z = curve(s)
    #mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0))
    x,y,z = f(X,Y,Z,u,v,w)

    #Estimation des distance entre les images
    d = np.sqrt((x-u)*(x-u) + (y-v)*(y-v) + (z-w)*(z-w))
    d /= np.sum(d) 
    d = np.cumsum(d) 

    #Echantillonage non uniforme afin que les images soient régulièrement espacées
    interp = interp1d(s,d,kind='linear')
    new_s = interp(np.linspace(0, 1, N_courbe))
    X,Y,Z = curve(new_s)
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))

    #Afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    #Maillage de la sphere
    dphi, dtheta = pi/N_sphere, 2*pi/N_sphere
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    X_sphere = sin(phi)*cos(theta)
    Y_sphere = cos(phi)
    Z_sphere = sin(phi)*sin(theta)

    n,m = np.shape(X_sphere)
    d = np.zeros(np.shape(X_sphere))

    #Calcul de la distance d_v d'un point du maillage à la courbe \Sigma_v
    for i in range(n):
        print("Progression : " + str(i+1) + "/" + str(n))
        for j in range(m):
            k0 = 0
            d[i,j] = np.arccos(X_sphere[i,j]*x[k0]+Y_sphere[i,j]*y[k0] + Z_sphere[i,j]*z[k0])
            for k in range(1,N_courbe):
                 c = np.arccos(X_sphere[i,j]*x[k]+Y_sphere[i,j]*y[k] + Z_sphere[i,j]*z[k])
                 if c < d[i,j]:
                     k0 = k
                     d[i,j] = c
            k1 = (k0 + 1) % N_courbe
            u = np.array([X_sphere[i,j],Y_sphere[i,j], Z_sphere[i,j]])
            v0 = np.array([x[k0],y[k0],z[k0]])
            v1 = np.array([x[k1],y[k1],z[k1]])
            w = np.cross(u,v1-v0)
            r = np.linalg.norm(w + v0)
            if r < 1:
                d[i,j] = - d[i,j]
    
    e = (d < t).astype(int)

    x = X_sphere*e
    y = Y_sphere*e
    z = Z_sphere*e

    mlab.mesh(x, y, z, color=(1,0.55,0))

    e = 1-e
    x = X_sphere*e
    y = Y_sphere*e
    z = Z_sphere*e

    mlab.mesh(x, y, z, color=(1,0.7,0))

    mlab.show()

def draw_tubular(r):

    N_courbe = 1000
    N_sphere = 1000

    kappa = 7

    s = np.linspace(0, 1, N_courbe)
    t = np.linspace(-0.65, 3.8, N_courbe)

    X,Y,Z = curve(s)

    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(0,1,0))

    X1 = np.roll(X,1)
    Y1 = np.roll(Y,1)
    Z1 = np.roll(Z,1)
    x0 = np.column_stack((X,Y,Z))
    x1 = np.column_stack((X1,Y1,Z1))
    N = np.cross(x1-x0,x0)
    N = N / np.linalg.norm(N, axis=0)

    cost = (1-r/kappa*np.sin(t))*np.cos(r*np.cos(t))
    sint = (1-r/kappa*np.sin(t))*np.sin(r*np.cos(t))
    
    x = np.outer(cost,X) + np.outer(sint,N[:,0])
    y = np.outer(cost,Y) + np.outer(sint,N[:,1])
    z = np.outer(cost,Z) + np.outer(sint,N[:,2])

    mlab.mesh(x, y, z, color=(1,1,0))
    #mlab.quiver3d(X,Y,Z,N[:,0], N[:,1], N[:,2])
    mlab.show()

#draw_tubular(0.3)
