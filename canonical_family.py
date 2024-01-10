import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid

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

#Afficher image d'une courbe par f(.,.,.,u,v,w) sur mlab
def draw_curve(u,v,w):
    r = np.sqrt(u**2 + v**2 + w**2)
    if r > 1:
        print("Error")
        return
    
    N = 256
    M = N*16

    t = np.linspace(0, 2*np.pi, M)
    theta = t
    phi = np.pi/2 + np.cos(5*theta)/2
    X = np.sin(phi)*np.sin(theta)
    Y = np.sin(phi)*np.cos(theta)
    Z = np.cos(phi)
    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0))
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))
    
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    dphi, dtheta = pi/N, pi/N
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
    
    #paramètre du maillage
    N = 256
    #paramètre des courbes
    M = 512

    #courbe paramétrée sur S^3
    s = np.linspace(0, 2*np.pi, M)
    theta = s
    phi = np.pi/2 + np.cos(5*theta)/2
    X = np.sin(phi)*np.sin(theta)
    Y = np.sin(phi)*np.cos(theta)
    Z = np.cos(phi)

    #afficher \Sigma la courbe paramétrée sur S^3
    mlab.plot3d(X, Y, Z, tube_radius=0.003, color=(1,0,0))

    #afficher \Sigma_v l'image de la courbe paramétrée par f
    x,y,z = f(X,Y,Z,u,v,w)
    mlab.plot3d(x, y, z, tube_radius=0.003, color=(0,1,0))
    
    #afficher les points v/|v| et -v/|v|
    mlab.points3d([u/r], [v/r], [w/r], resolution = 32, scale_factor=0.05, color=(1,1,1))
    mlab.points3d([-u/r], [-v/r], [-w/r], resolution = 32, scale_factor=0.05, color=(0,0,0))

    #maillage de la sphere
    dphi, dtheta = pi/N, pi/N
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    X_sphere = sin(phi)*cos(theta)
    Y_sphere = cos(phi)
    Z_sphere = sin(phi)*sin(theta)

    n,m = np.shape(X_sphere)
    d = np.zeros(np.shape(X_sphere))

    #calcul de la distance d_v d'un point du maillage à la courbe \Sigma_v
    for i in range(n):
        for j in range(m):
            k0 = 0
            d[i,j] = np.arccos(X_sphere[i,j]*x[k0]+Y_sphere[i,j]*y[k0] + Z_sphere[i,j]*z[k0])
            for k in range(1,len(s)):
                 c = np.arccos(X_sphere[i,j]*x[k]+Y_sphere[i,j]*y[k] + Z_sphere[i,j]*z[k])
                 if c < d[i,j]:
                     k0 = k
                     d[i,j] = c
            k1 = (k0 + 1) % len(s)
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

    mlab.mesh(x, y, z, color=(0.55,1,0))

    mlab.show()

draw_canonical(0.6,0,0,0.2)