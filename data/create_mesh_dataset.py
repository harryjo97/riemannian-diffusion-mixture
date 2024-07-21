import os
import igl
import numpy as np
import argparse
import pickle
import scipy
import pyvista as pv
import random

def get_eig_vec(v, f, trunc):
    l = -igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    # generalized eigenvalue problem
    eig_val, eig_vec = scipy.sparse.linalg.eigsh(
        l, trunc+1, m, sigma=0, which="LM", maxiter=100000
    )
    # Only the non-zero eigenvalues and its eigenvectors
    eig_val, eig_vec = eig_val[1:trunc+1], eig_vec[:,1:trunc+1]

    return eig_vec


def plot_mesh_cfp(dataset, v, f, cfp, samples, save_path, step):
    os.makedirs(save_path, exist_ok=True)
    mesh_params={
        'color': 'white',
        'cmap' : 'Reds' #'coolwarm',
    }
    pts_params={
        'color': 'red',
        'point_size': 4
    }
    plot_params = {'spot': {'position': (2.0,2.0,2.0), 
                                    'angles': [0,150,0], 
                                    'focal': (0.,0.1,-0.1)
                    },
                    'bunny': {'position': (2.5,2.5,2.5), 
                            'angles': [-10,60,0], 
                            'focal': (0.,0.0,0.2)
                    }
    }
    pltp = plot_params[dataset]
    position = pltp['position']
    angles = pltp['angles']
    focal_point = pltp['focal']
    
    pv.start_xvfb()
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    
    def rotate(mesh):
        rot = {'point': axes.origin, 'inplace': False}
        mesh = mesh.rotate_x(angle=angles[0], **rot)
        mesh = mesh.rotate_y(angle=angles[1], **rot)
        mesh = mesh.rotate_z(angle=angles[2], **rot)
        return mesh

    pl = pv.Plotter()
    pl.camera = pv.Camera()
    pl.camera.position = position
    pl.camera.focal_point = focal_point
    
    pf = np.concatenate([np.array([3]*f.shape[0]).reshape(-1,1), f], axis=-1)
    poly = pv.PolyData(v, pf)
    poly = rotate(poly)
    pscalars = cfp
    pl.add_mesh(poly, scalars=pscalars, **mesh_params)

    if samples is not None:
        samples = pv.PolyData(samples)
        samples = rotate(samples)
        pl.add_mesh(samples, **pts_params)

    pl.remove_scalar_bar()
    pl.save_graphic(os.path.join(save_path, f"{step}.pdf"))
    pl.close()

    return None


def plot_mesh(dataset, v, f, samples, save_path, step):
    os.makedirs(save_path, exist_ok=True)
    mesh_params={
        'color': 'white',
        'cmap' : 'Reds' 3,
    }
    pts_params={
        'color': 'red',
        'point_size': 4
    }
    plot_params = {'spot': {'position': (2.0,2.0,2.0), 
                                    'angles': [0,150,0], 
                                    'focal': (0.,0.1,-0.1)
                    },
                    'bunny': {'position': (2.5,2.5,2.5), 
                            'angles': [-10,60,0], 
                            'focal': (0.,0.0,0.2)
                    }
    }
    pltp = plot_params[dataset]
    position = pltp['position']
    angles = pltp['angles']
    focal_point = pltp['focal']
    
    pv.start_xvfb()
    axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    
    def rotate(mesh):
        rot = {'point': axes.origin, 'inplace': False}
        mesh = mesh.rotate_x(angle=angles[0], **rot)
        mesh = mesh.rotate_y(angle=angles[1], **rot)
        mesh = mesh.rotate_z(angle=angles[2], **rot)
        return mesh

    pl = pv.Plotter()
    pl.camera = pv.Camera()
    pl.camera.position = position
    pl.camera.focal_point = focal_point

    pf = np.concatenate([np.array([3]*f.shape[0]).reshape(-1,1), f], axis=-1)
    poly = pv.PolyData(v, pf)
    poly = rotate(poly)
    pl.add_mesh(poly, **mesh_params)

    if samples is not None:
        samples = pv.PolyData(samples)
        samples = rotate(samples)
        pl.add_mesh(samples, **pts_params)

    pl.save_graphic(os.path.join(save_path, f"{step}.pdf"))
    pl.close()

    return None


def main(args):
    NUM_SAMPLES = 100000
    random.seed(0)
    np.random.seed(0)
    root = 'data/mesh'
    FILE = f'{args.data}_{args.k}.pkl'
    plot_save_path = 'test/mesh_pdf'

    file_path = os.path.join(root, f'{args.data}.obj')
    v, f = igl.read_triangle_mesh(file_path)

    # downsample for stanford bunny
    if args.data == 'bunny':
        # normalize mesh
        v = v / 250.

    mv, mf = v, f

    # upsample 3 times
    for _ in range(3):
        v, f = igl.upsample(v, f)

    if not os.path.exists(os.path.join(root, FILE)):
        print(f'Create dataset: {FILE}')
        eig_vec_k = get_eig_vec(v, f, args.k)[:,-1]

        # Determine the sign
        if (args.data == 'bunny' and args.k==10) \
            or (args.data == 'spot' and args.k==50) :
            eig_vec_k = -eig_vec_k

        # cfp_ = eig_vec_k[f].mean(-1)
        cfp = igl.average_onto_faces(f, eig_vec_k)

        # Threshold at zero and normalize pdf
        dfp = cfp.clip(0)
        dfp = dfp / dfp.sum() 

        fidx = np.random.choice(np.arange(len(f)), NUM_SAMPLES, p=dfp)
        br = np.sort(np.random.rand(2, NUM_SAMPLES), axis=0)
        br = np.column_stack([br[0], br[1]-br[0], 1.0-br[1]])
        samples = (br[...,None] * v[f[fidx]]).sum(1)

        with open(os.path.join(root, FILE), 'wb') as ff:
            pickle.dump(obj=(samples, cfp), file=ff, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print(f'Dataset exists: {FILE}')
        with open(os.path.join(root, FILE), 'rb') as ff:
            samples, cfp = pickle.load(ff)

    dfp = cfp.clip(0)
    dfp = dfp / dfp.sum() 

    print(f'Plot dataset: {FILE}')
    plot_mesh(args.data, 
        mv, mf, samples, 
        plot_save_path, f'{args.data}_{args.k}_dataset'
    )

    if args.plot:
        # print(f'Plot eigen function: {FILE}')
        # plot_mesh_cfp(args.data, 
        #     v, f, 
        #     cfp, 
        #     None, 
        #     plot_save_path, f'{args.data}_{args.k}_eigenfn'
        # )

        print(f'Plot density: {FILE}')
        plot_mesh_cfp(args.data, 
            v, f, 
            dfp, #cfp, 
            None, 
            plot_save_path, f'{args.data}_{args.k}_density'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='spot')
    parser.add_argument('--trunc', type=int, default=200)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_known_args()[0]
    main(args)