from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color, Line, Ellipse
from kivy.uix.label import Label
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

import os
import json
import cv2
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from scipy.ndimage import gaussian_filter1d, median_filter

import read_raw_file as RRF
import marker_detection

class MyApp(Widget):
    Window.maximize()
    Builder.load_file('design_interface.kv')
    Config.set('graphics', 'width', '1920')
    Config.set('graphics', 'height', '1080')

    global path
    path = ''

    # Fonction pour que les boutons changent de couleur lorsqu'enfoncés
    def press_color(self):
        self.background_normal = ''
        self.background_color = (100/255.0, 197/255.0, 209/255.0, 1)

    # Fonction pour sélectionner le répertoire .raw, puis définir le nombre d'images
    def im_select(self):
        print(self.width, self.height)
        timer_debut_im = time.process_time_ns()
        
        # flag pour savoir quelles infos sont disponibles
        global analyse_eff
        analyse_eff = False
        global detection_eff
        detection_eff = False
        global labelize_extent
        labelize_extent = False

        # Initie la variable pour numéro de l'image affichée à 1 pour voir la première image
        global image_nb
        image_nb = 1

        # chemins général/spécifiques vers les données de base/créées
        global path
        path = self.ids.path_input.text
        save_path = path+'/intensity/'
        save_path_xyz = path+'/xyz/'
        global save_path_im
        save_path_im = path+'/Preprocessed'

        try: # si chemin entré valide
            # Crée les répertoires pour images converties et prétraitées
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(path+'/landmarks/', exist_ok=True)
            os.makedirs(save_path_xyz, exist_ok=True)
            os.makedirs(save_path_im, exist_ok=True)
            # lit les fichiers .raw si pas déjà fait et crée les images
            if len(os.listdir(save_path)) == 0:
                RRF.read_raw_intensity_frames(path, save_path)
            
            # définit les dimensions pour rogner les images
            self.automatic_crop()

            # crée les images Preprocessed pour consultation
            if len(os.listdir(save_path_im)) == 0:
                for filename_i, filename_xyz in zip(os.listdir(save_path), os.listdir(path+'/xyz_nobg/')):
                    frame_display, preprocessed_frame = marker_detection.preprocess(cv2.imread(os.path.join(save_path, filename_i)), cv2.imread(os.path.join(path+'/xyz_nobg/', filename_xyz)), w1, w2, h1, h2)
                    cv2.imwrite(os.path.join(save_path_im, filename_i), preprocessed_frame)

            if len(os.listdir(save_path_xyz)) == 0:
                RRF.copy_xyz_frames(path, save_path_xyz)

            # Trouve le nombre d'images, définit le max du slider et le texte /tot
            global images_total
            images_total = len(os.listdir(save_path))
            self.ids.slider.max = images_total
            self.ids.image_total.text = f'/{images_total}'
            self.ids.label_ready.text = "Images prêtes, bougez le curseur ou entrez un numéro d'image"
        
            # Initie les variables pour dictionnaire de coordonnées et flag analyse_eff (pour affichage x,y,z)
            global dict_coordo
            dict_coordo = {}
            global dict_coordo_labels_manual
            dict_coordo_labels_manual = {}
            
            # Lance la détection des marqueurs et affiche la 1re image
            self.detect_marqueurs()
            self.show_image()

            timer_fin_im = time.process_time_ns()
            print(timer_debut_im, timer_fin_im)
            print(f'Temps création images + détection marqueurs : {timer_fin_im - timer_debut_im} ns')

        except FileNotFoundError:
            self.ids.label_ready.text = "Le chemin entré est introuvable. Essayez à nouveau."

    # Trouve les paramètres pour rogner les images (utilisé pour preprocess (RRF) et marker_detection)
    def automatic_crop(self):
        timer_debut = time.process_time_ns()

        path_nobg = path+'/xyz_nobg/'
        if not 'xyz_nobg' in os.listdir(path):
            os.mkdir(path_nobg)
            RRF.read_raw_xyz_frames(path, path_nobg)

        z_nobg = cv2.imread(os.path.join(path_nobg, os.listdir(path_nobg)[0]))
        z_nobg = median_filter(z_nobg, 3)
        print(z_nobg.shape)
        body = np.argwhere(z_nobg[1000,:,0] > 100) #identifie points n'appartenant pas au bg, donc au corps du patient

        #body_sorted = sorted(body, key=lambda list: list[1]) #tri selon x (coordos horizontalement)
        left = int(body[0])
        right = int(body[-1])

        global w1
        global w2
        global h1
        global h2

        if 'BG' in os.listdir(path_nobg)[0]:
            w1 = left-50
            w2 = w1+600
        elif 'BD' in os.listdir(path_nobg)[0]:
            w2 = right+50
            w1 = w2-600
        else:
            w1 = left-50
            w2 = w1+600

        h1 = 560
        h2 = 1280

        print(w1, w2, h1, h2)

        self.ids.width.text = f'({w2-w1}, 0)'
        self.ids.height.text = f'(0, {h2-h1})'

        timer_fin = time.process_time_ns()
        print(timer_debut, timer_fin)
        print(f'Temps automatic crop : {timer_fin - timer_debut} ns')

    # Fonction pour définir le nombre de marqueurs utilisés
    def nb_marqueurs_input(self):
        global nb_marqueurs
        nb_marqueurs = int(self.ids.marq_nb_input.text)
        self.ids.grid.size_hint = (.22, .04 + .03*nb_marqueurs) # taille du tableau variable selon le nombre de marqueurs
        self.ids.grid.rows = 1 + nb_marqueurs
        print(f'{nb_marqueurs} marqueurs utilisés')

    # Fonction pour sélectionner le numéro de l'image à afficher, actualise la position du curseur
    def image_nb_input(self):
        global image_nb
        image_nb = int(self.ids.image_nb_input.text) # Définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb <= images_total:
            self.ids.slider.value = image_nb #lien entre position du curseur et numéro de l'image
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés
        else:
            self.ids.image_nb_input.text = 'Invalide' #si numéro entré à l'extérieur de l'intervalle acceptable

    # Fonction pour parcourir les images avec le curseur, actualise le numéro de l'image
    def slider_pos(self, *args):
        global image_nb
        image_nb = args[1] # définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb  <= images_total:
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés

    # Fonction pour afficher l'image et effacer les informations relatives à la précédente (marqueurs)        
    def show_image(self):
        self.ids.image_nb_input.text = f'{image_nb}'
        self.ids.image_show.clear_widgets()
        self.ids.image_show.source = os.path.join(save_path_im, os.listdir(save_path_im)[image_nb-1])
        self.canvas.remove_group(u"circle") # efface les cercles verts des marqueurs
        self.ids.rep_continuity.text = ''
        
        # Affiche les marqueurs si bouton activé
        if self.ids.button_showmarks.state == 'down':
            self.show_marqueurs()
        if self.ids.button_distances.state == 'down':
            self.show_marqueurs_gold()
            self.remove_widget(self.Distances)
            self.show_distances()

        # Affiche les numéros d'images n'ayant pas le bon nb de marqueurs si bouton activé
        if self.ids.button_verif_nb.state == 'down':
            self.verif_nb()
        # Tag marqueurs discontinus si présents dans l'image actuelle
        if self.ids.button_verif_continuity.state == 'down':
            im_prob_continuity = self.verif_continuity()
            for el in im_prob_continuity:
                if image_nb == el:
                    self.ids.rep_continuity.text = 'Marqueur(s) à vérifier :\n'
                    for m in im_prob_continuity[image_nb]:
                        self.ids.rep_continuity.text += f'{m}, '
                    self.ids.rep_continuity.text = self.ids.rep_continuity.text[:-2]

        # Affiche positions dans le tableau après labellisation manuelle
        if labelize_extent and not analyse_eff:
            self.ids.grid.clear_widgets()
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            for l in labels:
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                    self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                else:
                    self.ids.grid.add_widget(Label(text=f'?', color=(0,0,0,1)))
    
        # Actualisation tableau de coordonnées avec coordos x,y,z si analyse effectuée
        if analyse_eff:
            self.ids.grid.clear_widgets()
            self.ids.grid.cols = 4
            self.ids.grid.rows = 1 + nb_marqueurs
            self.ids.grid.size_hint = (.27, .2)
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Coordonnées (x,y,z)', color=(0,0,0,1)))
            d_im = dict_coordo_xyz_labels[f'image{image_nb}']
            for key, l in zip(d_im.keys(), labels):
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                else:
                    p = [np.nan, np.nan]
                self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                self.ids.grid.add_widget(Label(text=f'{key}', color=(0,0,0,1)))
                c = d_im[key]
                self.ids.grid.add_widget(Label(text=f'({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})', color=(0,0,0,1)))
            self.ids.origine.text = ''
            self.ids.width.text = ''
            self.ids.height.text = ''
        
        # Efface les marques de labellisation manuelle (ronds bleus) si désactivé
        if self.ids.labelize_manual.state == 'normal':
            self.canvas.remove_group(u"label")
        
    # Fonction pour détecter les marqueurs de toutes les images du répertoire
    def detect_marqueurs(self):
        timer_debut_detection = time.process_time_ns()
        global detection_eff
        
        if len(path) > 1:
            os.makedirs(path+'/annotated_frames/', exist_ok=True)
            # Détecte les marqueurs, crée images annotées et fichiers txt avec positions
            if len(os.listdir(path+'/annotated_frames/')) == 0: #or self.ids.check_new.state == 'down':
                marker_detection.annotate_frames(path, w1, w2, h1, h2)
        
        global dict_coordo
        dict_coordo = {}
        i = 1
        for filename in os.listdir(path+'/Preprocessed/'):
            key_points = marker_detection.detect_markers(cv2.imread(os.path.join(path+'/Preprocessed/', filename)))
            points = [key_points[j] for j in range(len(key_points))]
            #marker_array[0][i] = [[point.pt[0], point.pt[1]] for point in points]
            dict_coordo.update({f'image{i}' : [[float(point.pt[0]), float(point.pt[1])] for point in points]})
            i += 1

        """ # Recrée le dictionnaire de coordonnées à partir des fichiers de landmarks
        for filename in os.listdir(path+'/landmarks/'):
            coordinates = []
            keypoints = []
            with open(os.path.join(path+'/landmarks/', filename), 'r') as f:
                for line in f.readlines():
                    keypoints.append(line.split(' '))
            for n in range(len(keypoints)):
                coordo_xy = [float(keypoints[n][0]), float(keypoints[n][1])]
                coordinates.append(coordo_xy)
            i = int(filename[10:-4])
            dict_coordo[f'image{i+1}'] = coordinates """
            
        detection_eff = True

        global im_dim
        im_dim = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[0])).shape

        # Va chercher les positions corrigées enregistrées si mode Ouvrir
        if self.ids.check_open.state == 'down':
            #global analyse_eff
            #analyse_eff = 'Metriques' in os.listdir(path+'') #Analyse effectuée (et utilisable) si métriques enregistrées
            
            if 'Positions' in os.listdir(path):
                # Recrée dictionnaire de positions avec labels
                global dict_coordo_labels_manual
                jsonfile = open(path+'/Positions/positions_corrigees.json')
                dict_coordo_labels_manual = json.load(jsonfile)
                for key, marqueurs in dict_coordo_labels_manual.items():
                    dict_coordo.update({key:[]})
                    for m in marqueurs.values():
                        dict_coordo[key].append(m)

                global labels
                labels = dict_coordo_labels_manual['image1'].keys()

                global nb_marqueurs
                nb_marqueurs = len(dict_coordo_labels_manual['image1'])
                self.ids.grid.size_hint = (.22, .04 + .03*nb_marqueurs)
                self.ids.grid.rows = 1 + nb_marqueurs

                detection_eff = True

                if 'coordonnees_xyz.csv' in os.listdir(path+'/Positions/'):
                # Recrée le dictionnaire de coordonnées x,y,z
                    global dict_coordo_xyz_labels
                    dict_coordo_xyz_labels = {}
                    with open(path+'/Positions/coordonnees_xyz.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        j = 0
                        for row in reader: #skip headline
                            if j == 0:
                                entete = row[1::3]
                                labels_xyz = [e[:-2] for e in entete]
                                print(labels_xyz)
                            elif j > 0:
                                key = f'image{row[0]}'
                                dict_coordo_xyz_labels.update({key: {}})
                                row = [float(i) for i in row[1:]]
                                i = 0
                                for l in labels_xyz:
                                    dict_coordo_xyz_labels[key].update({l : [row[i], row[i+1], row[i+2]]})
                                    i += 3
                            j += 1
                        print(dict_coordo_xyz_labels)
                else:
                    self.coordo_xyz_marqueurs()

                self.analyse()

        timer_fin_detection = time.process_time_ns()
        print(timer_debut_detection, timer_fin_detection)
        print(f'Temps détection des marqueurs :{timer_fin_detection - timer_debut_detection} ns')

    # Fonction pour afficher les marqueurs sur l'image actuelle
    def show_marqueurs(self):
        # Affichage des marqueurs si bouton activé
        if detection_eff == True:
            for coordinates in dict_coordo[f'image{image_nb}']:
                x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.03
                y = 0.85 - (coordinates[1]/im_dim[0])*0.78
                with self.canvas:
                    Color(0,1,0,1)
                    Line(circle=(self.width*x, self.height*y,6,0,360), width=1.1, group=u"circle") #(center_x, center_y, radius, angle_start, angle_end, segments)
        # Efface les marqueurs si bouton désactivé
        if self.ids.button_showmarks.state == 'normal':
            self.canvas.remove_group(u"circle")

    # Fonction pour vérifier le nombre de marqueurs détectés pour toutes les images du répertoire
    # (avec dictionnaire de coordonnées créé)
    # Retourne la liste des images n'ayant pas 5 marqueurs détectés et les liste dans une nouvelle fenêtre
    def verif_nb(self):
        if detection_eff == True:
            im_prob_nb = [] #liste des images avec ±5 marqueurs
            for im, coordo in dict_coordo.items():
                if len(coordo) != nb_marqueurs:
                    im_prob_nb.append(int(im[5:])) #ajoute l'image et le nombre de marqueurs détectés à la liste
            for im, dict in dict_coordo_labels_manual.items():
                    coordo = list(dict.values())
                    if [np.nan, np.nan] in coordo and int(im[5:])not in im_prob_nb:
                        im_prob_nb.append(int(im[5:]))
            
            im_prob_nb = sorted(im_prob_nb)

            if len(im_prob_nb) > 0:
                txt = f"Numéros des images n'ayant pas {nb_marqueurs} marqueurs :\n"
                txt_multiline = ''
                i = len(txt)
                count = 0
                for im in im_prob_nb:
                    txt += f'{im}, '
                for n in range(len(txt)):
                    if txt[n]==',':
                        count+=1
                        if count == 20:
                            txt_multiline = txt[:n+1] + '\n' + txt[n+2:]
                        if count > 20 and count%20 == 0:
                            txt_multiline = txt_multiline[:n+1] + '\n' + txt_multiline[n+2:]
                if len(txt_multiline) > 1:
                    self.ids.im_prob_nb.text = txt_multiline[:-2]
                else:
                    self.ids.im_prob_nb.text = txt[:-2]
            elif len(im_prob_nb) == 0:
                self.ids.im_prob_nb.text = f'{nb_marqueurs} marqueurs détectés sur toutes les images !'
            return im_prob_nb
   
    # Fonction pour convertir la position touchée en coordonnées de marqueur, puis choisir l'action à exécuter (delete or add)
    def pos_marqueur(self, touch_pos):
        if self.ids.labelize_manual.state == 'normal':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            if 0.03*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.03*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.03)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Détermine s'il y a un marqueur à effacer ou si on en ajoute un manquant
                add_m = False
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 10 and abs(m_pos[1]-c[1]) < 10:
                        dict_coordo[f'image{image_nb}'].remove(c)
                        if f'image{image_nb}' in dict_coordo_labels_manual:
                            dict_coordo_labels_manual[f'image{image_nb}'] = {m: c for m, c in dict_coordo_labels_manual[f'image{image_nb}'].items() if c in dict_coordo[f'image{image_nb}']}
                        if labelize_extent == True:
                            self.extend_labelisation()
                        #self.labelize()
                        if self.ids.button_verif_continuity.state == 'down':
                            self.verif_continuity()
                        self.show_image()
                        add_m = False
                        break
                    else:
                        add_m = True
                if add_m or len(dict_coordo[f'image{image_nb}']) == 0:
                        self.add_marqueur(m_pos)
            else:
                pass

    # Ajout manuel d'un marqueur sur commande par un clic sur l'image
    def add_marqueur(self, m_pos):
        x = (m_pos[0]/im_dim[1]*(702/1960)+0.02)
        y = (0.85 - m_pos[1]/im_dim[0]*0.78)
        with self.canvas:
            Color(0,0,1,1)
            d = 5
            Ellipse(pos=(x - d/2, y - d/2), size=(d, d), group=u"new_mark")
        dict_coordo[f'image{image_nb}'].append([m_pos[0], m_pos[1]])
        
        #self.labelize()
        if labelize_extent == True:
            self.extend_labelisation()
        self.show_image()
    
    # Supprime les marqueurs qui n'ont pas été identifiés dans la prolongation de la labellisation manuelle
    def delete_by_continuity(self):
        for im in dict_coordo.keys():
            for c in dict_coordo[im]:
                if c not in dict_coordo_labels_manual[im].values():
                    dict_coordo[im].remove(c)
        self.show_image()

    # Ajoute des marqueurs manquants selon les splines d'interpolation
    def add_by_continuity(self):
        im_prob_nb = self.verif_nb()
        splines_smooth = self.interpolate_spline()
        for im in im_prob_nb:
            for l in labels:
                c = dict_coordo_labels_manual[f'image{im}'][l]
                if len(splines_smooth[l][0]) > 1:
                    c_interpolate = [float(splines_smooth[l][0][im-1]), float(splines_smooth[l][1][im-1])]
                    if c == [np.nan, np.nan]:
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
                    # Modification des marqueurs loin de leur courbe d'inteprolation
                    if (abs(c[0] - c_interpolate[0]) > 5 or abs(c[1] - c_interpolate[1]) > 5) and c in dict_coordo[f'image{im}']:
                        dict_coordo[f'image{im}'].remove(c)
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
                        print('modif interpolation')

        #self.labelize()
        self.show_image()
    
    # Fonction pour détecter les discontinuités (changement important de pente) ...de moins en moins utile avec l'inteprolation
    def verif_continuity(self):
        im_prob_continuity = {}
        if detection_eff:
            for im in range(2, images_total):
                coordo_prec = dict_coordo_labels_manual[f'image{im-1}']
                coordo_act = dict_coordo_labels_manual[f'image{im}']
                coordo_next = dict_coordo_labels_manual[f'image{im+1}']
                for label in labels:
                    if abs((coordo_next[label][0]-coordo_act[label][0])-(coordo_act[label][0]-coordo_prec[label][0])) > 10:
                        if im not in im_prob_continuity:
                            im_prob_continuity.update({im: [label]})
                        else:
                            im_prob_continuity[im].append(label)
                    if abs((coordo_next[label][1]-coordo_act[label][1])-(coordo_act[label][1]-coordo_prec[label][1])) > 10:
                        if im not in im_prob_continuity:
                            im_prob_continuity.update({im: [label]})
                        elif label not in im_prob_continuity[im]:
                            im_prob_continuity[im].append(label)

        return im_prob_continuity
    
    # Fonction pour définir une spline d'inteprolation pour chaque position x, y des marqueurs
    def interpolate_spline(self):
        x_axis = np.arange(images_total)
        x, y = {}, {}
        splines_smooth, spl = {}, {}
        for l in labels:
            y.update({l : [[], []]})
            x.update({l: []})
            spl.update({l: [[], []]})
            splines_smooth.update({l: [[], []]})
        for im, coordos in dict_coordo_labels_manual.items():
            for l, c in coordos.items():
                if not math.isnan(c[0]):
                    y[l][0].append(c[0])
                    y[l][1].append(c[1])
                    x[l].append(int(im[5:]))

        for l in labels:
            m = len(x[l])
            sm = m-math.sqrt(2*m)
            #w = np.ones(m) #poids de 1 à tous les points
            if m > 7:
                print(y[l][0], type(y[l][0][0]))
                y[l][0] = gaussian_filter1d(y[l][0], 3) #filtre les données avant interpolation
                y[l][1] = gaussian_filter1d(y[l][1], 3)

                spl[l][0] = splrep(x[l], y[l][0], k=3)
                spl[l][1] = splrep(x[l], y[l][1], k=3)
                splines_smooth[l][0] = splev(x_axis, spl[l][0], ext=3)
                splines_smooth[l][1] = splev(x_axis, spl[l][1], ext=3)   

        return splines_smooth

    # Graphiques des positions des marqueurs selon l'image, avec splines d'interpolation
    def graph_continuity(self):
        if detection_eff == True:
            xaxis = range(1, images_total+1)
            splines_smooth = self.interpolate_spline()
            fig, (ax1, ax2) = plt.subplots(2,1)
            colors = ['tab:orange', 'tab:red', 'tab:green', 'k', 'tab:blue', 'tab:purple', 'y', 'c', 'tab:gray', 'tab:pink']
            for m, color in zip(labels, colors[0:nb_marqueurs]):
                try:
                    plot_x = [c[m][0] for c in dict_coordo_labels_manual.values()]
                    plot_y = [c[m][1] for c in dict_coordo_labels_manual.values()]
                except KeyError:
                    continue
                ax1.scatter(xaxis, plot_x, s=2, label=m, c=color)
                ax1.plot(xaxis, splines_smooth[m][0], c=color)
                ax2.scatter(xaxis, plot_y, s=2, label=m, c=color)
                ax2.plot(xaxis, splines_smooth[m][1], c=color)
            ax1.legend(loc='center right', bbox_to_anchor=(1.13, -0.2), fontsize=9, frameon=False)
            ax1.set_title("Coordonnées des marqueurs selon l'image", fontsize=10)
            ax1.set_ylabel("Coordonnée en x", fontsize=9)
            ax2.set_ylabel("Coordonnée en y", fontsize=9)
            ax2.set_xlabel("Numéro de l'image", fontsize=9)
            #plt.savefig(r'C:\Users\LEA\Desktop\Poly\H2023\Projet 3\graph_continuity_1.png')
            self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
            plt.close()

        if self.ids.button_graph_continuity.state == 'normal':
            self.ids.graph.clear_widgets()

    # Fonction de labelisation par régions, fonctionne peu importe le nombre de marqueurs (était utilisé pour le Participant 1 à l'hiver)
    """ def labelize(self):
        global dict_coordo_labels1
        dict_coordo_labels1 = {}
        for im, coordo in dict_coordo.items():
            dict_coordo_labels1.update({im:{'C7':(np.nan, np.nan), 'G':(np.nan, np.nan), 'D':(np.nan, np.nan), 'T':(np.nan, np.nan), 'L':(np.nan, np.nan)}})
            for el in coordo:
                x = el[0]
                y = el[1]
                if 120 < x < 260 and 100 < y < 270:
                    dict_coordo_labels1[im]['G'] = (x,y)
                elif 260 < x < 400 and 100 < y < 270:
                    dict_coordo_labels1[im]['D'] = (x,y)
                elif 200 < x < 320 and 10 < y < 100:
                    dict_coordo_labels1[im]['C7'] = (x,y)
                elif 200 < x < 320 and 270 < y < 350:
                    dict_coordo_labels1[im]['T'] = (x,y)
                elif 200 < x < 320 and 350 < y < 480:
                    dict_coordo_labels1[im]['L'] = (x,y) """

    # Groupes de fonctions pour labellisation manuelle sur une image, puis étendue sur les autres par proximité
    # Détecte clic et associe marqueur à labelliser
    def labelize_manual(self, touch_pos):
        if self.ids.labelize_manual.state == 'down':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            # détecte si le clic est dans le cadre de l'image
            if 0.03*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.03*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.03)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Trouve le marqueur le plus près pour lui associer le label
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 20 and abs(m_pos[1]-c[1]) < 20:
                        # ajout d'un cercle bleu pa^le pour montrer qu'un marqueur est sélectionné
                        with self.canvas:
                            Color(171/255.0, 222/255.0, 231/255.0, .8)
                            d = 7
                            Ellipse(pos=(touch_pos[0] - d/2, touch_pos[1] - d/2), size=(d, d), group=u"label")
                        global m_to_label
                        m_to_label = c
                        break   
        else:
            pass
    
    # Entre le label du marqueur sélectionne dans le tableau et dans le dictionnaire, extend labelisation si tous les marqueurs labellisés
    def label_in(self):
        label = self.ids.label_input.text
        #self.ids.label_input.text = ''
        self.canvas.remove_group(u"label")
        global dict_coordo_labels_manual
        if f'image{image_nb}' in dict_coordo_labels_manual.keys():
            if label not in dict_coordo_labels_manual[f'image{image_nb}'] and m_to_label not in dict_coordo_labels_manual[f'image{image_nb}'].values():
                dict_coordo_labels_manual[f'image{image_nb}'].update({label : m_to_label})
            else:
                dict_coordo_labels_manual[f'image{image_nb}'][label] = m_to_label
        else:
            dict_coordo_labels_manual.update({f'image{image_nb}': {label : m_to_label}})

        if not labelize_extent:
            self.ids.grid.add_widget(Label(text=f'{label}', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text=f'({m_to_label[0]:.0f}, {m_to_label[1]:.0f})', color=(0,0,0,1)))
        if labelize_extent:
            self.show_image()
            
        print(dict_coordo_labels_manual[f'image{image_nb}'])
        
        if len(list(dict_coordo_labels_manual[f'image{image_nb}'].keys())) == nb_marqueurs:
            global labels
            labels = list(dict_coordo_labels_manual[f'image{image_nb}'].keys())
            
            self.extend_labelisation()
    
    # Étend la labellisation de la 1re image aux suivantes (réexécutée en cours de correction)
    def extend_labelisation(self):
        global labelize_extent
        labelize_extent = True
        for im in list(sorted(dict_coordo.keys(), key=lambda item : int(item[5:])))[1:]:
            num = int(im[5:])
            if im not in dict_coordo_labels_manual:
                dict_coordo_labels_manual.update({im:{}})
            # définit les références à partir des positions correspondantes sur les images précédentes
            for label in dict_coordo_labels_manual['image1'].keys():
                ref_prec = [0,0]
                if label in dict_coordo_labels_manual[im] and dict_coordo_labels_manual[im][label] != [np.nan, np.nan]:
                    continue
                else:
                    i = 1
                    while i < num:
                        if label in dict_coordo_labels_manual[f'image{num-i}'] and dict_coordo_labels_manual[f'image{num-i}'][label] != [np.nan, np.nan]:
                            ref = dict_coordo_labels_manual[f'image{num-i}'][label]
                            if num > 2:
                                j = i+1
                                while j <= (num-i):
                                    if label in dict_coordo_labels_manual[f'image{num-j}'] and dict_coordo_labels_manual[f'image{num-j}'][label] != [np.nan, np.nan]:
                                        ref_prec = dict_coordo_labels_manual[f'image{num-j}'][label]
                                        break
                                    else:
                                        j += 1
                            break
                        else:
                            i += 1
                    # cherche dans les marqueurs du dictionnaire de l'image actuelle s'il y en a un qui peut être labellisé (proche de ref ou ref_prec)
                    for coordos in dict_coordo[im]:
                        if coordos not in dict_coordo_labels_manual[im].values():
                            if (abs(coordos[0] - ref[0]) < 12 and abs(coordos[1] - ref[1]) < 10) or (abs(coordos[0] - ref_prec[0]) < 12 and abs(coordos[1] - ref_prec[1]) < 10):
                                dict_coordo_labels_manual[im][label] = coordos                            
                                break
                            elif ref_prec != [0,0] and -7 < (coordos[0] - (i*(ref[0]-ref_prec[0])/(j-i)+ref[0])) < 9 and -7 < (coordos[1] - (i*(ref[1]-ref_prec[1])/(j-i)+ref[1])) < 9:
                                dict_coordo_labels_manual[im][label] = coordos
                                break
                        else:
                            dict_coordo_labels_manual[im][label] = [np.nan, np.nan]
                        
                        
    # Fonction pour extraire les coordonnées (x,y,z) des marqueurs des fichiers _xyz_.raw
    def coordo_xyz_marqueurs(self):
        global dict_coordo_xyz
        dict_coordo_xyz = {}
        global save_xyz
        save_xyz = path+'/XYZ_converted/'
        # Lis les xyz.raw et crée les fichiers contenant les x,y,z des marqueurs
        if not 'XYZ_converted' in os.listdir(path):
            os.mkdir(save_xyz, )

        for key in dict_coordo_labels_manual.keys():
            dict_coordo[key] = list(dict_coordo_labels_manual[key].values())

        RRF.write_xyz_coordinates(path, dict_coordo, w1, w2, h1, h2)
        # Récupère les données des fichiers csv des coordonnées x,y,z des marqueurs
        for filename in os.listdir(save_xyz):
            index_XYZ = filename.find('_XYZ') + 5
            key = f'image{int(filename[index_XYZ:-4])+1}'
            with open(os.path.join(save_xyz, filename), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                dict_coordo_xyz.update({key : []})
                for row in reader:
                    row = [float(i) for i in row]
                    dict_coordo_xyz[key].append([row[1], row[0], row[2]])
    
    # Fonction de labelisation par tri (avec 5 marqueurs uniquement, selon coordonnées x,y,z)
    # Utilisée pour l'analyse
    def labelize_5(self):
        global dict_coordo_xyz_labels
        dict_coordo_xyz_labels = {}
        for im, coordos in dict_coordo_xyz.items():
            coordos_sorted_x = sorted(coordos, key=lambda tup: tup[0])
            dict_coordo_xyz_labels.update({im: {'G': coordos_sorted_x[0]}}) #plus petite valeur en x = gauche
            dict_coordo_xyz_labels[im].update({'D': coordos_sorted_x[-1]}) #plus grande valeur en x = droite
            del coordos_sorted_x[0]
            del coordos_sorted_x[-1]
            coordos_sorted_y = sorted(coordos_sorted_x, key=lambda tup: tup[1])
            dict_coordo_xyz_labels[im].update({'C': coordos_sorted_y[-1]}) #plus grande valeur en y = C7
            dict_coordo_xyz_labels[im].update({'T': coordos_sorted_y[-2]})
            dict_coordo_xyz_labels[im].update({'L': coordos_sorted_y[0]})
    
    def labelize_8(self):
        global dict_coordo_xyz_labels
        dict_coordo_xyz_labels = {}
        for im, coordos in dict_coordo_xyz.items():
            coordos_sorted_y = sorted(coordos, key=lambda tup: tup[1])
            dict_coordo_xyz_labels.update({im: {'C': coordos_sorted_y[-1]}}) #plus petite valeur en y = C7
            dict_coordo_xyz_labels[im].update({'T1': coordos_sorted_y[-2]})
            dict_coordo_xyz_labels[im].update({'L': coordos_sorted_y[2]})
            epines = coordos_sorted_y[0:2]
            epines = sorted(epines, key=lambda tup: tup[0])
            dict_coordo_xyz_labels[im].update({'IG': epines[0]})
            dict_coordo_xyz_labels[im].update({'ID': epines[1]})
            del coordos_sorted_y[0:3]
            del coordos_sorted_y[-2:]
            coordos_sorted_x = sorted(coordos_sorted_y, key=lambda tup: tup[0])
            dict_coordo_xyz_labels[im].update({'D': coordos_sorted_x[-1]}) #plus grande valeur en x = droite
            dict_coordo_xyz_labels[im].update({'G': coordos_sorted_x[0]})
            dict_coordo_xyz_labels[im].update({'T2': coordos_sorted_x[1]})

    def analyse(self):
        timer_debut_analyse = time.process_time_ns()
        im_prob_nb = self.verif_nb()
        if len(im_prob_nb) == 0 and self.ids.button_analyze.state == 'down':
            global analyse_eff
            analyse_eff = True
            if self.ids.check_new.state == 'down' or 'coordonnees_xyz.csv' not in os.listdir(path+'/Positions/'):
                self.coordo_xyz_marqueurs()
                if nb_marqueurs in [5, 6]:
                    self.labelize_5()
                if nb_marqueurs in [8, 9]:
                    self.labelize_8()
            if self.ids.button_analyze.state == 'down' or self.ids.check_open.state == 'down':
                global dict_metriques
                dict_metriques = {'angle_scap_vert' : [], 'angle_scap_prof': [], 'diff_dg': []}

                for im, coordo in dict_coordo_xyz_labels.items():
                    scap_y = np.degrees(np.arctan((coordo['D'][1] - coordo['G'][1])/(coordo['D'][0] - coordo['G'][0])))
                    dict_metriques['angle_scap_vert'].append(scap_y)
                    scap_z = np.degrees(np.arctan((coordo['G'][2] - coordo['D'][2])/(coordo['D'][0] - coordo['G'][0]))) #ajouter avec z
                    dict_metriques['angle_scap_prof'].append(scap_z)

                    # calcul distance horizontale entre marqueurs D/G et l'axe du rachis (x=ay+b)
                    a = (coordo['L'][0]-coordo['C'][0])/(coordo['L'][1]-coordo['C'][1])
                    b = coordo['C'][0] - a*coordo['C'][1]
                    x1 = coordo['G'][0]
                    x2 = (a*coordo['G'][1])+b
                    d1 = abs(x1 - x2) #distance entre G et l'axe de la colonne
                    x3 = coordo['D'][0]
                    x4 = a*coordo['D'][1]+b
                    d2 = abs(x3 - x4) #distance entre D et l'axe de la colonne
                    diff_d1d2 = abs(d1 - d2)
                    dict_metriques['diff_dg'].append(diff_d1d2)

                if nb_marqueurs in [5,6]:
                    dict_metriques.update(self.analyse_5())
                if nb_marqueurs in [8,9]:
                    dict_metriques.update(self.analyse_8())

                # Recherche des metriques optimales (et images associées)
                global min_metriques
                min_metriques = {}
                for key, vals in dict_metriques.items():
                    min = np.nanmin(np.absolute(vals))
                    if min in vals:
                        min_metriques.update({key: [vals.index(min), min]})
                    else:
                        min_metriques.update({key: [vals.index(-min), -min]})

                # Crée le graphique et l'affiche sur l'interface
                self.graph_analyze()
                self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
                plt.close()
                print(self.max_symmetry())

        if self.ids.button_analyze.state == 'normal':
            self.ids.graph.clear_widgets()

        self.show_image()

        timer_fin_analyse = time.process_time_ns()
        print(timer_debut_analyse, timer_fin_analyse)
        print(f'Temps calcul des métriques + graphiques :{timer_fin_analyse - timer_debut_analyse} ns')


    # Calcule des métriques pour 5 marqueurs
    def analyse_5(self):
        dict_metriques = {'angle_rachis': [], 'var_rachis': []}
        # Calcul des métriques d'analyse et ajout au dictionnaire de métriques
        for im, coordo in dict_coordo_xyz_labels.items():

            rachis_x = np.degrees(np.arctan((coordo['L'][0] - coordo['C'][0])/(coordo['L'][1] - coordo['C'][1])))
            dict_metriques['angle_rachis'].append(rachis_x)
            rachis_h = [coordo['L'][0], coordo['T'][0], coordo['C'][0]] #positions horizontales
            var_rachis = np.std(rachis_h)/abs(np.mean(rachis_h)) #devrait être nulle pour un alignement parfait
            dict_metriques['var_rachis'].append(var_rachis)

        return dict_metriques

    # Calcule des métriques pour 8 marqueurs
    def analyse_8(self):
        dict_metriques = {'angle_psis': [], 'scoliosis': []}
        for im, coordo in dict_coordo_xyz_labels.items():
            psis_y = np.degrees(np.arctan((coordo['ID'][1] - coordo['IG'][1])/(coordo['ID'][0] - coordo['IG'][0])))
            dict_metriques['angle_psis'].append(psis_y)
            
            a = np.sqrt((coordo['T1'][0]-coordo['T2'][0])**2+(coordo['T1'][1]-coordo['T2'][1])**2)
            b = np.sqrt((coordo['T2'][0]-coordo['L'][0])**2+(coordo['T2'][1]-coordo['L'][1])**2)
            c = np.sqrt((coordo['T1'][0]-coordo['T2'][0])**2+(coordo['T1'][1]-coordo['T2'][1])**2)
            scoliosis_angle = np.degrees(np.arccos((c**2+b**2-a**2)/(2*a*b)))
            dict_metriques['scoliosis'].append(scoliosis_angle)
        
        return dict_metriques

    # Calcule le score d'une métrique pour une image selon toutes les métriques de cette catégorie
    def map_metriques(self, m, metriques):
        pond = 20*(1 - abs(m)/np.max(np.absolute(metriques)))
        return pond
    
    # Calcule le score global de chaque image et trouve le maximum de symétrie atteint (score et #image)
    def max_symmetry(self):
        dict_metriques.update({'scores': np.zeros(images_total)})
        for metriques in dict_metriques.values():
            for i in range(len(metriques)):
                m = metriques[i]
                pond = self.map_metriques(m, metriques)
                dict_metriques['scores'][i] += pond
        max_sym = np.max(dict_metriques['scores'])
        max_sym_im = np.argmax(dict_metriques['scores'])+1

        self.ids.im_best.text = f'Image no {max_sym_im}'
        self.ids.sym_best.text = f'Score : {max_sym:.2f} %'

        return max_sym_im, max_sym

    def graph_analyze(self):
        # Graphiques des metriques calculées selon l'image
        xaxis = range(1, images_total+1)
        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2)

        ax1.plot(xaxis, dict_metriques['angle_scap_vert'], label='Hauteur')
        ax1.scatter(min_metriques['angle_scap_vert'][0], min_metriques['angle_scap_vert'][1], marker='*', c='r')
        ax1.plot(xaxis, dict_metriques['angle_scap_prof'], label='Profondeur')
        ax1.scatter(min_metriques['angle_scap_prof'][0], min_metriques['angle_scap_prof'][1], marker='*', c='g')
        ax1.legend(fontsize=7)
        ax1.set_title("Angles entre les scapulas", fontsize=9)
        ax1.set_ylabel('Angle (degrés)', fontsize=9)

        ax2.plot(xaxis, dict_metriques['diff_dg'])
        ax2.scatter(min_metriques['diff_dg'][0], min_metriques['diff_dg'][1], marker='*', c='r')
        ax2.set_title("|Distance rachis-G - Distance rachis-D|", fontsize=9)
        ax2.set_ylabel('Distance (mm)', fontsize=9)

        if nb_marqueurs in [5,6]:
            metrique_3 = 'angle_rachis'
            metrique_4 = 'var_rachis'
            ax3.set_title("Angle entre l'axe du rachis et la verticale", fontsize=9)
            ax3.set_ylabel('Angle (degrés)', fontsize=9)
            ax4.set_title('Écart-type (C, T, L) / Moyenne en x', fontsize=9)
            ax4.set_ylabel('Variabilité', fontsize=9)
        
        elif nb_marqueurs in[8,9]:
            metrique_3 = 'angle_psis'
            metrique_4 = 'scoliosis'
            ax3.set_title("Angle vertical entre les PSIS", fontsize=9)
            ax3.set_ylabel('Angle (degrés)', fontsize=9)
            ax4.set_title('Angle de scoliose', fontsize=9)
            ax4.set_ylabel('180 - Angle (degrés)', fontsize=9)

        ax3.plot(xaxis, dict_metriques[metrique_3])
        ax3.scatter(min_metriques[metrique_3][0], min_metriques[metrique_3][1], marker='*', c='r')
        ax3.set_xlabel("Numéro de l'image", fontsize=9)
        
        ax4.plot(xaxis, dict_metriques[metrique_4])
        ax4.scatter(min_metriques[metrique_4][0], min_metriques[metrique_4][1], marker='*', c='r')
        ax4.set_xlabel("Numéro de l'image", fontsize=9)
        
        plt.tight_layout()
    
    # Calculer les distances des marqueurs de chaque frame au gold frame
    def distance_to_gold(self):
        global gold_nb
        gold_nb = self.max_symmetry()[0]

        marker_array = np.zeros((len(os.listdir(path+'/Preprocessed/')), 8, 3))
    
        with open(path+'/Positions/coordonnees_xyz.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            j = 0
            for row in reader: #skip headline
                if j == 0:
                    entete = row[1::3]
                    labels = [e[:-2] for e in entete]
                elif j > 0:
                    im = int(row[0])
                    row = [float(i) for i in row[1:]]
                    row_by_label = [[row[3*i], row[3*i+1], row[3*i+2]] for i in range(0,int(len(row)/3))]
                    marker_array[im-1] = row_by_label
                j += 1

        distances = np.zeros((len(marker_array[:,0,0]), 8, 3))
        for i in range(len(marker_array[:,0,0])):
            distances[i] = np.subtract(marker_array[i], marker_array[gold_nb-1])
        
        return distances, labels
    
    # Afficher les marqueurs du gold frame
    def show_marqueurs_gold(self):
        for coordinates in dict_coordo[f'image{gold_nb}']:
            x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.03
            y = 0.85 - (coordinates[1]/im_dim[0])*0.78
            with self.canvas:
                Color(245/255,168/255,2/255,1)
                Line(circle=(self.width*x, self.height*y,6,0,360), width=1.1, group=u"circle") #(center_x, center_y, radius, angle_start, angle_end, segments)
    

    # Afficher les distances sur l'image actuelle, une fois l'analyse effectuée
    def show_distances(self):
        distances, labels = self.distance_to_gold()
        self.show_marqueurs_gold()

        self.ids.xyz_axis.size_hint = (.05, .09)
        self.ids.xyz_axis.source = 'xyz_axis.png'
        dist_act = {}
        for i, l in enumerate(labels):
            dist_act.update({l: distances[image_nb-1][i]})

        self.Distances = Widget()
        for l in labels:
            coordinates = dict_coordo_labels_manual[f'image{image_nb}'][l]
            x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.03
            y = 0.85 - (coordinates[1]/im_dim[0])*0.78
            self.dist_txt = Label(text=f'{l}\nX : {dist_act[l][1]:.1f}\nY : {dist_act[l][0]:.1f}\nZ : {dist_act[l][2]:.1f}',
                                fontsize='10sp', color=(1,1,1,1), size_hint=(.2, .2), pos=(self.width*x, self.height*y))
            self.Distances.add_widget(self.dist_txt)

        self.add_widget(self.Distances)

    # Sauvegarder les informations souhaitées selon ce qui est coché
    def to_save(self):
        timer_debut_save = time.process_time_ns()
      
        if self.ids.save_positions.state == 'down':
            if not 'landmarks' in os.listdir(path):
                os.mkdir(path+'/landmarks', )
            if not 'Positions' in os.listdir(path):
                os.mkdir(path+'\\Positions', )
            self.save_positions()
            
        if analyse_eff == True:
            if self.ids.save_metriques.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                if not 'Positions' in os.listdir(path):
                    os.mkdir(path+'\\Positions', )
                self.save_metriques()
            if self.ids.save_graph.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                self.save_graph_analyze()
        else:
            pass
        timer_fin_save = time.process_time_ns()
        print(timer_debut_save, timer_fin_save)
        print(f'Temps sauvegarde :{timer_fin_save - timer_debut_save} ns')
    
    # Crée un csv et y écrit les coordonnées x,y,z des 5 marqueurs selon le numéro de l'image
    def save_positions(self):
        save_pos = path+'\\Positions'
        print('writing to ' + save_pos+'/positions_corrigees.json')
        with open(save_pos+'\\positions_corrigees.json', 'w') as positions:
            json.dump(dict_coordo_labels_manual, positions)
    
    # Crée un csv et y écrit les métriques et le score global pour chaque image
    def save_metriques(self):
        save_pos = path+'/Positions'
        with open(save_pos+'/coordonnees_xyz.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            entete = ['image no']
            for l in dict_coordo_xyz_labels['image1'].keys():
                entete += [f'{l} x', f'{l} y', f'{l} z']
            writer.writerow(entete)
            for im, coordos in dict_coordo_xyz_labels.items():
                row = [im[5:]]
                for l in dict_coordo_xyz_labels['image1'].keys():
                    row += [coordos[l][0], coordos[l][1], coordos[l][2]]
                writer.writerow(row)

        save_met = path+'/Metriques/metriques.csv'
        with open(save_met, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['image no'] + list(dict_metriques.keys()))
            for i in range(images_total):
                row = [i+1]
                for metrique in dict_metriques.values():
                    row.append(metrique[i])
                writer.writerow(row)
                i += 1
    
    # Enregistre le graphique des métriques sous format png
    def save_graph_analyze(self):
        self.graph_analyze()
        plt.savefig(path+'\\Metriques\\graph_analyze.png')
        plt.close()

        
class Interface(App):
    def build(self):
        return MyApp()

if __name__=='__main__':
    Interface().run()
