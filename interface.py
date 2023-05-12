from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color, Line, Ellipse
from kivy.uix.label import Label
import os
import json
import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import read_raw_file as RRF
import marker_detection

class MyApp(Widget):
    Window.maximize()
    Builder.load_file('design_interface.kv')
    Config.set('graphics', 'width', '1920')
    Config.set('graphics', 'height', '1080')

    global path
    path = ''
    global analyse_eff
    analyse_eff = False
    global detection_eff
    detection_eff = False
    global labelize_extent
    labelize_extent = False

    def press_color(self):
        self.background_normal = ''
        self.background_color = (100/255.0, 197/255.0, 209/255.0, 1)

    # Fonction pour sélectionner le répertoire .raw, puis définir le nombre d'images
    def im_select(self):
        print(self.width, self.height)
        timer_debut_im = time.process_time_ns()
        
        global path
        path = self.ids.path_input.text
        save_path = path+'/intensity/'
        save_path_xyz = path+'/xyz/'
        global save_path_im
        save_path_im = path+'/Preprocessed'
        try: # si chemin entré valide
            # Crée les répertoires pour images converties et prétraitées
            if not 'intensity' in os.listdir(path):
                os.mkdir(save_path, )
            if not 'landmarks' in os.listdir(path):
                os.mkdir(path+'/landmarks/')
            if not 'xyz' in os.listdir(path):
                os.mkdir(save_path_xyz)
            if not 'Preprocessed' in os.listdir(path):
                os.mkdir(save_path_im, )
            if len(os.listdir(save_path)) == 0:
                RRF.read_raw_intensity_frames(path, save_path)
            if len(os.listdir(save_path_im)) == 0:
                for name in os.listdir(save_path):
                    frame_display, preprocessed_frame = marker_detection.preprocess(cv2.imread(os.path.join(save_path, name)))
                    cv2.imwrite(os.path.join(save_path_im, name), preprocessed_frame)
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
            
            # Lance la détection des marqueurs
            self.detect_marqueurs()

            # Initie la variable pour numéro de l'image affichée à 1 pour voir la première image
            global image_nb
            image_nb = 1
            self.show_image()

            timer_fin_im = time.process_time_ns()
            print(timer_debut_im, timer_fin_im)
            print(f'Temps création + affichage images : {timer_fin_im - timer_debut_im}')

        except FileNotFoundError:
            self.ids.label_ready.text = "Le chemin entré est introuvable. Essayez à nouveau."

    def nb_marqueurs_input(self):
        global nb_marqueurs
        nb_marqueurs = int(self.ids.marq_nb_input.text)
        self.ids.grid.size_hint = (.22, .04 + .03*nb_marqueurs)
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
        if not analyse_eff:
            '''
            # Affiche les coordonnées des marqueurs de l'image actuelle dans le tableau si détectés, sinon case vide
            if dict_coordo_labels1[f'image{image_nb}']['C7'] != (np.nan, np.nan):
                self.ids.coordo_1.text = f"({dict_coordo_labels1[f'image{image_nb}']['C7'][0]:.0f}, {dict_coordo_labels1[f'image{image_nb}']['C7'][1]:.0f})"
            if dict_coordo_labels1[f'image{image_nb}']['C7'] == (np.nan, np.nan):
                self.ids.coordo_1.text = ''
            if dict_coordo_labels1[f'image{image_nb}']['G'] != (np.nan, np.nan):
                self.ids.coordo_2.text = f"({dict_coordo_labels1[f'image{image_nb}']['G'][0]:.0f}, {dict_coordo_labels1[f'image{image_nb}']['G'][1]:.0f})"
            if dict_coordo_labels1[f'image{image_nb}']['G'] == (np.nan, np.nan):
                self.ids.coordo_2.text = ''
            if dict_coordo_labels1[f'image{image_nb}']['D'] != (np.nan, np.nan):
                self.ids.coordo_3.text = f"({dict_coordo_labels1[f'image{image_nb}']['D'][0]:.0f}, {dict_coordo_labels1[f'image{image_nb}']['D'][1]:.0f})"
            if dict_coordo_labels1[f'image{image_nb}']['D'] == (np.nan, np.nan):
                self.ids.coordo_3.text = ''
            if dict_coordo_labels1[f'image{image_nb}']['T'] != (np.nan, np.nan):
                self.ids.coordo_4.text = f"({dict_coordo_labels1[f'image{image_nb}']['T'][0]:.0f}, {dict_coordo_labels1[f'image{image_nb}']['T'][1]:.0f})"
            if dict_coordo_labels1[f'image{image_nb}']['T'] == (np.nan, np.nan):
                self.ids.coordo_4.text = ''
            if dict_coordo_labels1[f'image{image_nb}']['L'] != (np.nan, np.nan):
                self.ids.coordo_5.text = f"({dict_coordo_labels1[f'image{image_nb}']['L'][0]:.0f}, {dict_coordo_labels1[f'image{image_nb}']['L'][1]:.0f})"
            if dict_coordo_labels1[f'image{image_nb}']['L'] == (np.nan, np.nan):
                self.ids.coordo_5.text = ''
            '''
        
        # Affiche les marqueurs si bouton activé
        if self.ids.button_showmarks.state == 'down':
            self.show_marqueurs()
        # Tag marqueurs discontinus si présents dans l'image actuelle
        if self.ids.button_verif_continuity.state == 'down':
            im_prob_continuity = self.continuity()
            for el in im_prob_continuity:
                if image_nb == el:
                    self.ids.rep_continuity.text = 'Marqueur(s) à vérifier :\n'
                    for m in im_prob_continuity[image_nb]:
                        self.ids.rep_continuity.text += f'{m}, '
                    self.ids.rep_continuity.text = self.ids.rep_continuity.text[:-2]

        # Affiche positions dans le tableau après labellisation manuelle
        if labelize_extent == True:
            for val in dict_coordo_labels_manual[f'image{image_nb}'].values():
                self.ids.grid.clear_widgets()
                self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
                self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
                for dict in dict_coordo_labels_manual.values():
                    for l in dict.keys():
                        self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                        c = dict_coordo_labels_manual[f'image{image_nb}'][l]
                        self.ids.grid.add_widget(Label(text=f'({c[0]:.0f}, {c[1]:.0f})', color=(0,0,0,1)))
                    break
        
        '''
        # Actualisation tableau de coordonnées avec coordos x,y,z si analyse effectuée
        if analyse_eff:
            self.ids.entete_coordos.text = 'Coordonnées (x,y,z)'
            d_im = dict_coordo_xyz_labels[f'image{image_nb}']
            self.ids.coordo_1.text = f"({d_im['C7'][0]:.0f}, {d_im['C7'][1]:.0f}, {d_im['C7'][2]:.0f})"
            self.ids.coordo_2.text = f"({d_im['G'][0]:.0f}, {d_im['G'][1]:.0f}, {d_im['G'][2]:.0f})"
            self.ids.coordo_3.text = f"({d_im['D'][0]:.0f}, {d_im['D'][1]:.0f}, {d_im['D'][2]:.0f})"
            self.ids.coordo_4.text = f"({d_im['T'][0]:.0f}, {d_im['T'][1]:.0f}, {d_im['T'][2]:.0f})"
            self.ids.coordo_5.text = f"({d_im['L'][0]:.0f}, {d_im['L'][1]:.0f}, {d_im['L'][2]:.0f})"
            self.ids.origine.text = ''
            self.ids.width.text = ''
            self.ids.height.text = ''
        '''
        
        # Efface les marques de labellisation manuelle (ronds bleus) si désactivé
        if self.ids.labelize_manual.state == 'normal':
            self.canvas.remove_group(u"label")
        
    # Fonction pour détecter les marqueurs de toutes les images du répertoire
    def detect_marqueurs(self):
        timer_debut_detection = time.process_time_ns()
        global detection_eff
        
        if len(path) > 1:
            # Détection avec algo si mode Nouveau
            if self.ids.check_new.state == 'down' and len(os.listdir(path+'/landmarks/')) == 0:
                marker_detection.annotate_frames(path)
            #i = 0
            for filename in os.listdir(path+'/landmarks/'):
                coordinates = []
                keypoints = []
                # image = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[i]))
                with open(os.path.join(path+'/landmarks/', filename), 'r') as f:
                    for line in f.readlines():
                        keypoints.append(line.split(' '))
                
                for n in range(len(keypoints)):
                    coordo_xy = [float(keypoints[n][0]), float(keypoints[n][1])]
                    coordinates.append(coordo_xy)
                i = int(filename[5:-14])
                dict_coordo[f'image{i+1}'] = coordinates
                #i += 1
            self.labelize()
            detection_eff = True
            
        # Détection avec algo si mode Nouveau
            # if self.ids.check_new.state == 'down':
            #     for i in range(images_total):
            #         coordinates = []
            #         image = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[i]))
            #         keypoints = marker_detection.detect_markers(image)
            #         for n in range(len(keypoints)):
            #             coordo_xy = [keypoints[n].pt[0], keypoints[n].pt[1]]
            #             coordinates.append(coordo_xy)
            #         dict_coordo[f'image{i+1}'] = coordinates
            #     self.labelize()
            #     detection_eff = True
            global im_dim
            im_dim = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[0])).shape
            # Va chercher les positions corrigées enregistrées si mode Ouvrir
            if self.ids.check_open.state == 'down':
                global analyse_eff
                analyse_eff = 'Metriques' in os.listdir(path+'') #Analyse effectuée (et utilisable) si métriques enregistrées
                if 'landmarks' in os.listdir(path+''):
                    # Recrée dictionnaire de positions avec labels
                    global dict_coordo_labels1
                    jsonfile = open(path+'/landmarks/positions_corrigees.json')
                    dict_coordo_labels1 = json.load(jsonfile)
                    for key, marqueurs in dict_coordo_labels1.items():
                        dict_coordo.update({key:[]})
                        for m in marqueurs.values():
                            dict_coordo[key].append(m)
                    # Recrée dictionnaire de coordonnées x,y,z
                    global dict_coordo_xyz_labels
                    dict_coordo_xyz_labels = {}
                    with open(path+'/landmarks/coordonnees_xyz.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        next(reader)
                        for row in reader: #skip headline
                            key = f'image{row[0]}'
                            row = [float(i) for i in row[1:]]
                            C7 = [row[0], row[1], row[2]]
                            G = [row[3], row[4], row[5]]
                            D = [row[6], row[7], row[8]]
                            T = [row[9], row[10], row[11]]
                            L = [row[12], row[13], row[14]]
                            dict_coordo_xyz_labels.update({key: {'C7':C7, 'G':G, 'D':D, 'T':T, 'L':L}})
                detection_eff = True
        timer_fin_detection = time.process_time_ns()
        print(timer_debut_detection, timer_fin_detection)
        print(f'Temps détection des marqueurs :{timer_fin_detection - timer_debut_detection}')

    # Fonction pour afficher les marqueurs de l'image actuelle
    def show_marqueurs(self):
        # Affichage des marqueurs si bouton activé
        if detection_eff == True:
            # key_points = detect_markers(preprocessed_frame)
            # im_with_key_points = cv2.drawKeypoints(frame_display, key_points, np.array([]), (0, 255, 0),
            #                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
                    im_prob_nb.append([int(im[5:]), len(coordo)]) #ajoute l'image et le nombre de marqueurs détectés à la liste
                    im_prob_nb = sorted(im_prob_nb)
            if len(im_prob_nb) > 0:
                txt = f"Numéros des images n'ayant pas {nb_marqueurs} marqueurs :\n"
                txt_multiline = ''
                i = len(txt)
                count = 0
                for im, nb in im_prob_nb:
                    txt += f'{im}, '
                for n in range(len(txt)):
                    if txt[n]==',':
                        count+=1
                        if count == 14:
                            txt_multiline = txt[:n+1] + '\n' + txt[n+2:]
                        if count > 14 and count%14 == 0:
                            txt_multiline = txt_multiline[:n+1] + '\n' + txt_multiline[n+2:]
                if len(txt_multiline) > 1:
                    self.ids.im_prob_nb.text = txt_multiline[:-2]
                else:
                    self.ids.im_prob_nb.text = txt[:-2]
            elif len(im_prob_nb) == 0:
                self.ids.im_prob_nb.text = f'{nb_marqueurs} marqueurs détectés sur toutes les images !'
            return im_prob_nb
   
    # Fonction pour convertir la position touchée en coordonnées de marqueur, puis choisir l'action à exécuter
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
                    if abs(m_pos[0]-c[0]) < 15 and abs(m_pos[1]-c[1]) < 15:
                        dict_coordo[f'image{image_nb}'].remove(c)
                        if labelize_extent == True:
                            self.extend_labelisation()
                        self.labelize()
                        if self.ids.button_verif_nb.state == 'down':
                            self.verif_nb()
                        if self.ids.button_verif_continuity.state == 'down':
                            self.continuity()
                        self.show_image()
                        add_m = False
                        break
                    else:
                        add_m = True
                if add_m:
                        self.add_marqueur(m_pos)
            else:
                pass

    def add_marqueur(self, m_pos):
        x = (m_pos[0]/im_dim[1]*(702/1960)+0.02)
        y = (0.85 - m_pos[1]/im_dim[0]*0.78)
        with self.canvas:
            Color(0,0,1,1)
            d = 5
            Ellipse(pos=(x - d/2, y - d/2), size=(d, d), group=u"new_mark")
        dict_coordo[f'image{image_nb}'].append([m_pos[0], m_pos[1]])
        
        if self.ids.button_verif_nb.state == 'down':
            self.verif_nb()
        if self.ids.button_verif_continuity.state == 'down':
            self.continuity()
        self.labelize()
        if labelize_extent == True:
            self.extend_labelisation()
        self.show_image()
    
    def delete_by_continuity(self):
        for im in dict_coordo.keys():
            for c in dict_coordo[im]:
                if c not in dict_coordo_labels_manual[im].values():
                    dict_coordo[im].remove(c)
        self.verif_nb()
        self.show_image()

    def add_by_continuity(self):
        im_prob_nb = self.verif_nb()
        im_prob = [im for [im,nb] in im_prob_nb]
        splines = self.interpolate_spline()
        for im in im_prob:
            for l in labels:
                if dict_coordo_labels_manual[f'image{im}'][l] == [np.nan, np.nan]:
                    c = [float(splines[l][0](im)), float(splines[l][1](im))]
                    dict_coordo[f'image{im}'].append(c)
                    dict_coordo_labels_manual[f'image{im}'][l] = c
                    print(dict_coordo_labels_manual[f'image{im}'][l])

        '''
        for im, nb in im_prob_nb:
            if im-1 not in im_prob and im+1 not in im_prob and nb == 4:
                index_nan = list(dict_coordo_labels1[f'image{im}'].values()).index((np.nan, np.nan))
                m_manquant = list(dict_coordo_labels1[f'image{im}'].keys())[index_nan]
                x = (dict_coordo_labels1[f'image{im+1}'][m_manquant][0] + dict_coordo_labels1[f'image{im-1}'][m_manquant][0])/2
                y = (dict_coordo_labels1[f'image{im+1}'][m_manquant][1] + dict_coordo_labels1[f'image{im-1}'][m_manquant][1])/2
                dict_coordo[f'image{im}'].append([x,y]) #ajout marqueur par interpolation linéaire avec images précédente et suivante
        '''
        self.verif_nb()
        #self.labelize()
        self.show_image()
    
    def continuity(self):
        im_prob_continuity = {}
        if detection_eff:
            for im in range(2, images_total):
                coordo_prec = dict_coordo_labels_manual[f'image{im-1}']
                coordo_act = dict_coordo_labels_manual[f'image{im}']
                coordo_next = dict_coordo_labels_manual[f'image{im+1}']
                for label in labels:
                    if abs((coordo_next[label][0]-coordo_act[label][0])-(coordo_act[label][0]-coordo_prec[label][0])) > 5:
                        if im not in im_prob_continuity:
                            im_prob_continuity.update({im: [label]})
                        else:
                            im_prob_continuity[im].append(label)
                    if abs((coordo_next[label][1]-coordo_act[label][1])-(coordo_act[label][1]-coordo_prec[label][1])) > 5:
                        if im not in im_prob_continuity:
                            im_prob_continuity.update({im: [label]})
                        elif label not in im_prob_continuity[im]:
                            im_prob_continuity[im].append(label)

        return im_prob_continuity
    
    def interpolate_spline(self):
        x_axis = np.arange(images_total)
        x = {}
        y = {}
        splines = {}
        for l in labels:
            y.update({l : [[], []]})
            x.update({l: []})
            splines.update({l: [[], []]})
        for im, coordos in dict_coordo_labels_manual.items():
            for l, c in coordos.items():
                if not math.isnan(c[0]):
                    y[l][0].append(c[0])
                    y[l][1].append(c[1])
                    x[l].append(im[5:])
        print(y)
        fig, ax = plt.subplots()
        for l in labels:
            cs_x = CubicSpline(x[l], y[l][0])
            cs_y = CubicSpline(x[l], y[l][1])
            splines[l][0] = cs_x
            splines[l][1] = cs_y
            '''
            ax.plot(x_axis, cs_x(x_axis), label=f'{l} x')
            ax.plot(x_axis, cs_y(x_axis), label=f'{l} y')
        ax.legend()
        plt.savefig(path+'/interpolate/test3.png')
        plt.close()
        '''
        return splines


    def graph_continuity(self):
        # Graphiques des positions des marqueurs selon l'image
        if detection_eff == True:
            xaxis = range(1, images_total+1)
            fig, (ax1, ax2) = plt.subplots(2,1)
            colors = ['tab:orange', 'tab:red', 'tab:green', 'k', 'tab:blue', 'tab:purple', 'y', 'c']
            for m, color in zip(labels, colors[0:nb_marqueurs]):
                plot_x = []
                plot_y = []
                for coordo in dict_coordo_labels_manual.values():
                    plot_x.append(coordo[m][0])
                    plot_y.append(coordo[m][1])
                ax1.scatter(xaxis, plot_x, s=2, label=m, c=color)
                ax2.scatter(xaxis, plot_y, s=2, label=m, c=color)
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

    # Fonction de labelisation par régions, fonctionne peu importe le nombre de marqueurs
    def labelize(self):
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
                    dict_coordo_labels1[im]['L'] = (x,y)

    # Fonction pour labellisation manuelle sur une image, puis étendue sur les autres par proximité
    def labelize_manual(self, touch_pos):
        if self.ids.labelize_manual.state == 'down':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            if 0.03*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.03*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.03)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Trouve le marqueur le plus près pour lui associer le label
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 30 and abs(m_pos[1]-c[1]) < 30:
                        with self.canvas:
                            Color(171/255.0, 222/255.0, 231/255.0, .8)
                            d = 7
                            Ellipse(pos=(touch_pos[0] - d/2, touch_pos[1] - d/2), size=(d, d), group=u"label")
                        global m_to_label
                        m_to_label = c
                        break   
        else:
            pass
    
    global dict_coordo_labels_manual
    dict_coordo_labels_manual = {}

    def label_in(self):
        label = self.ids.label_input.text
        #self.ids.label_input.text = ''
        self.canvas.remove_group(u"label")
        global dict_coordo_labels_manual
        if f'image{image_nb}' in dict_coordo_labels_manual.keys():
            dict_coordo_labels_manual[f'image{image_nb}'].update({label : m_to_label})
        else:
            dict_coordo_labels_manual.update({f'image{image_nb}': {label : m_to_label}})
        self.ids.grid.add_widget(Label(text=f'{label}', color=(0,0,0,1)))
        self.ids.grid.add_widget(Label(text=f'({m_to_label[0]:.0f}, {m_to_label[1]:.0f})', color=(0,0,0,1)))

        if len(list(dict_coordo_labels_manual[f'image{image_nb}'].keys())) == nb_marqueurs:
            global labels
            labels = []
            dict_labels = dict_coordo_labels_manual[f'image{image_nb}'].keys()
            for l in dict_labels:
                labels.append(l)
                #self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                c = dict_coordo_labels_manual[f'image{image_nb}'][l]
                #self.ids.grid.add_widget(Label(text=f'({str(int(c[0]))}, {str(int(c[1]))})', color=(0,0,0,1)))
            self.extend_labelisation()
    
    def extend_labelisation(self):
        global labelize_extent
        labelize_extent = True
        for im in list(sorted(dict_coordo.keys(), key=lambda item : int(item[5:])))[1:]:
            num = int(im[5:])
            print(num)
            dict_coordo_labels_manual.update({im:{}})
            for label in dict_coordo_labels_manual['image1'].keys():
                i = 1
                while i < num:
                    print('i=', i)
                    if label in dict_coordo_labels_manual[f'image{num-i}'] and dict_coordo_labels_manual[f'image{num-i}'][label] != [np.nan, np.nan]:
                        ref = dict_coordo_labels_manual[f'image{num-i}'][label]
                        break
                    else:
                        i += 1
                print(ref)
                for coordos in dict_coordo[im]:
                    dict_coordo_labels_manual[im][label] = [np.nan, np.nan]
                    if abs(coordos[0] - ref[0]) < 30 and abs(coordos[1] - ref[1]) < 10:
                        print('=', coordos)
                        dict_coordo_labels_manual[im][label] = coordos
                        break
    
    # Fonction pour extraire les coordonnées (x,y,z) des marqueurs des fichiers _xyz_.raw
    def coordo_xyz_marqueurs(self):
        global dict_coordo_xyz
        dict_coordo_xyz = {}
        global save_xyz
        save_xyz = path+'/XYZ_converted'
        # Lis les xyz.raw et crée les fichiers contenant les x,y,z des marqueurs
        if not 'XYZ_converted' in os.listdir(path):
            os.mkdir(save_xyz, )
        RRF.read_raw_xyz_frames(path, dict_coordo)
        # Récupère les données des fichiers csv des coordonnées x,y,z des marqueurs
        for filename, key in zip(os.listdir(save_xyz), dict_coordo.keys()):
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
            dict_coordo_xyz_labels[im].update({'D': coordos_sorted_x[4]}) #plus grande valeur en x = droite
            del coordos_sorted_x[0]
            del coordos_sorted_x[-1]
            coordos_sorted_y = sorted(coordos_sorted_x, key=lambda tup: tup[1])
            dict_coordo_xyz_labels[im].update({'C7': coordos_sorted_y[2]}) #plus grande valeur en y = C7
            dict_coordo_xyz_labels[im].update({'T': coordos_sorted_y[1]})
            dict_coordo_xyz_labels[im].update({'L': coordos_sorted_y[0]})

    def analyse(self):
        timer_debut_analyse = time.process_time_ns()
        im_prob_nb = self.verif_nb()
        if len(im_prob_nb) == 0 and self.ids.button_analyze.state == 'down':
            global analyse_eff
            analyse_eff = True
            self.coordo_xyz_marqueurs()
            self.labelize_5()
            if self.ids.button_analyze.state == 'down':
                global dict_metriques
                dict_metriques = {'angle_scap_vert' : [], 'angle_scap_prof': [], 'diff_dg': [], 'angle_rachis': [], 'var_rachis': []}
                # Calcul des métriques d'analyse et ajout au dictionnaire de métriques
                for im, coordo in dict_coordo_xyz_labels.items():
                    scap_y = np.degrees(np.arctan((coordo['D'][1] - coordo['G'][1])/(coordo['D'][0] - coordo['G'][0])))
                    dict_metriques['angle_scap_vert'].append(scap_y)
                    scap_z = np.degrees(np.arctan((coordo['G'][2] - coordo['D'][2])/(coordo['D'][0] - coordo['G'][0]))) #ajouter avec z
                    dict_metriques['angle_scap_prof'].append(scap_z)

                    rachis_x = np.degrees(np.arctan((coordo['L'][0] - coordo['C7'][0])/(coordo['L'][1] - coordo['C7'][1])))
                    dict_metriques['angle_rachis'].append(rachis_x)
                    rachis_h = [coordo['L'][0], coordo['T'][0], coordo['C7'][0]] #positions horizontales
                    var_rachis = np.std(rachis_h)/abs(np.mean(rachis_h)) #devrait être nulle pour un alignement parfait
                    dict_metriques['var_rachis'].append(var_rachis)

                    # calcul distance horizontale entre marqueurs D/G et l'axe du rachis (x=ay+b)
                    a = (coordo['L'][0]-coordo['C7'][0])/(coordo['L'][1]-coordo['C7'][1])
                    b = coordo['C7'][0] - a*coordo['C7'][1]
                    x1 = coordo['G'][0]
                    x2 = (a*coordo['G'][1])+b
                    d1 = abs(x1 - x2) #distance entre G et l'axe de la colonne
                    x3 = coordo['D'][0]
                    x4 = a*coordo['D'][1]+b
                    d2 = abs(x3 - x4) #distance entre D et l'axe de la colonne
                    diff_d1d2 = abs(d1 - d2)
                    dict_metriques['diff_dg'].append(diff_d1d2)

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

        timer_fin_analyse = time.process_time_ns()
        print(timer_debut_analyse, timer_fin_analyse)
        print(f'Temps calcul des métriques + graphiques :{timer_fin_analyse - timer_debut_analyse}')
              
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

        ax2.plot(xaxis, dict_metriques['angle_rachis'])
        ax2.scatter(min_metriques['angle_rachis'][0], min_metriques['angle_rachis'][1], marker='*', c='r')
        ax2.set_title("Angle entre l'axe du rachis et la verticale", fontsize=9)
        ax2.set_ylabel('Angle (degrés)', fontsize=9)

        ax3.plot(xaxis, dict_metriques['diff_dg'])
        ax3.scatter(min_metriques['diff_dg'][0], min_metriques['diff_dg'][1], marker='*', c='r')
        ax3.set_title("|Distance rachis-G - Distance rachis-D|", fontsize=9)
        ax3.set_xlabel("Numéro de l'image", fontsize=9)
        ax3.set_ylabel('Distance (mm)', fontsize=9)

        ax4.plot(xaxis, dict_metriques['var_rachis'])
        ax4.scatter(min_metriques['var_rachis'][0], min_metriques['var_rachis'][1], marker='*', c='r')
        ax4.set_title('Écart-type (C7, T, L) / Moyenne en x', fontsize=9)
        ax4.set_xlabel("Numéro de l'image", fontsize=9)
        ax4.set_ylabel('Variabilité', fontsize=9)
        
        plt.tight_layout()
    
    # Sauvegarder les informations souhaitées selon ce qui est coché
    def to_save(self):
        timer_debut_save = time.process_time_ns()
      
        if self.ids.save_positions.state == 'down':
            if not 'landmarks' in os.listdir(path):
                os.mkdir(path+'/landmarks', )
            self.save_positions()
            
        if self.ids.save_positions.state == 'down':
            if not 'Positions' in os.listdir(path):
                os.mkdir(path+'\\Positions', )
            self.save_positions()
            
        if analyse_eff == True:
            if self.ids.save_metriques.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                self.save_metriques()
            if self.ids.save_graph.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                self.save_graph_analyze()
        else:
            pass
        timer_fin_save = time.process_time_ns()
        print(timer_debut_save, timer_fin_save)
        print(f'Temps sauvegarde :{timer_fin_save - timer_debut_save}')
    
    # Crée un csv et y écrit les coordonnées x,y,z des 5 marqueurs selon le numéro de l'image
    def save_positions(self):
        save_pos1 = path+'/landmarks'
        print('writing to ' + save_pos1+'/positions_corrigees.json')
        with open(save_pos1+'/positions_corrigees.json', 'w') as positions:
            json.dump(dict_coordo_labels1, positions)
        save_pos2 = path+'\\Positions'
        with open(save_pos2+'\\positions_corrigees.json', 'w') as positions:
            json.dump(dict_coordo_labels_manual, positions)
    
    # Crée un csv et y écrit les métriques et le score global pour chaque image
    def save_metriques(self):
        save_pos = path+'/landmarks'
        with open(save_pos+'/coordonnees_xyz.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['image no', 'C7 x', 'C7 y', 'C7 z', 'G x', 'G y', 'G z', 'D x', 'D y', 'D z',
                             'T x', 'T y', 'T z', 'L x', 'L y', 'L z'])
            for im, coordos in dict_coordo_xyz_labels.items():
                writer.writerow([im[5:], coordos['C7'][0], coordos['C7'][1], coordos['C7'][2],
                                        coordos['G'][0], coordos['G'][1], coordos['G'][2],
                                        coordos['D'][0], coordos['D'][1], coordos['D'][2],
                                        coordos['T'][0], coordos['T'][1], coordos['T'][2],
                                        coordos['L'][0], coordos['L'][1], coordos['L'][2]])
        save_met = path+'/Metriques/metriques.csv'
        with open(save_met, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['image no'] + list(dict_metriques.keys()))
            for i in range(images_total):
                writer.writerow([i+1, dict_metriques['angle_scap_vert'][i], dict_metriques['angle_scap_prof'][i],
                                 dict_metriques['diff_dg'][i],dict_metriques['angle_rachis'][i],
                                 dict_metriques['var_rachis'][i], dict_metriques['scores'][i]])
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
