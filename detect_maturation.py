import cv2
import numpy as np

def detect_artificial_ripening(img):
    """
    Retourne un score entre 0 et 100 indiquant la probabilité
    de mûrissement artificiel.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    sat_mean = np.mean(s)
    sat_score = np.interp(sat_mean, [60, 200], [0, 100])  
    sat_score = np.clip(sat_score, 0, 100)

    # pour les Couleur trop homogène = mûri artificiellement
    sat_std = np.std(s)
    homog_score = np.interp(sat_std, [40, 10], [0, 100])  
    homog_score = np.clip(homog_score, 0, 100)

    #Détection de zones vertes persistantes
    green_mask = (h > 35) & (h < 85)
    green_ratio = np.sum(green_mask) / img.size * 100
    green_score = np.interp(green_ratio, [0, 10], [100, 0])
    green_score = np.clip(green_score, 0, 100)

    # Score final pondéré
    final_score = (sat_score * 0.4) + (homog_score * 0.4) + (green_score * 0.2)
    return int(final_score)
