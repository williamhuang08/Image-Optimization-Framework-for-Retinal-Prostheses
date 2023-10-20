import pulse2percept as p2p

import numpy as np
import cv2

import pulse2percept as p2p
import numpy as np


logo = p2p.stimuli.LogoBVL()

#logo = cv2.imread("sunflower.jpg")

print(logo)
logo_gray = logo.invert().rgb2gray()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
logo.plot(ax=ax1)
logo_gray.plot(ax=ax2)

logo_gray.scale(0.75).rotate(30).trim().plot()

logo_edge = logo_gray.filter('scharr')

from skimage.morphology import dilation
logo_dilate = logo_edge.apply(dilation)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
# Edges extracted with the Scharr operator:
logo_edge.plot(ax=ax1)
# Edges thickened with dilation:
logo_dilate.plot(ax=ax2)

logo_dilate.save('dilated_logo.png')

# Simulate only what we need (14x14 deg sampled at 0.1 deg):
model = p2p.models.ScoreboardModel(xrange=(-7, 7), yrange=(-7, 7), xystep=0.1)
model.build()

from pulse2percept.implants import ArgusII
implant = ArgusII()

# Show the visual field we're simulating (dashed lines) atop the implant:
#model.plot()
implant.plot()

implant.stim = logo_gray.resize(implant.shape)

percept_gray = model.predict_percept(implant)

implant.stim = logo_dilate.trim().resize(implant.shape)
percept_dilate = model.predict_percept(implant)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
percept_gray.plot(ax=ax1)
percept_dilate.plot(ax=ax2)

implant.stim = logo_dilate.trim().resize(implant.shape).encode(amp_range=(0,20),pulse=(10, 1, 0.2, stim_dur=200))

PulseTrain(20, implant.stim, stim_dur=40).plot()
biphasic_ramp = Stimulus(np.concatenate((implant.stim.data, -implant.stim.data), axis=1),
                         time=np.concatenate((implant.stim.time, implant.stim.time + 5)))

from_img = p2p.stimuli.images('sunflower.jpg', implant)


from pulse2percept.stimuli import BiphasicPulse

biphasic = BiphasicPulse(10, 0.78, interphase_dur=0.46, stim_dur=100)
biphasic.plot()

from pulse2percept.stimuli import Stimulus

stim = Stimulus({
    'A1': BiphasicPulse(-20, 1, stim_dur=75),
    'B2': BiphasicPulse(-20, 1, delay_dur=25, stim_dur=30)
})
stim.plot()