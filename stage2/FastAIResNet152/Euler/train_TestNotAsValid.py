import numpy as np
from fastai import *
from fastai.vision import *
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend ( 'agg' )


path = Path("./fastAI/train")

# Fix the Transformation only to vertical and horizontal transforms
tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_lighting=0.1,
                      max_rotate=0.0, 
                      max_zoom=0.0,
                      max_warp=0.0,
                      p_affine=0.0)

data = ImageDataBunch.from_folder(path, 
                                  valid_pct=0.1,
                                  ds_tfms= tfms,
                                  test='../test',
                                  size=512,
		                          bs=8).normalize()
#print(data)
print(data.batch_stats())
print(data.c)
print(data.classes)
# data.show_batch(rows=3, figsize=(10,10))

learn = cnn_learner(data, models.resnet152, pretrained=False,metrics=accuracy)
learn.summary()
learn.unfreeze()
#learn.freeze_to(10)
learn.fit_one_cycle(200,max_lr=slice(1e-6,1e-4))

interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
interp.plot_top_losses(9, figsize=(10,10))
interp.plot_confusion_matrix()
img = learn.data.test_ds[0][0]
learn.predict(img)

# Save the finetuned model
learn.save('res152')
