from fastai.vision.all import *

def is_samoyed(x):
    return 'samoyed' in x.lower() if isinstance(x, str) else False

path = untar_data(URLs.PETS)/'images'
# path = "images"
# print(path)

dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_samoyed,
    item_tfms=Resize(192))

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
# learn.export('model.pkl')

# For local machine
dls.show_batch()
plt.savefig("batch_display.png")
plt.show()
