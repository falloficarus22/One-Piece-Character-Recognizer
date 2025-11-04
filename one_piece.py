from fastai.vision.all import *
from icrawler.builtin import GoogleImageCrawler
from pathlib import Path
import time

# Folder in which we store our images
path = Path('one_piece_data')

# List of all characters to be easily iterable
characters = ['luffy', 'nami', 'sanji', 'zoro', 'usopp', 'robin', 'franky', 'brook', 'jinbe']
print(f'Prepared {len(characters)} for download')

# Download function
def download_images(character, max_images = 60):
    
    # Create subfolders for characters
    char_path = path / character
    char_path.mkdir(parents = True, exist_ok = True)

    try:
        # Google Image Crawler
        crawler = GoogleImageCrawler(storage = {'root_dir': str(char_path)})

        # Scraping web for images with optimised search queries
        crawler.crawl(keyword = f'{character} one piece anime', max_num = max_images, min_size = (200, 200))

        # Count downloaded images
        downloaded_images = list(char_path.glob('*.jpg')) + list(char_path.glob('*.png'))
        num_downloads = len(downloaded_images)

        print(f'Downloaded {num_downloads} images successfully for {character}')

        # Handle rate limiting
        time.sleep(2)

        return num_downloads
    
    except Exception as e:
        print(f'Error: {e}')
        return 0

# Downloading all characters    
print('Starting Download, this should take about 20 minutes depending your internet speed')

download_stats = {}

for c in characters:
    num_images = download_images(c, max_images = 60)
    download_stats[c] = num_images

# Download summary
total_images = 0
successful_chars = 0
failed_chars = []

for char, count in download_stats.items():
    status = "✓" if count > 0 else "✗"
    print(f"{status} {char:15} → {count:3} images")

    total_images += count

    if count > 0:
        successful_chars += 1
    else:
        failed_chars.append(char)

print(f'Total Images Downloaded: {total_images}')    
print(f'Total Successful Characters Downloaded: {successful_chars}')

if failed_chars:
    print(f"\n⚠ Failed Characters: {', '.join(failed_chars)}")
    print("You can retry these manually later")

# Data Cleaning
print('Checking for corrupted images...')

failed_images = verify_images(get_image_files(path))

if len(failed_images) > 0:
    print(f'Found {len(failed_images)} corrupt files')
    failed_images.map(Path.unlink)
    print('Removed all the corrupt images')
else:
    print('All images are valid')

# Final Check
for character in characters:
    char_path = path / character

    if char_path.exists():
        images = get_image_files(char_path)

# Training the model

dls = ImageDataLoaders.from_folder(path, valid_pct = 0.2, bs = 32, val_bs = 64,
    seed = 42, item_tfms = Resize(224), batch_tfms = aug_transforms(
        mult = 2.0,
        do_flip = True,
        flip_vert = False,
        max_rotate = 10.0,
        max_lighting = 0.2,
        max_warp = 0.2,
        p_affine = 0.75,
        p_lighting = 0.75
))

print('Dataloader successfully created')
print('Training the model...')

learn = vision_learner(dls, resnet34, metrics = accuracy)
learn.fine_tune(3)

print('Training complete now saving your model')
learn.export('one_piece.pkl')