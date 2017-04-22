from moviepy.editor import ImageSequenceClip
import argparse
import re
from os import walk
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import csv

def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()
    mypath = args.image_folder
    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    #create new image file
    if(0):
    #    print (f)
        t_list={}
        with open("driving_log.csv","r") as fi:
            reader=csv.reader(fi)
            for row in reader:
                fn=re.sub(r'.*center', 'center',row[0])
                t_list[fn]=row[3]
                #print (fn, row[3])

        out_folder='Center_Img'
        for i in f:
            if(re.search(r'center', i)):
                fn=mypath+'/'+i
                img=Image.open(fn)
                draw=ImageDraw.Draw(img)
                #font=ImageFont.truetype("sans-serif.ttf",16)
                st=i+'--->>'+t_list[i]
                draw.text((0,0),st,(255,255,255))
                if(float(t_list[i])!=0):
                    img.save(out_folder+'/'+i)
    else:
        out_folder=mypath
    video_file = out_folder + '.mp4'
    
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(out_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
