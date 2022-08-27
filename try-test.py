#import required libraries
import cv2
import boto3
import os
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import matplotlib.pyplot as plt


#get access to laptop camera
#camera will be turned on
camera = cv2.VideoCapture(0)

#initialize number of iterations
i = 0

#set i to the number to which you want to take photos, i set it to 5, so 5 images will be taken and then program will be end
while i < 5:
    input('Press Enter to capture')

    #in loop captures the first image, then second and so on
    return_value, image = camera.read()

    #resize the images to 416 by 416 as it's my model requirement
    width= 416
    height= 416
    dsize = (width, height)
    output = cv2.resize(image, dsize)

    #save the resized image locally as image0, image 1,... image 5 (depends on value of i)
    #images are saved in same folder where this python file is
    cv2.imwrite('image'+str(i)+'.jpg', output)

    #consider for first image as 'image0' which is saved locally, this will now be saved in s3 bucket
    def upload_files(path):
        session = boto3.Session(
            #put your access key, secret key and region
            aws_access_key_id='YOUR ACCESS KEY',
            aws_secret_access_key='YOUR SECRET KEY',
            region_name='us-east-1'
    )
        #calling s3 buckets
        s3 = session.resource('s3')
        #replace the argument by name of bucket in which you want to put the image0
        bucket = s3.Bucket('images-for-detection')

        for subdir, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(subdir, file)
                with open(full_path, 'rb') as data:
                    bucket.put_object(Key=full_path[len(path)+1:], Body=data)


    #now to call rekognition model for inference
    def show_custom_labels(model,bucket,photo, min_confidence):


        client=boto3.client('rekognition')

    # Load image from S3 bucket
        s3_connection = boto3.resource('s3')

        s3_object = s3_connection.Object(bucket,photo)
        s3_response = s3_object.get()

        stream = io.BytesIO(s3_response['Body'].read())
        image=Image.open(stream)

        #Call DetectCustomLabels
        response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},MinConfidence=min_confidence,ProjectVersionArn=model)

        #gives height and width of resultant image
        imgWidth, imgHeight = image.size
        print(imgWidth)
        print(imgHeight)

        #plot and draw image0
        fig,ax = plt.subplots(1, 1, figsize=[8,8])

        draw = ImageDraw.Draw(image)

        #calculate and display bounding boxes for each detected custom label
        print('Detected custom labels for ' + photo)
        for customLabel in response['CustomLabels']:

            #print the label name along with confidence score
            print('Label ' + str(customLabel['Name']))
            print('Confidence ' + str(customLabel['Confidence']))

            if 'Geometry' in customLabel:
                box = customLabel['Geometry']['BoundingBox']
                left = imgWidth * box['Left']
                top = imgHeight * box['Top']
                width = imgWidth * box['Width']
                height = imgHeight * box['Height']

                #write name of label at top left of box
                draw.text((left,top), customLabel['Name'])

                #print geometry of box as json output
                print('Left: ' + '{0:.0f}'.format(left))
                print('Top: ' + '{0:.0f}'.format(top))
                print('Label Width: ' + "{0:.0f}".format(width))
                print('Label Height: ' + "{0:.0f}".format(height))

                points = ((left,top),(left + width, top),(left + width, top + height),(left , top + height),(left, top))

                #if label is a paper, draws a box with width of 2 and color corresponding to color code '#00d400'
                #if need thicker box, increase width, all labels have different color of boxes according to their color codes
                if str(customLabel['Name']) == 'paper':
                    draw.line(points, fill='#00d400', width=2)
                if str(customLabel['Name']) == 'cardboard':
                    draw.line(points, fill='#bc4100', width=2)
                if str(customLabel['Name']) == 'tissue':
                    draw.line(points, fill='#fffe1c', width=2)
                if str(customLabel['Name']) == 'can':
                    draw.line(points, fill='#221600', width=2)
                if str(customLabel['Name']) == 'bottle':
                    draw.line(points, fill='#4b5a00', width=2)
                if str(customLabel['Name']) == 'plastic_bag':
                    draw.line(points, fill='#ff32a7', width=2)
                if str(customLabel['Name']) == 'wrapper':
                    draw.line(points, fill='#0004bf', width=2)
                if str(customLabel['Name']) == 'trash':
                    draw.line(points, fill='#ffddfd', width=2)
                if str(customLabel['Name']) == 'wood':
                    draw.line(points, fill='#030030', width=2)
                if str(customLabel['Name']) == 'compost':
                    draw.line(points, fill='#cd4f39', width=2)

                #set axis as off and margins to zero
                ax.imshow(image)
                plt.axis("off")
                plt.margins(0,0)

                #save the augmented image locally as photo0 and rest as photo'i' in the same folder where this python file is located
                my_dpi=138
                plt.savefig('photo'+str(i)+'.jpg', transparent = True, bbox_inches='tight', pad_inches=0,dpi=my_dpi)
        return len(response['CustomLabels'])

    def main():

        #path to the folder from which laptop images are going to save in S3
        #i created a folder called test-1 on desktop which has this python file as demo-for-images.py
        #when laptop camera captures images, it resize it and store it in this folder as image0 and so on
        #when inference is done, these images are saved in this folder as photo0 and so on
        upload_files('/Users/durafsj/Desktop/test-1')

        #name of bucket where i have image for inference
        bucket="images-for-detection"

        #photo="image0.jpg"
        photo= 'image'+str(i)+'.jpg'


        #calling model with 10 classes by it's arn
        model='YOUR MODEL ARN'

        #set a high min confidence
        min_confidence=95

        label_count=show_custom_labels(model,bucket,photo, min_confidence)
        print("Custom labels detected: " + str(label_count))
        plt.show()

    #calling main function
    if __name__ == "__main__":
        main()

    #iterating the loop
    i += 1

#closing the camera
del(camera)

print('done')

#SUMMARY:
#For every iteration, camera takes a photo, store it locally as image0, upload it in S3
#Call the model, give image to do, do inference, and store infered image as photo0 locally
#it also displays the image for live demo
#this continues for the number of iterations you select
