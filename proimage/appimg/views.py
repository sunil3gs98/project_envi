from django.shortcuts import render

# Create your views here.

import cv2
import numpy as np
from exif import Image
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Image
from .serializers import ImageSerializer
from .serializers import ImageMetadata
from PIL import Image as PIL_Image
from io import BytesIO

from django.http import JsonResponse
from exif import Image
import io
import cv2


class ImageView(APIView):
    def post(self, request):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            std_dev = np.std(gray)
            contrast = std_dev / brightness
            # g = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(gray, 100, 200)
            edge_count = edges.sum()
            blur = edge_count / (gray.shape[0] * gray.shape[1])
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mean_v = np.mean(v)
            threshold = 100
            mask = np.where(v > threshold, 1, 0).astype(np.uint8)
            overexposed_ratio = np.sum(mask) / np.prod(mask.shape)

            if blur < 25.0 and overexposed_ratio <= 0.94 and brightness > 90 and contrast >= 0.22:
                # text = "Good Image"
                classification = "good"
            elif overexposed_ratio >= 0.95 and blur >= 21 and brightness > 90 and contrast < 0.27:
                # text = "Avg Image"
                classification = "average"
            else:
                # text = "Bad Image"
                classification = "bad"

            instance = serializer.save(classification=classification)

            # Return the classification results
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ImageData(APIView):
    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image=serializer.validated_data['image']
            image_bytes=image.file.read()
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            with io.BytesIO(image_bytes) as src:
                info = Image(src)
                try:
                    metadata = {}
                    metadata['make'] = info.make
                    metadata['software'] = info.software
                    metadata['_gps_ifd_pointer'] = info._gps_ifd_pointer
                    metadata['photographic_sensitivity'] = info.photographic_sensitivity
                    metadata['subject_distance_range'] = info.subject_distance_range
                    metadata['lens_specification'] = info.lens_specification
                    metadata['lens_make'] = info.lens_make
                    metadata['lens_model'] = info.lens_model
                    metadata['gps_version_id'] = info.gps_version_id
                    metadata['x_resolution'] = info.x_resolution
                    metadata['orientation'] = info.orientation

                    return JsonResponse(metadata)
                except:
                    return JsonResponse({'error': 'Some metadata are not avilable '})
        else:
            return JsonResponse({'error': 'Invalid request method'})


