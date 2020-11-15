from django.urls import path, include
from streamapp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('original/', views.original, name='original'),
    path('compressed/', views.compressed, name='compressed'),
    path('upscaled/', views.upscaled, name='upscaled'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('compressed_feed', views.compressed_feed, name='compressed_feed'),
    path('upscaled_feed', views.upscaled_feed, name='upscaled_feed')
    ]
