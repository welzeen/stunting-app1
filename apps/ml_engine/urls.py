from django.urls import path
from . import views

urlpatterns = [
    path('', views.ml_dashboard_view, name='ml_dashboard'),
    path('train/', views.train_model_view, name='train_model'),
    path('prediksi/', views.prediksi_view, name='prediksi'),
    path('evaluasi/', views.evaluasi_view, name='evaluasi'),
    path('evaluasi/export-pdf/', views.export_pdf_view, name='export_pdf'),
    path('batch-predict/', views.batch_predict_view, name='batch_predict'),
]
