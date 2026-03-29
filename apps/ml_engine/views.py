from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from apps.balita.models import Balita, PrediksiHasil
from .models import ModelNaiveBayes
from .forms import PrediksiForm
from . import naive_bayes_service as nbs
import json


@login_required
def ml_dashboard_view(request):
    model_obj = ModelNaiveBayes.objects.order_by('-trained_at').first()
    training_count = Balita.objects.filter(dataset_type='training').exclude(status_stunting__isnull=True).count()
    testing_count = Balita.objects.filter(dataset_type='testing').exclude(status_stunting__isnull=True).count()
    
    context = {
        'model_obj': model_obj,
        'training_count': training_count,
        'testing_count': testing_count,
    }
    return render(request, 'ml/dashboard.html', context)


@login_required
def train_model_view(request):
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat melatih model.')
        return redirect('ml_dashboard')

    if request.method == 'POST':
        try:
            training_qs = Balita.objects.filter(
                dataset_type='training'
            ).exclude(status_stunting__isnull=True).exclude(status_stunting='')

            if training_qs.count() < 10:
                messages.error(request, 'Data training terlalu sedikit (minimal 10 data).')
                return redirect('ml_dashboard')

            model, encoders, X_train, y_train = nbs.train_model(training_qs)

            testing_qs = Balita.objects.filter(
                dataset_type='testing'
            ).exclude(status_stunting__isnull=True).exclude(status_stunting='')

            metrics = None
            if testing_qs.count() > 0:
                metrics = nbs.evaluate_model(model, encoders, testing_qs)

            model_obj, _ = ModelNaiveBayes.objects.get_or_create(id=1)
            model_obj.nama_model = 'Naive Bayes - Klasifikasi Stunting Balita'
            model_obj.jumlah_training = training_qs.count()
            model_obj.status = 'trained'
            model_obj.trained_by = request.user

            if metrics:
                model_obj.akurasi = metrics['akurasi']
                model_obj.presisi = metrics['presisi']
                model_obj.recall = metrics['recall']
                model_obj.f1_score = metrics['f1_score']
                model_obj.jumlah_testing = metrics['jumlah_testing']
                model_obj.confusion_matrix_data = json.dumps(metrics['confusion_matrix'])
                model_obj.classification_report = json.dumps(metrics['classification_report'])

            model_obj.save()
            messages.success(request,
                f'Model berhasil dilatih dengan {training_qs.count()} data training. '
                + (f'Akurasi: {metrics["akurasi"]:.2%}' if metrics else 'Evaluasi memerlukan data testing.'))

        except Exception as e:
            messages.error(request, f'Gagal melatih model: {str(e)}')

    return redirect('ml_dashboard')


@login_required
def prediksi_view(request):
    form = PrediksiForm(request.POST or None)
    result = None

    if request.method == 'POST' and form.is_valid():
        input_data = form.cleaned_data
        try:
            predicted_class, proba_dict = nbs.predict_single(input_data)
            result = {
                'kelas': predicted_class,
                'probabilitas': proba_dict,
                'input': input_data,
            }
            # Save to history
            PrediksiHasil.objects.create(
                jenis_kelamin=input_data['jenis_kelamin'],
                umur=input_data['umur'],
                berat_badan=0,
                tinggi_badan=0,
                status_bbu=input_data['status_bbu'],
                status_tbu=input_data['status_tbu'],
                status_gizi=input_data['status_gizi'],
                status_asi=input_data['status_asi'],
                hasil_prediksi=predicted_class,
                probabilitas_tidak=proba_dict.get('Tidak', 0),
                probabilitas_potensi=proba_dict.get('Potensi Stunting', 0),
                probabilitas_stunting=proba_dict.get('Stunting', 0),
                predicted_by=request.user,
            )
        except ValueError as e:
            messages.error(request, str(e))

    return render(request, 'ml/prediksi.html', {'form': form, 'result': result})


@login_required
def evaluasi_view(request):
    model_obj = ModelNaiveBayes.objects.order_by('-trained_at').first()
    if not model_obj or model_obj.status != 'trained':
        messages.warning(request, 'Model belum dilatih.')
        return redirect('ml_dashboard')

    cm_data = model_obj.get_confusion_matrix()
    cr_data = model_obj.get_classification_report()

    kelas_list = ['Tidak', 'Potensi Stunting', 'Stunting']
    context = {
        'model_obj': model_obj,
        'cm_data': cm_data,
        'cr_data': cr_data,
        'cm_json': json.dumps(cm_data),
        'kelas_list': kelas_list,
    }
    return render(request, 'ml/evaluasi.html', context)


@login_required  
def batch_predict_view(request):
    """Predict all testing data that have no label prediction yet"""
    if not (hasattr(request.user, 'profile') and request.user.profile.is_admin()):
        messages.error(request, 'Hanya admin yang dapat menjalankan batch prediksi.')
        return redirect('ml_dashboard')

    if request.method == 'POST':
        try:
            testing_qs = Balita.objects.filter(dataset_type='testing')
            count = 0
            for balita in testing_qs:
                input_data = {
                    'jenis_kelamin': balita.jenis_kelamin,
                    'umur': balita.umur,
                    'status_bbu': balita.status_bbu,
                    'status_tbu': balita.status_tbu,
                    'status_gizi': balita.status_gizi,
                    'status_asi': balita.status_asi,
                }
                try:
                    predicted_class, proba_dict = nbs.predict_single(input_data)
                    PrediksiHasil.objects.create(
                        balita=balita,
                        jenis_kelamin=balita.jenis_kelamin,
                        umur=balita.umur,
                        berat_badan=balita.berat_badan,
                        tinggi_badan=balita.tinggi_badan,
                        status_bbu=balita.status_bbu,
                        status_tbu=balita.status_tbu,
                        status_gizi=balita.status_gizi,
                        status_asi=balita.status_asi,
                        hasil_prediksi=predicted_class,
                        probabilitas_tidak=proba_dict.get('Tidak', 0),
                        probabilitas_potensi=proba_dict.get('Potensi Stunting', 0),
                        probabilitas_stunting=proba_dict.get('Stunting', 0),
                        predicted_by=request.user,
                    )
                    count += 1
                except Exception:
                    pass
            messages.success(request, f'Batch prediksi selesai: {count} data diproses.')
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')

    return redirect('ml_dashboard')


@login_required
def export_pdf_view(request):
    """Export laporan evaluasi model sebagai PDF."""
    from datetime import datetime

    model_obj = ModelNaiveBayes.objects.order_by('-trained_at').first()
    if not model_obj or model_obj.status != 'trained':
        messages.warning(request, 'Model belum dilatih. Latih model terlebih dahulu sebelum export PDF.')
        return redirect('ml_dashboard')

    try:
        from .pdf_report import generate_evaluation_pdf

        # Ambil data testing
        testing_qs = Balita.objects.filter(
            dataset_type='testing'
        ).exclude(status_stunting__isnull=True).exclude(status_stunting='').order_by('kode_balita')

        if not testing_qs.exists():
            messages.warning(request, 'Tidak ada data testing untuk di-export.')
            return redirect('evaluasi')

        # Buat prediksi untuk semua data testing
        predicted_map = {}
        for balita in testing_qs:
            input_data = {
                'jenis_kelamin': balita.jenis_kelamin,
                'umur':          balita.umur,
                'status_bbu':    balita.status_bbu,
                'status_tbu':    balita.status_tbu,
                'status_gizi':   balita.status_gizi,
                'status_asi':    balita.status_asi,
            }
            try:
                predicted_class, _ = nbs.predict_single(input_data)
                predicted_map[balita.kode_balita] = predicted_class
            except Exception:
                predicted_map[balita.kode_balita] = '-'

        # Generate PDF
        pdf_buf = generate_evaluation_pdf(model_obj, testing_qs, predicted_map)

        filename = f"laporan_evaluasi_stunting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response = HttpResponse(pdf_buf, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        messages.error(request, f'Gagal membuat PDF: {str(e)}')
        return redirect('evaluasi')
