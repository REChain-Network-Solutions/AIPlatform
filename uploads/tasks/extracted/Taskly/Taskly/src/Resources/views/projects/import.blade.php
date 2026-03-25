{{ Form::open(['method' => 'post', 'enctype' => 'multipart/form-data', 'id' => 'upload_form', 'class' => 'needs-validation', 'novalidate']) }}
<div class="modal-body">
    <div class="row">
        <div class="col-md-12 mb-6">
            {{ Form::label('file', __('Download Sample Project CSV File'), ['class' => 'col-form-label text-danger mx-1']) }}
            @if(check_file('uploads/sample/sample_project.csv'))
                <a href="{{ asset('uploads/sample/sample_project.csv') }}"
                    class="btn btn-sm btn-primary btn-icon-only">
                    <i class="fa fa-download"></i>
                </a>
            @endif
        </div>
        <div class="col-md-12 mt-1">
            {{ Form::label('file', __('Select CSV File'), ['class' => 'col-form-label']) }}<x-required></x-required>
            <div class="choose-file form-group">
                <label for="file" class="col-form-label">
                    <input type="file" class="form-control" name="file" id="file" data-filename="upload_file"
                        required>
                </label>
                <p class="upload_file"></p>
            </div>
        </div>
    </div>
</div>
<div class="modal-footer">
    <input type="button" value="{{ __('Cancel') }}" class="btn btn-secondary btn-light" data-bs-dismiss="modal">
    <button type="submit" value="{{ __('Upload') }}" class="btn btn-primary ms-2">
        {{__('Upload')}}
    </button>
    <a href="" data-url="{{ route('project.import.modal') }}" data-ajax-popup-over="true" title="{{ __('Create') }}" data-size="xl" data-title="{{ __('Import Project CSV Data') }}"  class="d-none import_modal_show"></a>
</div>
{{ Form::close() }}

<script>
    $('#upload_form').on('submit', function(event) {

        event.preventDefault();
        let data = new FormData(this);
        data.append('_token', "{{ csrf_token() }}");
        $.ajax({
            url: "{{ route('project.import') }}",
            method: "POST",
            data: data,
            dataType: 'json',
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                if (data.error != '')
                {
                    toastrs('Error',data.error, 'error');
                } else {
                    $('#commonModal').modal('hide');
                    $(".import_modal_show").trigger( "click" );
                    setTimeout(function() {
                        SetData(data.output);
                    }, 700);
                }
            }
        });

    });
</script>
