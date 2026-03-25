<form class="needs-validation" method="post" action="{{ route('barcode.setting') }}" novalidate>
    @csrf
    <div class="modal-body">
        <div class="row">
            <div class="form-group col-md-12">
                {{ Form::label('barcode_type', __('Barcode Type'), ['class' => 'form-label text-dark']) }}<x-required></x-required>
                {{ Form::select('barcode_type', ['code128' => 'Code 128', 'code39' => 'Code 39', 'code93' => 'Code 93'], !empty($settings['barcode_type']) ? $settings['barcode_type'] : '', ['class' => 'form-control', 'data-toggle' => 'select', 'required'=>'required']) }}
            </div>
            <div class="form-group col-md-12">
                {{ Form::label('barcode_format', __('Barcode Format'), ['class' => 'form-label text-dark']) }}<x-required></x-required>
                {{ Form::select('barcode_format', ['css' => 'CSS', 'bmp' => 'BMP'], !empty($settings['barcode_format']) ? $settings['barcode_format'] : '', ['class' => 'form-control', 'data-toggle' => 'select', 'required'=>'required']) }}
            </div>

        </div>
    </div>
    <div class="modal-footer">
        <input type="button" value="{{ __('Cancel') }}" class="btn  btn-light" data-bs-dismiss="modal">
        <input type="submit" value="{{__('Save Changes')}}" class="btn btn-primary">
    </div>
</form>
