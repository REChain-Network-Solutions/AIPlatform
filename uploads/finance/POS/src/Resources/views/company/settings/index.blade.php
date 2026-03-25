<div class="card" id="pos-sidenav">
    {{ Form::open(['route' => 'pos.setting.store']) }}
    <div class="card-header p-3">
        <h5 class="">{{ __('POS Settings') }}</h5>
    </div>
    <div class="card-body p-3 pb-0">
        <div class="row">
            <div class="col-md-6">
                <div class="form-group">
                    <label for="low_product_stock_threshold"
                        class="form-label">{{ __('Low Product Stock Threshold') }}</label>
                    <input type="number" name="low_product_stock_threshold" class="form-control"
                        placeholder="{{ __('Low Product Stock Threshold') }}"
                        value="{{ !empty($settings['low_product_stock_threshold']) ? $settings['low_product_stock_threshold'] : '' }}"
                        id="low_product_stock_threshold">
                </div>
            </div>
        </div>
    </div>
    <div class="card-footer text-end p-3">
        <input class="btn btn-print-invoice  btn-primary" type="submit" value="{{ __('Save Changes') }}">
    </div>
    {{ Form::close() }}
</div>


<div id="pos-print-sidenav" class="card">
    <div class="card-header p-3">
        <h5>{{ __('Pos Print Settings') }}</h5>
        <small class="text-muted">{{ __('Edit details about your Company Bill') }}</small>
    </div>
    <div class="bg-none">
        <div class="row company-setting">
            <form id="setting-form" method="post" action="{{ route('pos.template.setting') }}"
                enctype="multipart/form-data">
                @csrf
                <div class="card-body border-bottom border-1 p-3 pb-0">
                    <div class="row">
                        <div class="col-sm-4">
                            <div class="form-group">
                                {{ Form::label('pos_prefix', __('Prefix'), ['class' => 'form-label']) }}
                                {{ Form::text('pos_prefix', isset($settings['pos_prefix']) && !empty($settings['pos_prefix']) ? $settings['pos_prefix'] : '#PUR', ['class' => 'form-control', 'placeholder' => __('Enter Pos Prefix')]) }}
                            </div>
                        </div>
                        <div class="col-sm-4">
                            <div class="form-group">
                                {{ Form::label('pos_footer_title', __('Footer Title'), ['class' => 'form-label']) }}
                                {{ Form::text('pos_footer_title', isset($settings['pos_footer_title']) && !empty($settings['pos_footer_title']) ? $settings['pos_footer_title'] : '', ['class' => 'form-control', 'placeholder' => __('Enter Footer Title')]) }}
                            </div>
                        </div>
                        <div class="col-xxl-8">
                            <div class="form-group">
                                {{ Form::label('pos_footer_notes', __('Footer Notes'), ['class' => 'form-label']) }}
                                {{ Form::textarea('pos_footer_notes', isset($settings['pos_footer_notes']) && !empty($settings['pos_footer_notes']) ? $settings['pos_footer_notes'] : '', ['class' => 'form-control', 'rows' => '2', 'placeholder' => __('Enter Pos Footer Notes')]) }}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card-body p-3">
                    <div class="row row-gap">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body p-2">
                                    <div class="form-group d-flex align-items-center justify-content-between mb-0">
                                        {{ Form::label('pos_shipping_display', __('Shipping Display?'), ['class' => 'form-label mb-0']) }}
                                        <div class=" form-switch form-switch-left">
                                            <input type="checkbox" class="form-check-input" name="pos_shipping_display"
                                                id="pos_shipping_display"
                                                {{ isset($settings['pos_shipping_display']) && $settings['pos_shipping_display'] == 'on' ? 'checked' : '' }}>
                                            <label class="form-check-label" for="pos_shipping_display"></label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-body p-2">
                                    <div class="form-group d-flex flex-wrap align-items-center gap-2 mb-0">
                                        <label for="address" class="form-label mb-0">{{ __('POS Template') }}</label>
                                        <select class="form-control flex-1" name="pos_template">
                                            @foreach (Workdo\Pos\Entities\Pos::templateData()['templates'] as $key => $template)
                                                <option value="{{ $key }}"
                                                    {{ isset($settings['pos_template']) && !empty($settings['pos_template']) && $settings['pos_template'] == $key ? 'selected' : '' }}>
                                                    {{ $template }}</option>
                                            @endforeach
                                        </select>
                                    </div>
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-header p-2">
                                    <h6 class="mb-0">{{ __('Color Input') }}</h6>
                                </div>
                                <div class="card-body p-2">
                                    @foreach (Workdo\Pos\Entities\Pos::templateData()['colors'] as $key => $color)
                                        <label class="colorinput">
                                            <input name="pos_color" type="radio" value="{{ $color }}"
                                                class="colorinput-input"
                                                {{ !empty($settings['pos_color']) && $settings['pos_color'] == $color ? 'checked' : '' }}>
                                            <span class="colorinput-color rounded-circle"
                                                style="background: #{{ $color }}"></span>
                                        </label>
                                @endforeach
                                </div>
                            </div>
                            <div class="card">
                                <div class="card-header p-2">
                                    <h3 class="h6 mb-0">{{ __('Logo')}}</h3>
                                </div>
                                <div class="card-body setting-card setting-logo-box p-3">
                                    <div class="logo-content img-fluid logo-set-bg  text-center">
                                        <img alt="image" src="{{ isset($settings['pos_logo']) ? get_file($settings['pos_logo']) : get_file('uploads/logo/logo_dark.png') }}" id="pre_pos_logo">
                                    </div>
                                    <div class="choose-files text-center  mt-3">
                                        <label for="blah8">
                                            <div class="bg-primary"> <i class="ti ti-upload px-1"></i>{{ __('Choose file here')}}</div>
                                            <input type="file" class="form-control file" name="pos_logo" id="blah8" data-filename="blah8" onchange="document.getElementById('pre_pos_logo').src = window.URL.createObjectURL(this.files[0])">
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <div class="form-group pt-4 mb-0 text-left">
                                <input type="submit" value="{{ __('Save Changes') }}"
                                    class="btn btn-print-invoice  btn-primary">
                            </div>
                        </div>
                        <div class="col-md-8">
                            @if (isset($settings['pos_template']) &&
                                    isset($settings['pos_color']) &&
                                    !empty($settings['pos_template']) &&
                                    !empty($settings['pos_color']))
                                <iframe id="pos_frame" class="w-100 h-100 rounded-1" frameborder="0"
                                    src="{{ route('pos.preview', [$settings['pos_template'], $settings['pos_color']]) }}"></iframe>
                            @else
                                <iframe id="pos_frame" class="w-100 h-100 rounded-1" frameborder="0"
                                    src="{{ route('pos.preview', ['template1 ', 'fffff']) }}"></iframe>
                            @endif
                        </div>
                    </div>
                </div>
            </form>
        </div>
    </div>
</div>


<script>
    $(document).on("change", "select[name='pos_template'], input[name='pos_color']", function() {

        var template = $("select[name='pos_template']").val();
        var color = $("input[name='pos_color']:checked").val();
        $('#pos_frame').attr('src', '{{ url('/pos/preview') }}/' + template + '/' + color);
    });

    $(document).on("change", "select[name='purchase_template'], input[name='purchase_color']", function() {
        var template = $("select[name='purchase_template']").val();
        var color = $("input[name='purchase_color']:checked").val();
        $('#purchase_frame').attr('src', '{{ url('/purchase/preview') }}/' + template + '/' + color);
    });

    document.getElementById('purchase_logo').onchange = function() {
        var src = URL.createObjectURL(this.files[0])
        document.getElementById('purchase_image').src = src
    }


    document.getElementById('pos_logo').onchange = function() {
        var src = URL.createObjectURL(this.files[0])
        document.getElementById('pos_image').src = src
    }
</script>
