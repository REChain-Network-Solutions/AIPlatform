@extends('layouts.app')

@push('datatable-styles')
    <!-- for sortable content -->
    <link rel="stylesheet" href="{{ asset('vendor/css/jquery-ui.css') }}">

    <!-- to highlight html content -->
    <link rel="stylesheet" href="{{ asset('vendor/css/default.min.css') }}">
@endpush

@section('content')
    <!-- CONTENT WRAPPER START -->
    <div class="content-wrapper">
        <div class="row">
            <div class="col-lg-6">
                <div class="row">
                    <div class="col-sm-12">
                        <div class="card bg-white border-0 b-shadow-4">
                            <div class="card-body ">
                                <div class="col-md-12 mb-3">
                                    <div class="row">
                                        <div class="col-md-3 f-w-500">#</div>
                                        <div class="col-md-5 f-w-500">@lang('app.fields')</div>
                                        <div class="col-md-4 f-w-500">@lang('app.status')</div>
                                    </div>
                                </div>
                                <div class="col-md-12">
                                    <x-form id="editSettings" method="PUT">
                                        <div id="sortable">
                                            @foreach ($ticketFormFields as $item)
                                                <div class="row py-3 pt-4 border-bottom">
                                                    <div class="col-md-3">
                                                        <span class="ui-icon ui-icon-arrowthick-2-n-s"></span>
                                                        <input type="hidden" name="sort_order[]"
                                                            value="{{ $item->id }}">
                                                    </div>
                                                    <div class="col-md-5">{{ mb_ucwords($item->field_display_name)}}</div>
                                                    <div class="col-md-4">
                                                        @if (!in_array($item->field_name, ['name', 'email', 'ticket_subject', 'message']))
                                                            <div class="custom-control custom-switch">
                                                                <input type="checkbox"
                                                                    class="custom-control-input change-setting"
                                                                    data-setting-id="{{ $item->id }}"
                                                                    @if ($item->status == 'active') checked @endif id="{{ $item->id }}">
                                                                <label class="custom-control-label f-14"
                                                                    for="{{ $item->id }}"></label>
                                                            </div>
                                                        @else
                                                            --
                                                        @endif
                                                    </div>
                                                </div>
                                            @endforeach

                                        </div>
                                    </x-form>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-sm-12 mt-4">
                        <x-cards.data>
                            <p class="f-w-500">@lang('modules.lead.iframeSnippet')</p>
                            <code>
                                &lt;iframe src="{{ route('front.ticket_form',company()->hash) }}" width="100%"
                                frameborder="0">&lt;/iframe&gt;
                            </code>
                        </x-cards.data>

                        <x-cards.data>
                            <p class="f-w-500">Share Direct link</p>
                            <p class="f-12"><a href="{{ route('front.ticket_form', [company()->hash]).'?styled=1' }}" target="_blank">{{ route('front.ticket_form', [company()->hash]).'?styled=1' }}</a></p>
                            <p class="f-12"><a href="{{ route('front.ticket_form', [company()->hash]).'?styled=1&with_logo=1' }}" target="_blank">{{ route('front.ticket_form', [company()->hash]).'?styled=1&with_logo=1' }}</a></p>
                        </x-cards.data>
                    </div>
                </div>
            </div>

            <div class="col-lg-6">
                <x-cards.data>
                    <h4>@lang('app.preview')</h4>
                    <iframe src="{{ route('front.ticket_form', company()->hash) }}" id="previewIframe" width="100%"
                        onload="resizeIframe(this)" frameborder="0"></iframe>
                </x-cards.data>
            </div>
        </div>

    </div>
    <!-- CONTENT WRAPPER END -->
@endsection

@push('scripts')
    <!-- for sortable content -->
    <script src="{{ asset('vendor/jquery/jquery-ui.min.js') }}"></script>

    <!-- to highlight html content -->
    <script src="{{ asset('vendor/jquery/highlight.min.js') }}"></script>

    <script>
        $(function() {
            $("#sortable").sortable({
                update: function(event, ui) {
                    var sortedValues = new Array();
                    $('input[name="sort_order[]"]').each(function(index, value) {
                        sortedValues[index] = $(this).val();
                    });
                    $.easyAjax({
                        url: "{{ route('ticket-form.sort_fields') }}",
                        type: "POST",
                        blockUI: true,
                        data: {
                            'sortedValues': sortedValues,
                            '_token': '{{ csrf_token() }}'
                        },
                        success: function(response) {
                            var iframe = document.getElementById('previewIframe');
                            iframe.src = iframe.src;
                        }
                    })
                }
            });
        });

        $('.change-setting').change(function() {
            var id = $(this).data('setting-id');
            var sendEmail = $(this).is(':checked') ? 'active' : 'inactive';

            var url = "{{ route('ticket-form.update', ':id') }}";
            url = url.replace(':id', id);
            $.easyAjax({
                url: url,
                type: "POST",
                blockUI: true,
                data: {
                    'id': id,
                    'status': sendEmail,
                    '_method': 'PUT',
                    '_token': '{{ csrf_token() }}'
                },
                success: function(response) {
                    var iframe = document.getElementById('previewIframe');
                    iframe.src = iframe.src;
                }
            })
        });

        function resizeIframe(obj) {
            obj.style.height = obj.contentWindow.document.documentElement.scrollHeight + 50 + 'px';
        }
    </script>
@endpush
